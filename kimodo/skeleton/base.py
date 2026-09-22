# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Base skeleton class: hierarchy, joint metadata, and helpers for kinematics and motion."""

from pathlib import Path
from typing import Optional, Union

import torch

from kimodo.assets import skeleton_asset_path

from .kinematics import fk
from .transforms import (
    from_standard_tpose,
    global_rots_to_local_rots,
    to_standard_tpose,
)


def _resolve_to_target(
    args: tuple,
    kwargs: dict,
) -> tuple[Optional[Union[str, torch.device]], Optional[torch.dtype]]:
    device = kwargs.get("device")
    dtype = kwargs.get("dtype")

    if args:
        first = args[0]
        if isinstance(first, torch.dtype):
            dtype = first
        elif torch.is_tensor(first):
            device = first.device
            dtype = first.dtype
        else:
            device = first
        if len(args) > 1 and isinstance(args[1], torch.dtype):
            dtype = args[1]

    return device, dtype


class SkeletonBase(torch.nn.Module):
    """Base class that stores a skeleton hierarchy and helper metadata.

    Subclasses define the static joint layout (joint names and parent links) and semantic groups
    (feet, hands, hips). This class builds index mappings, parent tensors, and convenience helpers
    used by kinematics, constraints, and motion conversion utilities.
    """

    # these should be defined in the subclass
    name = None
    bone_order_names_with_parents = None
    bone_order_names_no_root = None
    root_idx = None
    foot_joint_names = None
    foot_joint_idx = None
    hip_joint_names = None  # in order [right, left]
    hip_joint_idx = None  # in order [right, left]

    def to(self, *args, **kwargs):
        """Move the skeleton while keeping MPS-safe floating dtypes."""

        device, dtype = _resolve_to_target(args, kwargs)
        if device is None or not str(device).lower().startswith("mps"):
            return super().to(*args, **kwargs)

        target_device = torch.device(device)
        target_dtype = torch.float32 if dtype == torch.float64 else dtype

        def _convert(tensor: torch.Tensor) -> torch.Tensor:
            if not isinstance(tensor, torch.Tensor):
                return tensor
            if tensor.is_floating_point():
                dtype_for_tensor = target_dtype
                if dtype_for_tensor is None:
                    dtype_for_tensor = torch.float32 if tensor.dtype == torch.float64 else tensor.dtype
                return tensor.to(device=target_device, dtype=dtype_for_tensor)
            return tensor.to(device=target_device)

        self._apply(_convert)
        return self

    def __init__(
        self,
        folder: Optional[str] = None,
        name: Optional[str] = None,
        load: bool = True,
        **kwargs,  # to catch addition args in configs
    ):
        """Initialize a skeleton instance and optional neutral-pose assets.

        Args:
            folder: Folder containing serialized skeleton assets (for example
                `joints.p` and optional `standard_t_pose_global_offsets_rots.p`).
            name: Optional runtime name used to validate subclass compatibility.
            load: Whether to load tensor assets from `folder`.
            **kwargs: Unused extra config keys kept for config compatibility.
        """
        super().__init__()

        if name is not None:
            # Check that the name is not too far from the actual skeleton class name
            assert self.name in name
            self.name = name

        if folder is None:
            # Take the skeleton asset folder of the repo from the name
            # in case we don't override it
            folder = str(skeleton_asset_path(self.name))

        self.folder = folder

        self.dim = len(self.bone_order_names_with_parents)

        if load and folder is not None:
            pfolder = Path(folder)
            neutral_joints = torch.load(pfolder / "joints.p").squeeze()
            self.register_buffer("neutral_joints", neutral_joints, persistent=False)

            if (pfolder / "bvh_joints.p").exists():
                bvh_neutral_joints = torch.load(pfolder / "bvh_joints.p").squeeze()
                self.register_buffer("bvh_neutral_joints", bvh_neutral_joints, persistent=False)

            global_offset_path = pfolder / "standard_t_pose_global_offsets_rots.p"
            if global_offset_path.exists():
                global_rot_offsets = torch.load(global_offset_path).squeeze()
                self.register_buffer("global_rot_offsets", global_rot_offsets, persistent=False)
            # Usefull for g1, where the rest pose is not zero
            baked_rest_path = pfolder / "rest_pose_local_rot.p"
            if baked_rest_path.exists():
                rest_pose_local_rot = torch.load(baked_rest_path).squeeze()
                self.register_buffer("rest_pose_local_rot", rest_pose_local_rot, persistent=False)

        self.bone_order_names = [x for x, y in self.bone_order_names_with_parents]

        self.bone_parents = dict(self.bone_order_names_with_parents)
        self.bone_index = {x: idx for idx, x in enumerate(self.bone_order_names)}
        self.bone_order_names_index = self.bone_index

        # create the parents tensor on the fly
        joint_parents = torch.tensor(
            [-1 if (y := self.bone_parents[x]) is None else self.bone_index[y] for x in self.bone_order_names]
        )
        self.register_buffer("joint_parents", joint_parents, persistent=False)

        self.nbjoints = len(self.bone_order_names)

        # check lengths
        assert self.nbjoints == len(self.joint_parents)
        if "neutral_joints" in self.__dict__:
            assert self.nbjoints == len(self.neutral_joints)

        root_indices = torch.where(joint_parents == -1)[0]
        assert len(root_indices) == 1  # should be one root only
        self.root_idx = root_indices[0].item()

        if "neutral_joints" in self.__dict__:
            assert (self.neutral_joints[0] == 0).all()

        # remove the root
        self.bone_order_names_no_root = (
            self.bone_order_names[: self.root_idx] + self.bone_order_names[self.root_idx + 1 :]
        )

        self.foot_joint_names = self.left_foot_joint_names + self.right_foot_joint_names
        self.foot_joint_names_index = {x: idx for idx, x in enumerate(self.foot_joint_names)}

        self.left_foot_joint_idx = [
            self.bone_order_names.index(foot_joint) for foot_joint in self.left_foot_joint_names
        ]

        self.right_foot_joint_idx = [
            self.bone_order_names.index(foot_joint) for foot_joint in self.right_foot_joint_names
        ]

        self.foot_joint_idx = self.left_foot_joint_idx + self.right_foot_joint_idx

        self.hip_joint_idx = [self.bone_order_names.index(hip_joint) for hip_joint in self.hip_joint_names]

    def expand_joint_names(self, joint_names):
        """Expand base EE names [LeftFoot, RightFoot, LeftHand, RightHand] actual joint names to
        constrain position and rotations.

        Args:
            joint_names: list of list of base EE names to constrain

        Returns:
            rot_joint_names: list of list of joint names to constrain rotations
            pos_joint_names: list of list of joint names to constrain positions
        """

        base_ee = ["LeftFoot", "RightFoot", "LeftHand", "RightHand", "Hips"]

        pelvis_name = self.bone_order_names[self.root_idx]

        base_pos_names = [
            self.left_foot_joint_names,
            self.right_foot_joint_names,
            self.left_hand_joint_names,
            self.right_hand_joint_names,
            [pelvis_name],
        ]
        # base of each chain
        base_rot_names = [
            self.left_foot_joint_names[:1],
            self.right_foot_joint_names[:1],
            self.left_hand_joint_names[:1],
            self.right_hand_joint_names[:1],
            [pelvis_name],
        ]
        rot_joint_names = []
        pos_joint_names = []
        # loop through each EE joint group to constrain in the current keyframe
        for jname in joint_names:
            idx = base_ee.index(jname)
            rot_joint_names += base_rot_names[idx]
            pos_joint_names += base_pos_names[idx]
        return rot_joint_names, pos_joint_names

    def expand_joint_names_batched(self, joint_names):
        """Expand base EE names [LeftFoot, RightFoot, LeftHand, RightHand] actual joint names to
        constrain position and rotations.

        Args:
            joint_names: list of list of base EE names to constrain

        Returns:
            rot_joint_names: list of list of joint names to constrain rotations
            pos_joint_names: list of list of joint names to constrain positions
        """

        base_ee = ["LeftFoot", "RightFoot", "LeftHand", "RightHand", "Hips"]

        pelvis_name = self.bone_order_names[self.root_idx]

        base_pos_names = [
            self.left_foot_joint_names,
            self.right_foot_joint_names,
            self.left_hand_joint_names,
            self.right_hand_joint_names,
            [pelvis_name],
        ]
        # base of each chain
        base_rot_names = [
            self.left_foot_joint_names[:1],
            self.right_foot_joint_names[:1],
            self.left_hand_joint_names[:1],
            self.right_hand_joint_names[:1],
            [pelvis_name],
        ]
        # loop through each keyframe
        rot_joint_names = []
        pos_joint_names = []
        for key_joint_names in joint_names:
            key_rot_names = []
            key_pos_names = []
            # loop through each EE joint group to constrain in the current keyframe
            for jname in key_joint_names:
                idx = base_ee.index(jname)
                key_rot_names += base_rot_names[idx]
                key_pos_names += base_pos_names[idx]
            rot_joint_names.append(key_rot_names)
            pos_joint_names.append(key_pos_names)
        return rot_joint_names, pos_joint_names

    def __repr__(self):
        if self.folder is None:
            return f"{self.__class__.__name__}()"
        return f'{self.__class__.__name__}(folder="{self.folder}")'

    @property
    def device(self):
        """Device where neutral-joint buffers are stored.

        Returns 'cpu' if neutral_joints is not present.
        """
        if getattr(self, "neutral_joints", None) is None:
            return "cpu"
        return self.neutral_joints.device

    def fk(self, local_joint_rots: torch.Tensor, root_positions: torch.Tensor):
        """Run forward kinematics for this skeleton layout.

        Args:
            local_joint_rots: Local joint rotation matrices with shape
                `(..., J, 3, 3)`.
            root_positions: Root translations with shape `(..., 3)`.

        Returns:
            Tuple of `(global_joint_rots, posed_joints, posed_joints_norootpos)`.
        """
        global_joint_rots, posed_joints, posed_joints_norootpos = fk(local_joint_rots, root_positions, self)
        return global_joint_rots, posed_joints, posed_joints_norootpos

    def to_standard_tpose(self, local_rot_mats: torch.Tensor):
        """Convert local rotations into the skeleton's standard T-pose frame."""
        return to_standard_tpose(local_rot_mats, self)

    def from_standard_tpose(self, local_rot_mats: torch.Tensor):
        """Convert local rotations from the skeleton's standard T-pose frame."""
        return from_standard_tpose(local_rot_mats, self)

    def global_rots_to_local_rots(self, global_joint_rots: torch.Tensor):
        """Convert global joint rotations to local rotations for this hierarchy."""
        return global_rots_to_local_rots(global_joint_rots, self)

    def get_skel_slice(self, skeleton: "SkeletonBase"):
        """Build index mapping from another skeleton into this skeleton order.

        Args:
            skeleton: Source skeleton whose joint order is used by input tensors.

        Returns:
            A list of source indices ordered as `self.bone_order_names`.

        Raises:
            ValueError: If at least one required joint is missing from `skeleton`.
        """
        try:
            skel_slice = [skeleton.bone_index[x] for x in self.bone_order_names]
        except KeyError:
            raise ValueError("The current skeleton contain joints that are not in the input")
        return skel_slice
