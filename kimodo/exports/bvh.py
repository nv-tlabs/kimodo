# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Export utilities for converting internal motion representations into common file formats.

This module is intended to hold lightweight serialization / export helpers that can be reused
outside of interactive demos.
"""

from pathlib import Path
from typing import Tuple, Union

import numpy as np
import torch
from scipy.spatial.transform import Rotation


def _coerce_batch(name: str, x: torch.Tensor, *, expected_ndim: int) -> torch.Tensor:
    """Coerce (T, ...) or (1, T, ...) into (T, ...)."""
    if x.ndim == expected_ndim:
        return x
    if x.ndim == expected_ndim + 1:
        if int(x.shape[0]) != 1:
            raise ValueError(
                f"{name} has batch dimension B={int(x.shape[0])}, but BVH export " "only supports a single clip (B==1)."
            )
        return x[0]
    raise ValueError(f"{name} must have shape (T, ...) or (1, T, ...); got {tuple(x.shape)}")


def _rotation_matrices_to_bvh_eulers(local_rot_mats: np.ndarray) -> np.ndarray:
    """Convert local rotation matrices to BVH Z/Y/X channel angles in degrees."""
    nframes, njoints = local_rot_mats.shape[:2]
    euler_rad = Rotation.from_matrix(local_rot_mats.reshape(-1, 3, 3)).as_euler("ZYX")
    euler_rad = euler_rad.reshape(nframes, njoints, 3)
    # Keep common +/-pi channel wraps continuous. Euler branch choices can still
    # flip coupled axes near gimbal lock, but the represented rotations are unchanged.
    return np.rad2deg(np.unwrap(euler_rad, axis=0))


def _prepare_bvh_rest_pose(
    local_rot_mats: torch.Tensor,
    skeleton,
    *,
    standard_tpose: bool,
) -> Tuple[torch.Tensor, np.ndarray]:
    if standard_tpose:
        if not hasattr(skeleton, "neutral_joints"):
            raise ValueError(f"BVH export requires neutral_joints on skeleton {skeleton.name!r}.")
        return local_rot_mats, skeleton.neutral_joints.detach().cpu().numpy()

    if not hasattr(skeleton, "global_rot_offsets") or not hasattr(skeleton, "bvh_neutral_joints"):
        raise ValueError(
            "BVH export with standard_tpose=False requires skeleton-specific BVH rest-pose assets "
            f"(global_rot_offsets and bvh_neutral_joints); skeleton {skeleton.name!r} does not provide them. "
            "Use standard_tpose=True for this skeleton."
        )

    local_rot_mats, _ = skeleton.from_standard_tpose(local_rot_mats)
    return local_rot_mats, skeleton.bvh_neutral_joints.detach().cpu().numpy()


def motion_to_bvh(
    local_rot_mats: torch.Tensor,
    root_positions: torch.Tensor,
    *,
    skeleton,
    fps: float,
    standard_tpose: bool = False,
) -> str:
    """Convert local rotations and root positions to BVH format; return UTF-8 string.

    Args:
        local_rot_mats: (T, J, 3, 3) or (1, T, J, 3, 3) local rotation matrices.
        root_positions: (T, 3) or (1, T, 3) root joint positions (e.g. from posed joints).
        skeleton: Skeleton with bone_order_names, joint_parents, and neutral_joints.
        fps: Frames per second for the motion.
        standard_tpose: If True, export with the skeleton's standard neutral pose as the BVH rest pose.
            If False, export with the skeleton-specific BVH rest pose, e.g. the BONES-SEED-compatible
            rest pose for SOMA. This mode requires ``global_rot_offsets`` and ``bvh_neutral_joints``.
    Notes:
        BVH is plain-text. Root is named "Root" with ZYX rotation order; leaf joints
        have no End Site blocks to match the source BVH convention.
    """
    local_rot_mats = local_rot_mats.detach()
    root_positions = root_positions.detach()
    # SOMA: accept either somaskel30 (convert to 77) or somaskel77 (use as-is)
    if skeleton.name == "somaskel30":
        local_rot_mats = skeleton.to_SOMASkeleton77(local_rot_mats)
        skeleton = skeleton.somaskel77

    local_rot_mats, neutral = _prepare_bvh_rest_pose(local_rot_mats, skeleton, standard_tpose=standard_tpose)

    joint_names = list(skeleton.bone_order_names)
    parents = skeleton.joint_parents.detach().cpu().numpy().astype(int)
    root_idx = int(skeleton.root_idx)

    local_rot_mats = _coerce_batch("local_rot_mats", local_rot_mats, expected_ndim=4)
    T, J = local_rot_mats.shape[:2]
    if J != len(joint_names):
        raise ValueError(f"local_rot_mats has {J} joints but skeleton {skeleton.name!r} has {len(joint_names)} joints.")
    local_eulers = _rotation_matrices_to_bvh_eulers(local_rot_mats.detach().cpu().numpy())

    root_xyz = _coerce_batch("root_positions", root_positions, expected_ndim=2)
    root_xyz = root_xyz.cpu().numpy()  # [T, 3]

    # Build BVH hierarchy: Root wrapper at origin -> skeleton root -> skeleton joints.
    children: dict[int, list[int]] = {i: [] for i in range(J)}
    for i, p in enumerate(parents):
        if p >= 0:
            children[int(p)].append(int(i))

    _ROOT_CHANNELS = [
        "Xposition",
        "Yposition",
        "Zposition",
        "Zrotation",
        "Yrotation",
        "Xrotation",
    ]
    _JOINT_CHANNELS = ["Zrotation", "Yrotation", "Xrotation"]

    # BVH writes offsets and root motion in centimeters, matching the SEED data scale.
    neutral = neutral * 100
    root_xyz = root_xyz * 100

    # Hips offset from Root: use skeleton neutral; if root is at origin (zeros), use a
    # nominal pelvis height so the hierarchy is non-degenerate in Blender.
    hips_offset = neutral[root_idx]
    if (hips_offset == 0).all():
        hips_offset = np.array([0.0, 100.0, 0.0], dtype=neutral.dtype)  # 1 m in cm

    bvh_lines = ["HIERARCHY", "ROOT Root", "{", "  OFFSET 0 0 0"]
    bvh_lines.append("  CHANNELS 6 " + " ".join(_ROOT_CHANNELS))
    ordered_joint_ids = []

    def _write_joint(i: int, indent: int) -> None:
        prefix = "  " * indent
        ordered_joint_ids.append(i)
        bvh_lines.append(f"{prefix}JOINT {joint_names[i]}")
        bvh_lines.append(f"{prefix}{{")
        if i == root_idx:
            off = hips_offset
            channels = _ROOT_CHANNELS
        else:
            p = int(parents[i])
            off = neutral[i] - neutral[p]
            channels = _JOINT_CHANNELS
        offset = " ".join(f"{float(x):.6f}" for x in off)
        bvh_lines.append(f"{prefix}  OFFSET {offset}")
        bvh_lines.append(f"{prefix}  CHANNELS {len(channels)} {' '.join(channels)}")
        for c in children[i]:
            _write_joint(int(c), indent + 1)
        bvh_lines.append(f"{prefix}}}")

    # Wrapper Root at origin; its single child is Hips, which carries root motion.
    _write_joint(root_idx, indent=1)
    bvh_lines.append("}")
    bvh_lines.append("MOTION")
    bvh_lines.append(f"Frames: {T}")
    bvh_lines.append(f"Frame Time: {1.0 / float(fps)}")

    for t in range(T):
        values = [0.0] * 6
        for jid in ordered_joint_ids:
            if jid == root_idx:
                values.extend(root_xyz[t].tolist())
            values.extend(local_eulers[t, jid].tolist())
        bvh_lines.append(" ".join(f"{float(value):.6f}" for value in values))

    return "\n".join(bvh_lines) + "\n"


def motion_to_bvh_bytes(
    local_rot_mats: torch.Tensor,
    root_positions: torch.Tensor,
    *,
    skeleton,
    fps: float,
    standard_tpose: bool = False,
) -> bytes:
    """Convert local rotations and root positions to BVH bytes (UTF-8).

    Convenience wrapper around :func:`motion_to_bvh`.
    """
    return motion_to_bvh(
        local_rot_mats,
        root_positions,
        skeleton=skeleton,
        fps=fps,
        standard_tpose=standard_tpose,
    ).encode("utf-8")


def save_motion_bvh(
    path: Union[str, Path],
    local_rot_mats: torch.Tensor,
    root_positions: torch.Tensor,
    *,
    skeleton,
    fps: float,
    standard_tpose: bool = False,
) -> None:
    """Write local rotations and root positions to a BVH file at the given path."""
    Path(path).write_text(
        motion_to_bvh(local_rot_mats, root_positions, skeleton=skeleton, fps=fps, standard_tpose=standard_tpose),
        encoding="utf-8",
    )


def read_bvh_frame_time_seconds(path: Union[str, Path]) -> float:
    """Read ``Frame Time`` from a BVH file (seconds per frame)."""
    with open(path, encoding="utf-8") as f:
        for line in f:
            if "Frame Time:" in line:
                parts = line.split()
                return float(parts[-1])
    raise ValueError(f"Could not find 'Frame Time:' in {path}")


def bvh_to_kimodo_motion(
    path: Union[str, Path],
    skeleton=None,
    *,
    standard_tpose: bool = False,
) -> Tuple:
    """Load a Kimodo-style SOMA BVH into a Kimodo motion dict.

    Expects the same hierarchy as :func:`save_motion_bvh` (``Root`` wrapper + SOMA77 joints).
    The frame rate is always read from the BVH ``Frame Time`` header.  Callers
    that need a different playback rate should resample the returned motion dict
    (see :func:`~kimodo.exports.motion_io.resample_motion_dict_to_kimodo_fps`).

    Returns:
        ``(motion_dict, source_fps)`` where ``source_fps`` is the native BVH
        frame rate read from the file header.
    """
    from kimodo.exports.motion_io import complete_motion_dict
    from kimodo.skeleton.bvh import parse_bvh_motion
    from kimodo.skeleton.registry import build_skeleton

    if skeleton is None:
        skeleton = build_skeleton(77)
    device = skeleton.neutral_joints.device

    local_rot_mats, root_trans, bvh_fps = parse_bvh_motion(str(path))
    local_rot_mats = local_rot_mats.to(device=device)
    root_trans = root_trans.to(device=device)

    if int(local_rot_mats.shape[1]) != int(skeleton.nbjoints):
        raise ValueError(
            f"BVH has {local_rot_mats.shape[1]} joints but skeleton has {skeleton.nbjoints}; "
            "use a Kimodo-exported SOMA BVH or matching skeleton."
        )
    if not standard_tpose:
        local_rot_mats, _ = skeleton.to_standard_tpose(local_rot_mats)

    return complete_motion_dict(local_rot_mats, root_trans, skeleton, float(bvh_fps)), bvh_fps
