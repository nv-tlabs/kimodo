# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Dataset utilities for G1 CSV training with Kimodo motion representation.

This module keeps the original phase-1 preprocessing pipeline and adds phase-2
constraint sampling support:
1. sample keyframes from GT motion,
2. build phase-2 constraint sets (full body / end-effectors / root2d / foot contacts),
3. convert constraints to ``(observed_motion, motion_mask)``,
4. expose a scheduler-friendly dataset interface.
"""

from __future__ import annotations

import math
import random
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import torch
from torch.utils.data import Dataset

from kimodo.constraints import (
    EndEffectorConstraintSet,
    FootContactConstraintSet,
    FullBodyConstraintSet,
    Root2DConstraintSet,
)
from kimodo.motion_rep.feature_utils import length_to_mask

from .g1_csv import PathLike, load_g1_csv_motion
from .timeline_annotations import TimelineAnnotationIndex

log = logging.getLogger(__name__)


def _pad_motion_to(motion: torch.Tensor, target_len: int) -> torch.Tensor:
    """Pad a ``[T, D]`` tensor with zeros to ``[target_len, D]``."""
    t, d = motion.shape
    if t == target_len:
        return motion
    if t > target_len:
        raise ValueError(f"Cannot pad motion with T={t} to smaller target_len={target_len}.")
    pad = torch.zeros((target_len - t, d), dtype=motion.dtype, device=motion.device)
    return torch.cat([motion, pad], dim=0)


class LinearKeyframeScheduler:
    """Linear scheduler for phase-2 keyframe count.

    Example:
        scheduler = LinearKeyframeScheduler(0, 500_000, 1, 12)
        num_keyframes = scheduler(global_step)
    """

    def __init__(
        self,
        start_step: int,
        end_step: int,
        start_keyframes: int,
        end_keyframes: int,
    ) -> None:
        self.start_step = int(start_step)
        self.end_step = int(end_step)
        self.start_keyframes = int(start_keyframes)
        self.end_keyframes = int(end_keyframes)

    def __call__(self, step: int) -> int:
        step = int(step)
        if step <= self.start_step:
            return self.start_keyframes
        if step >= self.end_step:
            return self.end_keyframes
        if self.end_step <= self.start_step:
            return self.end_keyframes
        alpha = (step - self.start_step) / float(self.end_step - self.start_step)
        value = self.start_keyframes + alpha * (self.end_keyframes - self.start_keyframes)
        return int(round(value))


class G1CSVTextDataset(Dataset):
    """G1 motion-text dataset with preprocessing and optional phase-2 constraints.

    Base preprocessing pipeline:
    1. CSV -> ``local_joint_rots`` + ``root_positions``
    2. optional frame-rate alignment (downsample to ``motion_rep.fps``)
    3. ``motion_rep.__call__`` -> ``[T, D]``
    4. optional augmentations
    5. optional normalization
    6. crop/pad to ``max_frames``

    Optional phase-2 pipeline:
    7. sample keyframes from GT sequence
    8. build one or two constraint patterns (paper-style) or legacy fullbody/endeffector
    9. convert constraints to ``(observed_motion, motion_mask)``
    """

    DEFAULT_END_EFFECTOR_JOINT_SETS: tuple[tuple[str, ...], ...] = (
        ("LeftFoot", "RightFoot"),
        ("LeftHand", "RightHand"),
        ("LeftFoot", "RightFoot", "LeftHand", "RightHand"),
        ("LeftFoot", "RightFoot", "Hips"),
    )
    DEFAULT_END_EFFECTOR_JOINT_POOL: tuple[str, ...] = (
        "LeftHand",
        "RightHand",
        "LeftFoot",
        "RightFoot",
    )
    PAPER_PATTERN_NAMES: tuple[str, ...] = (
        "fullbody_sparse",
        "endeffector_sparse",
        "fivepoint_dense",
        "root2d_sparse",
        "root2d_dense",
        "foot_contacts_sparse",
    )

    def __init__(
        self,
        csv_root: PathLike,
        motion_rep: Any,
        *,
        max_frames: int,
        to_normalize: bool = True,
        random_crop: bool = True,
        rotate_to_zero_prob: float = 0.0,
        randomize_first_heading_prob: float = 0.0,
        translate_to_zero_prob: float = 0.0,
        timelines_path: Optional[PathLike] = None,
        text_mode: str = "mixed",
        csv_paths: Optional[Sequence[PathLike]] = None,
        rng_seed: Optional[int] = None,
        skip_bad_samples: bool = False,
        max_bad_sample_retries: int = 8,
        # forward to load_g1_csv_motion
        source_coord_system: str = "mujoco",
        root_euler_order: str = "xyz",
        root_angle_unit: str = "degrees",
        joint_angle_unit: str = "degrees",
        root_position_unit: str = "auto",
        root_position_scale: float = 1.0,
        input_fps: Optional[int] = None,
        # phase-2 sparse constraints
        phase2_enabled: bool = False,
        phase2_num_keyframes: int = 0,
        phase2_constraint_mode: str = "mixed",
        phase2_fullbody_prob: float = 0.5,
        phase2_end_effector_joint_sets: Optional[Sequence[Sequence[str]]] = None,
        phase2_include_first: bool = True,
        phase2_include_last: bool = True,
        phase2_condition_normalized: Optional[bool] = None,
        phase2_return_constraints: bool = False,
        # paper-style phase-2 options
        phase2_constraint_policy: str = "legacy",
        phase2_no_constraint_prob: float = 0.0,
        phase2_mix_two_patterns_prob: float = 0.0,
        phase2_pattern_weights: Optional[Dict[str, float]] = None,
        phase2_bias_towards_fewer_keyframes: bool = False,
        phase2_dense_path_stride: int = 1,
        phase2_end_effector_joint_pool: Optional[Sequence[str]] = None,
    ):
        super().__init__()

        self.csv_root = Path(csv_root)
        self.motion_rep = motion_rep

        self.max_frames = int(max_frames)
        self.to_normalize = bool(to_normalize)
        self.random_crop = bool(random_crop)
        self.rotate_to_zero_prob = float(rotate_to_zero_prob)
        self.randomize_first_heading_prob = float(randomize_first_heading_prob)
        self.translate_to_zero_prob = float(translate_to_zero_prob)
        self.text_mode = str(text_mode).strip().lower()
        self.rng = random.Random(rng_seed)
        self.skip_bad_samples = bool(skip_bad_samples)
        self.max_bad_sample_retries = max(0, int(max_bad_sample_retries))
        self.input_fps = int(input_fps) if input_fps is not None else None
        self.target_fps = int(getattr(self.motion_rep, "fps", 0)) or None

        self.phase2_enabled = bool(phase2_enabled)
        self.phase2_num_keyframes = max(0, int(phase2_num_keyframes))
        mode = str(phase2_constraint_mode).strip().lower().replace("-", "").replace("_", "")
        self.phase2_constraint_mode = mode
        self.phase2_fullbody_prob = float(phase2_fullbody_prob)
        self.phase2_include_first = bool(phase2_include_first)
        self.phase2_include_last = bool(phase2_include_last)
        self.phase2_return_constraints = bool(phase2_return_constraints)
        self.phase2_constraint_policy = str(phase2_constraint_policy).strip().lower()
        self.phase2_no_constraint_prob = float(phase2_no_constraint_prob)
        self.phase2_mix_two_patterns_prob = float(phase2_mix_two_patterns_prob)
        self.phase2_bias_towards_fewer_keyframes = bool(phase2_bias_towards_fewer_keyframes)
        self.phase2_dense_path_stride = max(1, int(phase2_dense_path_stride))
        if phase2_end_effector_joint_pool is None:
            self.phase2_end_effector_joint_pool = list(self.DEFAULT_END_EFFECTOR_JOINT_POOL)
        else:
            self.phase2_end_effector_joint_pool = [str(x) for x in phase2_end_effector_joint_pool]

        if phase2_condition_normalized is None:
            self.phase2_condition_normalized = self.to_normalize
        else:
            self.phase2_condition_normalized = bool(phase2_condition_normalized)

        if phase2_end_effector_joint_sets is None:
            self.phase2_end_effector_joint_sets = [list(x) for x in self.DEFAULT_END_EFFECTOR_JOINT_SETS]
        else:
            self.phase2_end_effector_joint_sets = [list(x) for x in phase2_end_effector_joint_sets]
        if phase2_pattern_weights is None:
            self.phase2_pattern_weights = {name: 1.0 for name in self.PAPER_PATTERN_NAMES}
        else:
            self.phase2_pattern_weights = {str(k): float(v) for k, v in phase2_pattern_weights.items()}

        if self.max_frames <= 0:
            raise ValueError(f"max_frames must be > 0, got {self.max_frames}.")

        for name, prob in [
            ("rotate_to_zero_prob", self.rotate_to_zero_prob),
            ("randomize_first_heading_prob", self.randomize_first_heading_prob),
            ("translate_to_zero_prob", self.translate_to_zero_prob),
        ]:
            if not (0.0 <= prob <= 1.0):
                raise ValueError(f"{name} must be in [0, 1], got {prob}.")

        if self.phase2_constraint_mode not in {"mixed", "fullbody", "endeffector"}:
            raise ValueError(
                "phase2_constraint_mode must be one of: mixed, fullbody, endeffector. "
                f"Got {self.phase2_constraint_mode!r}."
            )
        if self.phase2_constraint_policy not in {"legacy", "paper"}:
            raise ValueError(
                f"phase2_constraint_policy must be one of {{'legacy','paper'}}, got {self.phase2_constraint_policy!r}."
            )
        if not (0.0 <= self.phase2_fullbody_prob <= 1.0):
            raise ValueError(f"phase2_fullbody_prob must be in [0, 1], got {self.phase2_fullbody_prob}.")
        if not (0.0 <= self.phase2_no_constraint_prob <= 1.0):
            raise ValueError(
                f"phase2_no_constraint_prob must be in [0, 1], got {self.phase2_no_constraint_prob}."
            )
        if not (0.0 <= self.phase2_mix_two_patterns_prob <= 1.0):
            raise ValueError(
                f"phase2_mix_two_patterns_prob must be in [0, 1], got {self.phase2_mix_two_patterns_prob}."
            )
        if not self.phase2_end_effector_joint_pool:
            raise ValueError("phase2_end_effector_joint_pool cannot be empty.")
        if self.phase2_enabled and not self.phase2_end_effector_joint_sets:
            raise ValueError("phase2_end_effector_joint_sets cannot be empty when phase2_enabled=True.")
        if self.phase2_enabled and not hasattr(self.motion_rep, "create_conditions_from_constraints"):
            raise TypeError("motion_rep must implement create_conditions_from_constraints for phase-2 constraints.")
        if self.phase2_enabled and not hasattr(self.motion_rep, "inverse"):
            raise TypeError("motion_rep must implement inverse() for phase-2 constraints.")
        valid_pattern_names = set(self.PAPER_PATTERN_NAMES)
        if not set(self.phase2_pattern_weights).issubset(valid_pattern_names):
            raise ValueError(
                "phase2_pattern_weights contains unknown keys. "
                f"Allowed keys: {sorted(valid_pattern_names)}. Got: {sorted(self.phase2_pattern_weights)}."
            )
        for name, weight in self.phase2_pattern_weights.items():
            if weight < 0.0 or not math.isfinite(weight):
                raise ValueError(f"phase2_pattern_weights[{name!r}] must be a finite >= 0 number, got {weight}.")
        if self.phase2_constraint_policy == "paper":
            total_weight = sum(self.phase2_pattern_weights.values())
            if total_weight <= 0.0:
                raise ValueError("For phase2_constraint_policy='paper', phase2_pattern_weights must sum to > 0.")

        self.frame_stride = 1
        if self.input_fps is not None:
            if self.target_fps is None:
                raise ValueError("motion_rep must expose fps when input_fps is provided.")
            if self.input_fps < self.target_fps:
                raise ValueError(
                    f"input_fps ({self.input_fps}) must be >= motion_rep.fps ({self.target_fps})."
                )
            ratio = self.input_fps / float(self.target_fps)
            ratio_rounded = int(round(ratio))
            if abs(ratio - ratio_rounded) > 1e-6:
                raise ValueError(
                    "input_fps must be an integer multiple of motion_rep.fps. "
                    f"Got input_fps={self.input_fps}, motion_rep.fps={self.target_fps}."
                )
            self.frame_stride = max(1, ratio_rounded)

        if csv_paths is None:
            self.csv_paths = sorted(self.csv_root.rglob("*.csv"))
        else:
            self.csv_paths = [Path(p) for p in csv_paths]
        if not self.csv_paths:
            raise FileNotFoundError(f"No CSV files found under {self.csv_root}")

        self.timeline_index: Optional[TimelineAnnotationIndex] = None
        if timelines_path is not None:
            self.timeline_index = TimelineAnnotationIndex.from_jsonl(timelines_path)

        self.g1_loader_kwargs: Dict[str, Any] = {
            "source_coord_system": source_coord_system,
            "root_euler_order": root_euler_order,
            "root_angle_unit": root_angle_unit,
            "joint_angle_unit": joint_angle_unit,
            "root_position_unit": root_position_unit,
            "root_position_scale": root_position_scale,
        }

    def __len__(self) -> int:
        return len(self.csv_paths)

    def set_phase2_num_keyframes(self, num_keyframes: int) -> None:
        """Update current phase-2 keyframe count (called by train loop scheduler)."""
        self.phase2_num_keyframes = max(0, int(num_keyframes))

    def get_phase2_num_keyframes(self) -> int:
        """Get current phase-2 keyframe count."""
        return int(self.phase2_num_keyframes)

    # Minimum number of frames (at target fps) for an event crop to be usable.
    MIN_EVENT_FRAMES: int = 15

    def _sample_text_and_time_range(
        self, csv_path: Path
    ) -> Tuple[str, Optional[float], Optional[float]]:
        """Sample text and optional time range.

        Returns:
            (text, start_time, end_time) where times are in seconds.
            For overview mode or when no timeline exists, times are None.
        """
        if self.text_mode in {"none", "empty", "no_text", "notext"}:
            return "", None, None

        if self.timeline_index is None:
            return csv_path.stem, None, None

        if self.text_mode == "overview":
            text = self.timeline_index.sample_text(csv_path, mode="overview", rng=self.rng)
            return text, None, None

        if self.text_mode == "event":
            return self._sample_event_with_range(csv_path)

        # mixed: 50% overview, 50% event-aligned
        if self.rng.random() < 0.5:
            text = self.timeline_index.sample_text(csv_path, mode="overview", rng=self.rng)
            return text, None, None
        return self._sample_event_with_range(csv_path)

    def _sample_event_with_range(
        self, csv_path: Path
    ) -> Tuple[str, Optional[float], Optional[float]]:
        """Pick a random event and return (text, start_time, end_time)."""
        try:
            rec = self.timeline_index.get_record(csv_path)
        except KeyError:
            return csv_path.stem, None, None

        events = rec["events"]
        if not events:
            overview = rec["overview_description"].strip()
            return (overview if overview else csv_path.stem), None, None

        event = self.rng.choice(events)
        start_t = float(event["start_time"])
        end_t = float(event["end_time"])

        # Check if the event is long enough after downsampling
        if self.input_fps is not None:
            event_frames = int((end_t - start_t) * self.target_fps)
        else:
            event_frames = int((end_t - start_t) * 30)  # fallback

        if event_frames < self.MIN_EVENT_FRAMES:
            # Event too short, fall back to overview
            overview = rec["overview_description"].strip()
            return (overview if overview else str(event["description"])), None, None

        return str(event["description"]), start_t, end_t

    def _encode_motion(self, local_joint_rots: torch.Tensor, root_positions: torch.Tensor) -> torch.Tensor:
        """Encode local rotations + root positions into Kimodo feature tensor ``[T, D]``."""
        t = int(local_joint_rots.shape[0])
        lengths = torch.tensor([t], dtype=torch.long, device=local_joint_rots.device)
        features = self.motion_rep(
            local_joint_rots.unsqueeze(0),
            root_positions.unsqueeze(0),
            to_normalize=False,
            lengths=lengths,
        )
        return features.squeeze(0)

    def _maybe_downsample_motion(
        self,
        local_joint_rots: torch.Tensor,
        root_positions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Optionally downsample frames to match ``motion_rep.fps``."""
        if self.frame_stride <= 1:
            return local_joint_rots, root_positions
        return local_joint_rots[:: self.frame_stride], root_positions[:: self.frame_stride]

    def _augment_features(self, features: torch.Tensor) -> torch.Tensor:
        """Apply optional feature-space augmentation (before normalization)."""
        x = features.unsqueeze(0)

        # At most one heading augmentation per sample.
        if self.randomize_first_heading_prob > 0.0 and self.rng.random() < self.randomize_first_heading_prob:
            x = self.motion_rep.randomize_first_heading(x)
        elif self.rotate_to_zero_prob > 0.0 and self.rng.random() < self.rotate_to_zero_prob:
            x = self.motion_rep.rotate_to_zero(x)

        if self.translate_to_zero_prob > 0.0 and self.rng.random() < self.translate_to_zero_prob:
            x = self.motion_rep.translate_2d_to_zero(x)
        return x.squeeze(0)

    def _crop_or_pad(self, features: torch.Tensor, start: Optional[int] = None) -> tuple[torch.Tensor, int, int]:
        """Crop/pad ``[T, D]`` features to ``[max_frames, D]``."""
        t = int(features.shape[0])
        if t > self.max_frames:
            if start is None:
                if self.random_crop:
                    start = self.rng.randint(0, t - self.max_frames)
                else:
                    start = 0
            end = start + self.max_frames
            return features[start:end], self.max_frames, start

        padded = _pad_motion_to(features, self.max_frames)
        return padded, t, 0

    def _sample_num_sparse_keyframes(self, length: int) -> int:
        if length <= 0 or self.phase2_num_keyframes <= 0:
            return 0

        max_k = min(int(self.phase2_num_keyframes), int(length))
        if max_k <= 0:
            return 0
        if not self.phase2_bias_towards_fewer_keyframes or max_k == 1:
            return max_k

        candidates = list(range(1, max_k + 1))
        weights = [1.0 / float(k) for k in candidates]
        return int(self.rng.choices(candidates, weights=weights, k=1)[0])

    def _sample_keyframe_indices(self, length: int, num_keyframes: Optional[int] = None) -> torch.Tensor:
        if length <= 0:
            return torch.zeros((0,), dtype=torch.long)

        if num_keyframes is None:
            num = self._sample_num_sparse_keyframes(length)
        else:
            num = min(int(num_keyframes), int(length))
        if num <= 0:
            return torch.zeros((0,), dtype=torch.long)

        chosen: set[int] = set()

        if self.phase2_include_first:
            chosen.add(0)
        if self.phase2_include_last and length > 1:
            chosen.add(length - 1)

        if len(chosen) > num:
            chosen = set(sorted(chosen)[:num])

        remaining = [i for i in range(length) if i not in chosen]
        need = num - len(chosen)
        if need > 0:
            chosen.update(self.rng.sample(remaining, k=min(need, len(remaining))))

        return torch.tensor(sorted(chosen), dtype=torch.long)

    def _sample_constraint_mode(self) -> str:
        if self.phase2_constraint_mode in {"fullbody", "endeffector"}:
            return self.phase2_constraint_mode
        if self.rng.random() < self.phase2_fullbody_prob:
            return "fullbody"
        return "endeffector"

    def _sample_weighted_without_replacement(self, names: list[str], weights: list[float], k: int) -> list[str]:
        if k <= 0:
            return []
        selected: list[str] = []
        pool_names = names[:]
        pool_weights = weights[:]

        for _ in range(min(k, len(pool_names))):
            total = float(sum(pool_weights))
            if total <= 0.0:
                break
            needle = self.rng.random() * total
            cum = 0.0
            pick_idx = 0
            for i, w in enumerate(pool_weights):
                cum += float(w)
                if needle <= cum:
                    pick_idx = i
                    break
            selected.append(pool_names.pop(pick_idx))
            pool_weights.pop(pick_idx)
        return selected

    def _dense_frame_indices(self, valid_length: int) -> torch.Tensor:
        indices = torch.arange(0, valid_length, self.phase2_dense_path_stride, dtype=torch.long)
        dense_set = set(indices.tolist())
        if self.phase2_include_first and valid_length > 0:
            dense_set.add(0)
        if self.phase2_include_last and valid_length > 1:
            dense_set.add(valid_length - 1)
        if not dense_set:
            return torch.zeros((0,), dtype=torch.long)
        return torch.tensor(sorted(dense_set), dtype=torch.long)

    def _sample_phase2_pattern_names(self) -> list[str]:
        if self.phase2_constraint_policy == "legacy":
            mode = self._sample_constraint_mode()
            if mode == "fullbody":
                return ["fullbody_sparse"]
            return ["endeffector_sparse"]

        # paper policy
        if self.rng.random() < self.phase2_no_constraint_prob:
            return []

        n_patterns = 2 if (self.rng.random() < self.phase2_mix_two_patterns_prob) else 1
        names = []
        weights = []
        for name in self.PAPER_PATTERN_NAMES:
            w = float(self.phase2_pattern_weights.get(name, 0.0))
            if w > 0.0:
                names.append(name)
                weights.append(w)
        return self._sample_weighted_without_replacement(names, weights, n_patterns)

    def _sample_end_effector_joint_subset(self) -> list[str]:
        if self.phase2_constraint_policy == "legacy":
            return list(self.rng.choice(self.phase2_end_effector_joint_sets))
        n = len(self.phase2_end_effector_joint_pool)
        subset_size = self.rng.randint(1, n)
        return list(self.rng.sample(self.phase2_end_effector_joint_pool, k=subset_size))

    def _empty_phase2_conditions(self, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        motion_dim = int(self.motion_rep.motion_rep_dim)
        observed_motion = torch.zeros((self.max_frames, motion_dim), device=device, dtype=torch.float32)
        motion_mask = torch.zeros((self.max_frames, motion_dim), device=device, dtype=torch.bool)
        return observed_motion, motion_mask

    def _build_constraints_for_pattern(
        self,
        pattern_name: str,
        *,
        valid_length: int,
        posed_joints: torch.Tensor,
        global_rot_mats: torch.Tensor,
        smooth_root_pos: torch.Tensor,
        global_root_heading: torch.Tensor,
        foot_contacts: torch.Tensor,
    ) -> tuple[list[Any], torch.Tensor]:
        if pattern_name == "fullbody_sparse":
            indices = self._sample_keyframe_indices(valid_length)
            if indices.numel() == 0:
                return [], indices
            constraints = [
                FullBodyConstraintSet(
                    self.motion_rep.skeleton,
                    frame_indices=indices,
                    global_joints_positions=posed_joints[indices],
                    global_joints_rots=global_rot_mats[indices],
                    smooth_root_2d=smooth_root_pos[indices][:, [0, 2]],
                )
            ]
            return constraints, indices

        if pattern_name == "endeffector_sparse":
            indices = self._sample_keyframe_indices(valid_length)
            if indices.numel() == 0:
                return [], indices
            constraints = [
                EndEffectorConstraintSet(
                    self.motion_rep.skeleton,
                    frame_indices=indices,
                    global_joints_positions=posed_joints[indices],
                    global_joints_rots=global_rot_mats[indices],
                    smooth_root_2d=smooth_root_pos[indices][:, [0, 2]],
                    joint_names=self._sample_end_effector_joint_subset(),
                )
            ]
            return constraints, indices

        if pattern_name == "fivepoint_dense":
            indices = self._dense_frame_indices(valid_length)
            if indices.numel() == 0:
                return [], indices
            constraints = [
                EndEffectorConstraintSet(
                    self.motion_rep.skeleton,
                    frame_indices=indices,
                    global_joints_positions=posed_joints[indices],
                    global_joints_rots=global_rot_mats[indices],
                    smooth_root_2d=smooth_root_pos[indices][:, [0, 2]],
                    joint_names=list(self.phase2_end_effector_joint_pool),
                )
            ]
            return constraints, indices

        if pattern_name == "root2d_sparse":
            indices = self._sample_keyframe_indices(valid_length)
            if indices.numel() == 0:
                return [], indices
            constraints = [
                Root2DConstraintSet(
                    self.motion_rep.skeleton,
                    frame_indices=indices,
                    smooth_root_2d=smooth_root_pos[indices][:, [0, 2]],
                    global_root_heading=global_root_heading[indices],
                )
            ]
            return constraints, indices

        if pattern_name == "root2d_dense":
            indices = self._dense_frame_indices(valid_length)
            if indices.numel() == 0:
                return [], indices
            constraints = [
                Root2DConstraintSet(
                    self.motion_rep.skeleton,
                    frame_indices=indices,
                    smooth_root_2d=smooth_root_pos[indices][:, [0, 2]],
                    global_root_heading=global_root_heading[indices],
                )
            ]
            return constraints, indices

        if pattern_name == "foot_contacts_sparse":
            indices = self._sample_keyframe_indices(valid_length)
            if indices.numel() == 0:
                return [], indices
            constraints = [
                FootContactConstraintSet(
                    self.motion_rep.skeleton,
                    frame_indices=indices,
                    foot_contacts=foot_contacts[indices],
                )
            ]
            return constraints, indices

        raise ValueError(f"Unknown phase2 pattern: {pattern_name!r}")

    def _build_phase2_conditions(
        self,
        motion_unnorm: torch.Tensor,
        valid_length: int,
    ) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor], Optional[str], Any]:
        """Build phase-2 sparse conditions from GT motion."""
        if not self.phase2_enabled or self.phase2_num_keyframes <= 0 or valid_length <= 0:
            return None, None, None, None, None

        pattern_names = self._sample_phase2_pattern_names()
        if len(pattern_names) == 0:
            observed_motion, motion_mask = self._empty_phase2_conditions(motion_unnorm.device)
            constraints_out = [] if self.phase2_return_constraints else None
            return observed_motion, motion_mask, torch.zeros((0,), dtype=torch.long), "none", constraints_out

        decoded = self.motion_rep.inverse(
            motion_unnorm.unsqueeze(0),
            is_normalized=False,
            return_numpy=False,
        )

        posed_joints = decoded["posed_joints"].squeeze(0)
        global_rot_mats = decoded["global_rot_mats"].squeeze(0)
        smooth_root_pos = decoded["smooth_root_pos"].squeeze(0)
        global_root_heading = decoded["global_root_heading"].squeeze(0)
        foot_contacts = decoded["foot_contacts"].squeeze(0).to(dtype=torch.float32)

        constraints: list[Any] = []
        all_indices: list[torch.Tensor] = []
        for pattern_name in pattern_names:
            pattern_constraints, frame_indices = self._build_constraints_for_pattern(
                pattern_name,
                valid_length=valid_length,
                posed_joints=posed_joints,
                global_rot_mats=global_rot_mats,
                smooth_root_pos=smooth_root_pos,
                global_root_heading=global_root_heading,
                foot_contacts=foot_contacts,
            )
            if pattern_constraints:
                constraints.extend(pattern_constraints)
                if frame_indices.numel() > 0:
                    all_indices.append(frame_indices)

        if not constraints:
            observed_motion, motion_mask = self._empty_phase2_conditions(motion_unnorm.device)
            constraints_out = [] if self.phase2_return_constraints else None
            return observed_motion, motion_mask, torch.zeros((0,), dtype=torch.long), "none", constraints_out

        observed_motion, motion_mask = self.motion_rep.create_conditions_from_constraints(
            constraints,
            length=self.max_frames,
            to_normalize=self.phase2_condition_normalized,
            device=motion_unnorm.device,
        )

        if all_indices:
            keyframe_indices = torch.unique(torch.cat(all_indices), sorted=True)
        else:
            keyframe_indices = torch.zeros((0,), dtype=torch.long)
        mode = "+".join(pattern_names)
        constraints_out: Any = constraints if self.phase2_return_constraints else None
        return observed_motion, motion_mask, keyframe_indices, mode, constraints_out

    def _crop_motion_by_time(
        self,
        local_joint_rots: torch.Tensor,
        root_positions: torch.Tensor,
        start_time: float,
        end_time: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Crop raw motion tensors to a time range (at input_fps)."""
        fps = self.input_fps if self.input_fps is not None else 30
        start_frame = max(0, int(start_time * fps))
        end_frame = min(int(local_joint_rots.shape[0]), int(end_time * fps))
        if end_frame <= start_frame:
            return local_joint_rots, root_positions
        return local_joint_rots[start_frame:end_frame], root_positions[start_frame:end_frame]

    def _build_item(self, idx: int) -> Dict[str, Any]:
        csv_path = self.csv_paths[idx]
        if os.environ.get("KIMODO_DEBUG_LOG_CSV_LOAD", "0") == "1":
            log.warning("Loading CSV idx=%d path=%s", idx, csv_path)
        motion_data = load_g1_csv_motion(csv_path, **self.g1_loader_kwargs)
        local_joint_rots = motion_data["local_joint_rots"]
        root_positions = motion_data["root_positions"]

        # Sample text and optional event time range
        text, event_start, event_end = self._sample_text_and_time_range(csv_path)

        # If event mode returned a time range, crop raw motion BEFORE downsampling
        if event_start is not None and event_end is not None:
            local_joint_rots, root_positions = self._crop_motion_by_time(
                local_joint_rots, root_positions, event_start, event_end
            )

        local_joint_rots, root_positions = self._maybe_downsample_motion(local_joint_rots, root_positions)

        features_unnorm = self._encode_motion(local_joint_rots, root_positions)
        features_unnorm = self._augment_features(features_unnorm)

        features = features_unnorm
        if self.to_normalize:
            features = self.motion_rep.normalize(features.unsqueeze(0)).squeeze(0)

        motion, length, frame_start = self._crop_or_pad(features)
        motion_unnorm, _, _ = self._crop_or_pad(features_unnorm, start=frame_start)
        pad_mask = length_to_mask([length], max_len=self.max_frames, device=motion.device)[0]

        observed_motion, motion_mask, keyframe_indices, constraint_mode, constraints = self._build_phase2_conditions(
            motion_unnorm=motion_unnorm,
            valid_length=length,
        )

        return {
            "motion": motion,  # [max_frames, D]
            "length": torch.tensor(length, dtype=torch.long),
            "pad_mask": pad_mask,  # [max_frames] bool, True=valid
            "text": text,
            "csv_path": str(csv_path),
            "clip_name": csv_path.stem,
            "frame_start": torch.tensor(frame_start, dtype=torch.long),
            "frame_end": torch.tensor(frame_start + length, dtype=torch.long),
            # phase-2 (optional)
            "observed_motion": observed_motion,  # [max_frames, D] or None
            "motion_mask": motion_mask,  # [max_frames, D] bool or None
            "keyframe_indices": keyframe_indices,  # [K] or None
            "constraint_mode": constraint_mode,  # pattern string (e.g. "fullbody_sparse+root2d_dense") or None
            "constraints": constraints,  # optional raw constraint objects
        }

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        if not self.skip_bad_samples:
            return self._build_item(idx)

        current_idx = int(idx)
        attempts = self.max_bad_sample_retries + 1
        last_exc: Optional[Exception] = None
        for attempt in range(1, attempts + 1):
            try:
                return self._build_item(current_idx)
            except Exception as exc:  # noqa: BLE001 - dataset robustness fallback
                last_exc = exc
                csv_path = self.csv_paths[current_idx]
                if attempt >= attempts or len(self.csv_paths) <= 1:
                    break
                next_idx = self.rng.randrange(len(self.csv_paths))
                if next_idx == current_idx:
                    next_idx = (next_idx + 1) % len(self.csv_paths)
                log.warning(
                    "Skipping bad sample idx=%d path=%s due to %s: %s. "
                    "Resampling idx=%d (attempt %d/%d).",
                    current_idx,
                    csv_path,
                    type(exc).__name__,
                    exc,
                    next_idx,
                    attempt,
                    attempts,
                )
                current_idx = next_idx

        assert last_exc is not None
        raise RuntimeError(
            f"Failed to load sample after {attempts} attempts (start_idx={idx})."
        ) from last_exc


def g1_text_collate_fn(batch: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Collate function for ``G1CSVTextDataset`` output dicts."""
    if len(batch) == 0:
        raise ValueError("Empty batch.")

    motions = torch.stack([item["motion"] for item in batch], dim=0)
    lengths = torch.stack([item["length"] for item in batch], dim=0)
    pad_masks = torch.stack([item["pad_mask"] for item in batch], dim=0)
    frame_starts = torch.stack([item["frame_start"] for item in batch], dim=0)
    frame_ends = torch.stack([item["frame_end"] for item in batch], dim=0)

    phase2_flags = [item["observed_motion"] is not None for item in batch]
    if any(phase2_flags):
        first = next(item for item in batch if item["observed_motion"] is not None)
        t_dim, d_dim = first["observed_motion"].shape
        obs_dtype = first["observed_motion"].dtype
        obs_device = first["observed_motion"].device
        mask_device = first["motion_mask"].device

        observed_motion_list = []
        motion_mask_list = []
        for item in batch:
            if item["observed_motion"] is None:
                observed_motion_list.append(torch.zeros((t_dim, d_dim), dtype=obs_dtype, device=obs_device))
                motion_mask_list.append(torch.zeros((t_dim, d_dim), dtype=torch.bool, device=mask_device))
            else:
                observed_motion_list.append(item["observed_motion"])
                motion_mask_list.append(item["motion_mask"])
        observed_motion = torch.stack(observed_motion_list, dim=0)
        motion_mask = torch.stack(motion_mask_list, dim=0)
    else:
        observed_motion = None
        motion_mask = None

    texts = [item["text"] for item in batch]
    csv_paths = [item["csv_path"] for item in batch]
    clip_names = [item["clip_name"] for item in batch]
    keyframe_indices = [item["keyframe_indices"] for item in batch]
    constraint_mode = [item["constraint_mode"] for item in batch]
    constraints = [item["constraints"] for item in batch]

    return {
        "motion": motions,  # [B, T, D]
        "lengths": lengths,  # [B]
        "pad_mask": pad_masks,  # [B, T]
        "observed_motion": observed_motion,  # [B, T, D] or None
        "motion_mask": motion_mask,  # [B, T, D] bool or None
        "text": texts,  # list[str]
        "csv_path": csv_paths,  # list[str]
        "clip_name": clip_names,  # list[str]
        "frame_start": frame_starts,  # [B]
        "frame_end": frame_ends,  # [B]
        "keyframe_indices": keyframe_indices,  # list[Tensor|None]
        "constraint_mode": constraint_mode,  # list[str|None]
        "constraints": constraints,  # list[list|None]
    }


def build_csv_subset(csv_root: PathLike, limit: Optional[int] = None, pattern: str = "*.csv") -> list[Path]:
    """Build a stable subset of CSV files under root (useful for debug runs)."""
    paths = sorted(Path(csv_root).rglob(pattern))
    if limit is None:
        return paths
    return paths[: int(limit)]
