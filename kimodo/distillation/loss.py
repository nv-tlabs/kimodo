# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Distillation losses for Kimodo student training.

This module implements a teacher-guided + GT-regularized objective:
- 7-term Kimodo loss against teacher prediction (teacher-dominant)
- 7-term Kimodo loss against dataset GT (GT-auxiliary)
- optional constraint-aware FK loss against sparse observed_motion/motion_mask conditions
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import torch
from torch import Tensor, nn

from kimodo.skeleton.kinematics import fk
from kimodo.skeleton.transforms import global_rots_to_local_rots
from kimodo.training.loss import DEFAULT_GAMMAS, LOSS_NAMES, compute_kimodo_loss

__all__ = ["DistillationKimodoLoss", "compute_constraint_loss", "constraint_feature_acceleration_loss"]


def _validate_weight(name: str, x: float) -> float:
    x = float(x)
    if x < 0.0:
        raise ValueError(f"{name} must be >= 0, got {x}.")
    return x


def _zero_like(x: Tensor) -> Tensor:
    return x.new_zeros(())


def _masked_weighted_l1(
    pred: Tensor,
    target: Tensor,
    mask: Tensor,
    weights: Tensor,
) -> Tensor:
    if pred.shape != target.shape or pred.shape != mask.shape or pred.shape != weights.shape:
        raise ValueError(
            "pred, target, mask, and weights must have matching shapes. "
            f"Got pred={pred.shape}, target={target.shape}, mask={mask.shape}, weights={weights.shape}."
        )
    mask_bool = mask.to(device=pred.device, dtype=torch.bool)
    mask_f = mask_bool.to(dtype=pred.dtype)
    denom = mask_f.sum().clamp(min=1.0)
    weighted_diff = torch.abs(pred - target) * weights.to(device=pred.device, dtype=pred.dtype)
    return torch.where(mask_bool, weighted_diff, torch.zeros_like(weighted_diff)).sum() / denom


def _masked_weighted_mean(values: Tensor, mask: Tensor, weights: Tensor) -> Tensor:
    if values.shape != mask.shape or values.shape != weights.shape:
        raise ValueError(
            "values, mask, and weights must have matching shapes. "
            f"Got values={values.shape}, mask={mask.shape}, weights={weights.shape}."
        )
    mask_bool = mask.to(device=values.device, dtype=torch.bool)
    mask_f = mask_bool.to(dtype=values.dtype)
    denom = mask_f.sum().clamp(min=1.0)
    weighted_values = values * weights.to(device=values.device, dtype=values.dtype)
    return torch.where(mask_bool, weighted_values, torch.zeros_like(weighted_values)).sum() / denom


def _joint_constraint_weights(
    motion_rep,
    *,
    hand_constraint_weight: float,
    foot_constraint_weight: float,
    root_constraint_weight: float,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    skel = motion_rep.skeleton
    nbjoints = int(motion_rep.nbjoints)
    weights = torch.full((nbjoints,), float(root_constraint_weight), device=device, dtype=dtype)

    bone_index = getattr(skel, "bone_index", {})

    def _set(names: Sequence[str], value: float) -> None:
        for name in names:
            idx = bone_index.get(name)
            if idx is not None:
                weights[int(idx)] = float(value)

    hand_names = list(getattr(skel, "left_hand_joint_names", [])) + list(getattr(skel, "right_hand_joint_names", []))
    foot_names = list(getattr(skel, "left_foot_joint_names", [])) + list(getattr(skel, "right_foot_joint_names", []))
    _set(foot_names, foot_constraint_weight)
    _set(hand_names, hand_constraint_weight)

    root_idx = getattr(skel, "root_idx", None)
    if root_idx is not None:
        weights[int(root_idx)] = float(root_constraint_weight)
    return weights


def _rotation_geodesic(pred_rot: Tensor, target_rot: Tensor) -> Tensor:
    rel = torch.matmul(pred_rot.transpose(-1, -2), target_rot)
    trace = rel.diagonal(offset=0, dim1=-2, dim2=-1).sum(dim=-1)
    cos = ((trace - 1.0) * 0.5).clamp(min=-1.0, max=1.0)
    skew = torch.stack(
        (
            rel[..., 2, 1] - rel[..., 1, 2],
            rel[..., 0, 2] - rel[..., 2, 0],
            rel[..., 1, 0] - rel[..., 0, 1],
        ),
        dim=-1,
    )
    sin = 0.5 * torch.linalg.norm(skew, dim=-1)
    return torch.atan2(sin, cos)


def _safe_cont6d_to_matrix(cont6d: Tensor, eps: float = 1e-6) -> Tensor:
    """Numerically stable 6D rotation decode for constraint loss under AMP."""
    if cont6d.shape[-1] != 6:
        raise ValueError(f"Expected last dim 6 for 6D rotations, got {cont6d.shape}.")

    work = cont6d.float()
    x_raw = work[..., 0:3]
    y_raw = work[..., 3:6]

    x_norm = torch.linalg.norm(x_raw, dim=-1, keepdim=True)
    default_x = torch.zeros_like(x_raw)
    default_x[..., 0] = 1.0
    x = torch.where(x_norm > eps, x_raw / x_norm.clamp_min(eps), default_x)

    y_proj = y_raw - (x * y_raw).sum(dim=-1, keepdim=True) * x
    y_norm = torch.linalg.norm(y_proj, dim=-1, keepdim=True)

    candidate_y = torch.zeros_like(x)
    candidate_y[..., 1] = 1.0
    candidate_z = torch.zeros_like(x)
    candidate_z[..., 2] = 1.0
    fallback_seed = torch.where(x[..., 1:2].abs() < 0.9, candidate_y, candidate_z)
    fallback_y = fallback_seed - (fallback_seed * x).sum(dim=-1, keepdim=True) * x
    fallback_y = fallback_y / torch.linalg.norm(fallback_y, dim=-1, keepdim=True).clamp_min(eps)

    y = torch.where(y_norm > eps, y_proj / y_norm.clamp_min(eps), fallback_y)
    z = torch.cross(x, y, dim=-1)

    return torch.cat([x[..., None], y[..., None], z[..., None]], dim=-1)


def _decode_pred_fk(pred: Tensor, motion_rep) -> Tuple[Tensor, Tensor]:
    """Decode predicted global rotations through FK without allowing degenerate 6D NaNs."""
    slice_dict = motion_rep.slice_dict
    bsz, nframes, _ = pred.shape
    nbjoints = int(motion_rep.nbjoints)
    skel = motion_rep.skeleton

    pred_rot6d = pred[..., slice_dict["global_rot_data"]].reshape(bsz, nframes, nbjoints, 6)
    pred_global_rots = _safe_cont6d_to_matrix(pred_rot6d)
    pred_local_rots = global_rots_to_local_rots(pred_global_rots, skel)

    pred_local_pos = pred[..., slice_dict["local_joints_positions"]].reshape(bsz, nframes, nbjoints, 3).float()
    pred_smooth_root = pred[..., slice_dict["smooth_root_pos"]].float()
    pred_joint_pos_from_features = pred_local_pos.clone()
    pred_joint_pos_from_features[..., 0] += pred_smooth_root[..., None, 0]
    pred_joint_pos_from_features[..., 2] += pred_smooth_root[..., None, 2]
    pred_root_pos = pred_joint_pos_from_features[..., skel.root_idx, :]

    _, pred_pos, _ = fk(pred_local_rots, pred_root_pos, skel)
    return pred_pos, pred_global_rots


def constraint_feature_acceleration_loss(
    *,
    pred_x0: Tensor,
    observed_motion: Optional[Tensor],
    motion_mask: Optional[Tensor],
    motion_rep,
    pad_mask: Optional[Tensor] = None,
    input_is_normalized: bool = False,
    feature_name: str,
    feature_dims: Optional[Sequence[int]] = None,
) -> Tensor:
    """Penalize high-frequency acceleration only for samples that contain a given constraint.

    This is intentionally target-free: ``observed_motion``/``motion_mask`` only decide whether the
    sample is constraint-conditioned. The loss then smooths the predicted clean trajectory for the
    requested feature across valid frames. It is useful for sparse root constraints, where keyframes
    are correct but intermediate root motion can jitter.
    """
    if observed_motion is None or motion_mask is None or pred_x0.shape[1] < 3:
        return _zero_like(pred_x0)
    if pred_x0.shape != motion_mask.shape:
        raise ValueError(
            "pred_x0 and motion_mask must have the same shape. "
            f"Got pred={pred_x0.shape}, mask={motion_mask.shape}."
        )

    feature_slice = getattr(motion_rep, "slice_dict", {}).get(feature_name, None)
    if feature_slice is None:
        return _zero_like(pred_x0)

    pred = pred_x0
    if input_is_normalized:
        if not hasattr(motion_rep, "unnormalize"):
            raise TypeError("motion_rep must provide unnormalize() when input_is_normalized=True.")
        pred = motion_rep.unnormalize(pred)

    feature_mask = motion_mask[..., feature_slice].to(device=pred.device, dtype=torch.bool)
    sample_has_constraint = feature_mask.flatten(start_dim=1).any(dim=1)
    if not bool(sample_has_constraint.any().item()):
        return _zero_like(pred_x0)

    feature = pred[..., feature_slice]
    if feature_dims is not None:
        dims = torch.as_tensor([int(x) for x in feature_dims], device=pred.device, dtype=torch.long)
        if dims.numel() == 0:
            return _zero_like(pred_x0)
        if int(dims.min().item()) < 0 or int(dims.max().item()) >= feature.shape[-1]:
            raise ValueError(
                f"feature_dims must be within [0, {feature.shape[-1] - 1}] for {feature_name}, "
                f"got {feature_dims}."
            )
        feature = feature.index_select(dim=-1, index=dims)

    acc = feature[:, 2:] - 2.0 * feature[:, 1:-1] + feature[:, :-2]
    valid = sample_has_constraint[:, None]
    if pad_mask is not None:
        if pad_mask.ndim != 2 or pad_mask.shape != pred_x0.shape[:2]:
            raise ValueError(
                f"pad_mask must have shape [B, T] matching pred, got {pad_mask.shape} vs {pred_x0.shape[:2]}."
            )
        valid = valid & pad_mask[:, 2:].to(device=pred.device, dtype=torch.bool)
        valid = valid & pad_mask[:, 1:-1].to(device=pred.device, dtype=torch.bool)
        valid = valid & pad_mask[:, :-2].to(device=pred.device, dtype=torch.bool)

    while valid.ndim < acc.ndim:
        valid = valid.unsqueeze(-1)
    valid = valid.expand_as(acc)
    valid_f = valid.to(dtype=pred.dtype)
    return (acc.abs() * valid_f).sum() / valid_f.sum().clamp(min=1.0)


def compute_constraint_loss(
    *,
    pred_x0: Tensor,
    observed_motion: Optional[Tensor],
    motion_mask: Optional[Tensor],
    motion_rep,
    pad_mask: Optional[Tensor] = None,
    input_is_normalized: bool = False,
    hand_constraint_weight: float = 3.0,
    foot_constraint_weight: float = 1.5,
    root_constraint_weight: float = 1.0,
) -> Dict[str, Tensor]:
    """Compute sparse constraint loss from existing ``observed_motion`` and ``motion_mask``.

    The conditioning tensors are already in Kimodo feature space. This loss decodes the student
    prediction through FK and compares only the feature dimensions marked by ``motion_mask``:
    - ``local_joints_positions`` mask -> global joint position targets
    - ``global_rot_data`` mask -> global joint rotation targets
    - ``smooth_root_pos`` / ``global_root_heading`` masks -> root position / heading targets
    """
    if observed_motion is None or motion_mask is None:
        z = _zero_like(pred_x0)
        return {
            "total": z,
            "position": z,
            "rotation": z,
            "root": z,
            "observed_count": z,
        }
    if pred_x0.shape != observed_motion.shape or pred_x0.shape != motion_mask.shape:
        raise ValueError(
            "pred_x0, observed_motion, and motion_mask must have the same shape. "
            f"Got pred={pred_x0.shape}, observed={observed_motion.shape}, mask={motion_mask.shape}."
        )
    if pred_x0.ndim != 3:
        raise ValueError(f"Expected [B, T, D] tensors, got pred_x0.ndim={pred_x0.ndim}.")

    hand_w = _validate_weight("hand_constraint_weight", hand_constraint_weight)
    foot_w = _validate_weight("foot_constraint_weight", foot_constraint_weight)
    root_w = _validate_weight("root_constraint_weight", root_constraint_weight)

    pred = pred_x0
    observed = observed_motion
    if input_is_normalized:
        if not hasattr(motion_rep, "unnormalize"):
            raise TypeError("motion_rep must provide unnormalize() when input_is_normalized=True.")
        pred = motion_rep.unnormalize(pred)
        observed = motion_rep.unnormalize(observed)

    mask = motion_mask.to(device=pred.device, dtype=torch.bool)
    if pad_mask is not None:
        if pad_mask.ndim != 2 or pad_mask.shape != pred.shape[:2]:
            raise ValueError(
                f"pad_mask must have shape [B, T] matching pred, got {pad_mask.shape} vs {pred.shape[:2]}."
            )
        mask = mask & pad_mask.to(device=pred.device, dtype=torch.bool).unsqueeze(-1)

    slice_dict = motion_rep.slice_dict
    bsz, nframes, _ = pred.shape
    nbjoints = int(motion_rep.nbjoints)

    pred_pos, pred_rot = _decode_pred_fk(pred, motion_rep)

    joint_weights = _joint_constraint_weights(
        motion_rep,
        hand_constraint_weight=hand_w,
        foot_constraint_weight=foot_w,
        root_constraint_weight=root_w,
        device=pred.device,
        dtype=pred.dtype,
    )

    position_loss = _zero_like(pred)
    if "local_joints_positions" in slice_dict:
        pos_mask = mask[..., slice_dict["local_joints_positions"]].reshape(bsz, nframes, nbjoints, 3)
        if pos_mask.any():
            obs_local_pos = observed[..., slice_dict["local_joints_positions"]].reshape(bsz, nframes, nbjoints, 3)
            obs_smooth_root = observed[..., slice_dict["smooth_root_pos"]]
            target_pos = obs_local_pos.clone()
            target_pos[..., 0] += obs_smooth_root[..., None, 0]
            target_pos[..., 2] += obs_smooth_root[..., None, 2]
            pos_mask = pos_mask & torch.isfinite(pred_pos) & torch.isfinite(target_pos)
            pos_weights = joint_weights.view(1, 1, nbjoints, 1).expand_as(pred_pos)
            position_loss = _masked_weighted_l1(pred_pos, target_pos, pos_mask, pos_weights)

    rotation_loss = _zero_like(pred)
    if "global_rot_data" in slice_dict:
        rot_mask_6d = mask[..., slice_dict["global_rot_data"]].reshape(bsz, nframes, nbjoints, 6)
        rot_mask = rot_mask_6d.all(dim=-1)
        if rot_mask.any():
            obs_rot6d = observed[..., slice_dict["global_rot_data"]].reshape(bsz, nframes, nbjoints, 6)
            target_rot = _safe_cont6d_to_matrix(obs_rot6d)
            rot_err = _rotation_geodesic(pred_rot, target_rot)
            rot_mask = rot_mask & torch.isfinite(rot_err)
            rot_weights = joint_weights.view(1, 1, nbjoints).expand_as(rot_err)
            rotation_loss = _masked_weighted_mean(rot_err, rot_mask, rot_weights)

    root_loss = _zero_like(pred)
    if "smooth_root_pos" in slice_dict:
        root_pos_mask = mask[..., slice_dict["smooth_root_pos"]]
        if root_pos_mask.any():
            root_weights = torch.full_like(root_pos_mask, root_w, dtype=pred.dtype)
            root_loss = root_loss + _masked_weighted_l1(
                pred[..., slice_dict["smooth_root_pos"]],
                observed[..., slice_dict["smooth_root_pos"]],
                root_pos_mask,
                root_weights,
            )
    if "global_root_heading" in slice_dict:
        heading_mask = mask[..., slice_dict["global_root_heading"]]
        if heading_mask.any():
            heading_weights = torch.full_like(heading_mask, root_w, dtype=pred.dtype)
            root_loss = root_loss + _masked_weighted_l1(
                pred[..., slice_dict["global_root_heading"]],
                observed[..., slice_dict["global_root_heading"]],
                heading_mask,
                heading_weights,
            )

    total = position_loss + rotation_loss + root_loss
    return {
        "total": total,
        "position": position_loss,
        "rotation": rotation_loss,
        "root": root_loss,
        "observed_count": mask.to(dtype=pred.dtype).sum(),
    }


class DistillationKimodoLoss(nn.Module):
    """Weighted 7+7 distillation loss plus optional constraint loss.

    The "7+7" means:
    - 7 Kimodo terms: student vs teacher (distill branch)
    - 7 Kimodo terms: student vs GT (aux branch)

    Final scalar:
        total = teacher_weight * L_distill + gt_weight * L_gt
                + constraint_loss_weight * L_constraint

    where each branch L_* already aggregates its own 7 weighted terms.
    """

    def __init__(
        self,
        motion_rep,
        *,
        teacher_gammas: Sequence[float] = DEFAULT_GAMMAS,
        gt_gammas: Sequence[float] = DEFAULT_GAMMAS,
        teacher_weight: float = 0.8,
        gt_weight: float = 0.2,
        constraint_loss_weight: float = 0.0,
        hand_constraint_weight: float = 3.0,
        foot_constraint_weight: float = 1.5,
        root_constraint_weight: float = 1.0,
        input_is_normalized: bool = False,
    ) -> None:
        super().__init__()
        self.motion_rep = motion_rep
        self.teacher_gammas = tuple(float(x) for x in teacher_gammas)
        self.gt_gammas = tuple(float(x) for x in gt_gammas)
        self.teacher_weight = _validate_weight("teacher_weight", teacher_weight)
        self.gt_weight = _validate_weight("gt_weight", gt_weight)
        self.constraint_loss_weight = _validate_weight("constraint_loss_weight", constraint_loss_weight)
        self.hand_constraint_weight = _validate_weight("hand_constraint_weight", hand_constraint_weight)
        self.foot_constraint_weight = _validate_weight("foot_constraint_weight", foot_constraint_weight)
        self.root_constraint_weight = _validate_weight("root_constraint_weight", root_constraint_weight)
        self.input_is_normalized = bool(input_is_normalized)
        if self.teacher_weight == 0.0 and self.gt_weight == 0.0 and self.constraint_loss_weight == 0.0:
            raise ValueError("teacher_weight, gt_weight, and constraint_loss_weight cannot all be zero.")

    def forward(
        self,
        *,
        pred_x0: Tensor,
        teacher_x0: Tensor,
        gt_x0: Tensor,
        pad_mask: Optional[Tensor] = None,
        observed_motion: Optional[Tensor] = None,
        motion_mask: Optional[Tensor] = None,
        teacher_weight: Optional[float] = None,
        gt_weight: Optional[float] = None,
        constraint_loss_weight: Optional[float] = None,
    ) -> Dict[str, Tensor]:
        if pred_x0.shape != teacher_x0.shape or pred_x0.shape != gt_x0.shape:
            raise ValueError(
                "pred_x0, teacher_x0, gt_x0 must have the same shape. "
                f"Got pred={pred_x0.shape}, teacher={teacher_x0.shape}, gt={gt_x0.shape}."
            )

        teacher_terms = compute_kimodo_loss(
            pred_x0=pred_x0,
            gt_x0=teacher_x0,
            motion_rep=self.motion_rep,
            pad_mask=pad_mask,
            gammas=self.teacher_gammas,
            input_is_normalized=self.input_is_normalized,
        )
        gt_terms = compute_kimodo_loss(
            pred_x0=pred_x0,
            gt_x0=gt_x0,
            motion_rep=self.motion_rep,
            pad_mask=pad_mask,
            gammas=self.gt_gammas,
            input_is_normalized=self.input_is_normalized,
        )

        tw = self.teacher_weight if teacher_weight is None else _validate_weight("teacher_weight", teacher_weight)
        gw = self.gt_weight if gt_weight is None else _validate_weight("gt_weight", gt_weight)
        cw = (
            self.constraint_loss_weight
            if constraint_loss_weight is None
            else _validate_weight("constraint_loss_weight", constraint_loss_weight)
        )
        if tw == 0.0 and gw == 0.0 and cw == 0.0:
            raise ValueError("teacher_weight, gt_weight, and constraint_loss_weight cannot all be zero.")

        if cw > 0.0:
            constraint_terms = compute_constraint_loss(
                pred_x0=pred_x0,
                observed_motion=observed_motion,
                motion_mask=motion_mask,
                motion_rep=self.motion_rep,
                pad_mask=pad_mask,
                input_is_normalized=self.input_is_normalized,
                hand_constraint_weight=self.hand_constraint_weight,
                foot_constraint_weight=self.foot_constraint_weight,
                root_constraint_weight=self.root_constraint_weight,
            )
        else:
            z = _zero_like(pred_x0)
            constraint_terms = {
                "total": z,
                "position": z,
                "rotation": z,
                "root": z,
                "observed_count": z,
            }

        total = (tw * teacher_terms["total"]) + (gw * gt_terms["total"]) + (cw * constraint_terms["total"])

        out: Dict[str, Tensor] = {
            "total": total,
            "loss_teacher_total": teacher_terms["total"],
            "loss_gt_total": gt_terms["total"],
            "loss_constraint_total": constraint_terms["total"],
            "loss_constraint_position": constraint_terms["position"],
            "loss_constraint_rotation": constraint_terms["rotation"],
            "loss_constraint_root": constraint_terms["root"],
            "constraint_observed_count": constraint_terms["observed_count"],
            "teacher_weight": torch.as_tensor(tw, device=pred_x0.device, dtype=pred_x0.dtype),
            "gt_weight": torch.as_tensor(gw, device=pred_x0.device, dtype=pred_x0.dtype),
            "constraint_loss_weight": torch.as_tensor(cw, device=pred_x0.device, dtype=pred_x0.dtype),
        }

        for name in LOSS_NAMES:
            out[f"teacher_{name}"] = teacher_terms[name]
            out[f"teacher_weighted_{name}"] = teacher_terms[f"weighted_{name}"]
            out[f"gt_{name}"] = gt_terms[name]
            out[f"gt_weighted_{name}"] = gt_terms[f"weighted_{name}"]

        return out
