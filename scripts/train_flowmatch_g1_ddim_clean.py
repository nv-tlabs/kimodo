#!/usr/bin/env python3
"""Continuous Flow-Matching distillation for Kimodo G1 using DDIM final clean samples.

This is the teacher-as-data FM route:
  teacher 100-step DDIM -> x_clean_teacher
  x_tau = (1 - tau) * x_clean + tau * z, with tau=1 noise and tau=0 clean
  v_target = dx/dtau = z - x_clean
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import time
import types
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import torch
import torch.distributed as dist
from omegaconf import DictConfig, OmegaConf
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP

from kimodo.distillation.train import (
    apply_cfg_dropout,
    build_dataset,
    build_dataset_motion_rep,
    build_device,
    cleanup_old_checkpoints,
    compute_first_heading_angle,
    cleanup_distributed,
    copy_teacher_layers_to_student,
    create_dataloader,
    create_keyframe_scheduler,
    encode_text_batch,
    get_batch,
    is_main_process,
    load_resume_checkpoint,
    load_text_encoder,
    make_autocast,
    maybe_init_wandb,
    maybe_to_device,
    reduce_scalar,
    save_checkpoint,
    setup_distributed,
    should_use_phase2,
)
from kimodo.model.diffusion import DDIMSampler, Diffusion
from kimodo.model.loading import instantiate_from_dict
from kimodo.tools import seed_everything
from kimodo.training.loss import compute_kimodo_loss
from kimodo.training.ema import EMA
from kimodo.training.optimizers import build_optimizer

log = logging.getLogger(__name__)

LOSS_NAMES: tuple[str, ...] = (
    "smooth_root_pos",
    "global_root_heading",
    "local_joints_positions",
    "velocities",
    "global_rot_data",
    "foot_contacts",
    "fk_consistency",
)
DEFAULT_GAMMAS: tuple[float, ...] = (10.0, 2.0, 10.0, 3.0, 10.0, 4.0, 5.0)


def _patch_continuous_timestep_embedding(model: nn.Module, num_base_steps: int) -> None:
    """Patch timestep embedders to accept float timesteps via linear interpolation."""
    for m in model.modules():
        if not hasattr(m, "embed_timestep"):
            continue
        emb = m.embed_timestep
        if getattr(emb, "_continuous_patch_applied", False):
            continue
        orig_forward = emb.forward

        def _forward_cont(self, timesteps, _orig_forward=orig_forward):
            if torch.is_floating_point(timesteps):
                t = timesteps.clamp(0.0, float(num_base_steps - 1))
                t0 = torch.floor(t).to(torch.long)
                t1 = torch.clamp(t0 + 1, max=num_base_steps - 1)
                # Keep shape aligned with TimestepEmbedder output [B, 1, D].
                w = (t - t0.to(t.dtype)).view(-1, 1, 1)
                e0 = _orig_forward(t0)
                e1 = _orig_forward(t1)
                return e0 * (1.0 - w) + e1 * w
            return _orig_forward(timesteps.to(torch.long))

        emb.forward = types.MethodType(_forward_cont, emb)
        emb._continuous_patch_applied = True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train continuous FM student from DDIM final clean samples.")
    parser.add_argument(
        "--config",
        type=str,
        default="kimodo/distillation/configs/flowmatch_g1_ddim_clean_teacher100_gt_50k_bs4x4.yaml",
        help="Path to flow-matching distillation config yaml.",
    )
    parser.add_argument("--resume", type=str, default=None, help="Optional checkpoint to resume from.")
    return parser.parse_args()


def _copy_matching_params(dst_model: nn.Module, src_model: nn.Module) -> tuple[int, int]:
    dst_state = dst_model.state_dict()
    src_state = src_model.state_dict()
    copied = 0
    for k, v in dst_state.items():
        src_v = src_state.get(k, None)
        if src_v is not None and src_v.shape == v.shape:
            dst_state[k] = src_v.detach().clone()
            copied += 1
    dst_model.load_state_dict(dst_state, strict=True)
    return copied, len(dst_state)


def _linear_weight_schedule(cfg: DictConfig, global_step: int, total_steps: int) -> tuple[float, float]:
    schedule = cfg.loss.weight_schedule
    if not bool(schedule.enabled):
        return float(cfg.loss.teacher_weight), float(cfg.loss.gt_weight)

    start_step = int(schedule.start_step)
    end_step = int(schedule.end_step)
    t0 = float(schedule.teacher_start)
    t1 = float(schedule.teacher_end)
    g0 = float(schedule.gt_start)
    g1 = float(schedule.gt_end)
    if end_step <= start_step:
        return t1, g1
    if global_step <= start_step:
        return t0, g0
    if global_step >= end_step:
        return t1, g1
    alpha = float(global_step - start_step) / float(end_step - start_step)
    return t0 + (t1 - t0) * alpha, g0 + (g1 - g0) * alpha


def set_lr_for_step(optimizer: torch.optim.Optimizer, cfg: DictConfig, global_step: int, total_steps: int) -> float:
    """Apply optional warmup+cosine/linear LR decay and return the active LR."""
    base_lr = float(cfg.training.lr)
    schedule_cfg = cfg.training.get("lr_schedule", None)
    if schedule_cfg is None or not bool(schedule_cfg.get("enabled", False)):
        lr = base_lr
    else:
        schedule_type = str(schedule_cfg.get("type", "cosine")).lower()
        warmup_steps = int(schedule_cfg.get("warmup_steps", 0))
        min_lr = float(schedule_cfg.get("min_lr", base_lr))
        if warmup_steps > 0 and global_step < warmup_steps:
            lr = base_lr * float(global_step + 1) / float(warmup_steps)
        elif schedule_type == "cosine":
            decay_steps = max(1, int(total_steps) - max(0, warmup_steps))
            progress = min(1.0, max(0.0, float(global_step - warmup_steps) / float(decay_steps)))
            lr = min_lr + 0.5 * (base_lr - min_lr) * (1.0 + math.cos(math.pi * progress))
        elif schedule_type == "linear":
            decay_steps = max(1, int(total_steps) - max(0, warmup_steps))
            progress = min(1.0, max(0.0, float(global_step - warmup_steps) / float(decay_steps)))
            lr = base_lr + (min_lr - base_lr) * progress
        else:
            raise ValueError(f"Unsupported training.lr_schedule.type={schedule_type!r}")

    for group in optimizer.param_groups:
        group["lr"] = lr
    return lr


@torch.no_grad()
def _teacher_clean_sample(
    *,
    teacher: nn.Module,
    diffusion: Diffusion,
    sampler: DDIMSampler,
    teacher_steps: int,
    gt_x0: torch.Tensor,
    pad_mask: torch.Tensor,
    text_feat: torch.Tensor,
    text_pad_mask: torch.Tensor,
    observed_motion: Optional[torch.Tensor],
    motion_mask: Optional[torch.Tensor],
    first_heading_angle: Optional[torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample final clean motion from the DDIM teacher.

    Returns (z, x_clean_teacher). Kimodo/DDIM convention here is high schedule
    index = noisy side and low schedule index = clean side.
    """
    bsz = gt_x0.shape[0]
    z = torch.randn_like(gt_x0)
    x = z
    use_timesteps, map_tensor = diffusion.space_timesteps(int(teacher_steps))
    diffusion.calc_diffusion_vars(use_timesteps)
    # iterate over schedule indices from high-noise -> low-noise
    for step_idx in reversed(range(1, int(teacher_steps))):
        t_idx = torch.full((bsz,), step_idx, device=gt_x0.device, dtype=torch.long)
        t_map = map_tensor[t_idx]
        pred_x0 = teacher(
            x=x,
            x_pad_mask=pad_mask,
            text_feat=text_feat,
            text_feat_pad_mask=text_pad_mask,
            timesteps=t_map,
            first_heading_angle=first_heading_angle,
            motion_mask=motion_mask,
            observed_motion=observed_motion,
        )
        x = sampler(use_timesteps, x, pred_x0, t_idx)
    return z, x


def _masked_mse(pred: torch.Tensor, target: torch.Tensor, pad_mask: torch.Tensor) -> torch.Tensor:
    # pred/target: [B,T,D], pad_mask: [B,T]
    w = pad_mask.to(dtype=pred.dtype).unsqueeze(-1)
    denom = torch.clamp(w.sum() * pred.shape[-1], min=1.0)
    return (((pred - target) ** 2) * w).sum() / denom


def _masked_rmse(pred: torch.Tensor, target: torch.Tensor, pad_mask: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(_masked_mse(pred, target, pad_mask).clamp(min=0.0))


def _validate_gammas(gammas: Sequence[float]) -> tuple[float, ...]:
    values = tuple(float(x) for x in gammas)
    if len(values) != 7:
        raise ValueError(f"Expected 7 gammas, got {len(values)}.")
    return values


def _repeat_batch(x: Optional[torch.Tensor], repeats: int) -> Optional[torch.Tensor]:
    if x is None or repeats == 1:
        return x
    return x.repeat_interleave(int(repeats), dim=0)


def _weighted_velocity_field_mse(
    *,
    pred_v: torch.Tensor,
    target_v: torch.Tensor,
    pad_mask: torch.Tensor,
    motion_rep,
    gammas: Sequence[float],
) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    gammas = _validate_gammas(gammas)
    slices = motion_rep.slice_dict
    term_pairs = [
        ("smooth_root_pos", "smooth_root_pos"),
        ("global_root_heading", "global_root_heading"),
        ("local_joints_positions", "local_joints_positions"),
        ("velocities", "velocities"),
        ("global_rot_data", "global_rot_data"),
    ]
    weighted_terms: Dict[str, torch.Tensor] = {}
    total = pred_v.new_tensor(0.0)
    for name, slice_key in term_pairs:
        gamma = gammas[LOSS_NAMES.index(name)]
        raw = _masked_mse(pred_v[..., slices[slice_key]], target_v[..., slices[slice_key]], pad_mask)
        weighted = float(gamma) * raw
        weighted_terms[name] = weighted
        total = total + weighted
    weighted_terms["foot_contacts"] = pred_v.new_zeros(())
    weighted_terms["fk_consistency"] = pred_v.new_zeros(())
    return total, weighted_terms


def _clean_contact_fk_aux_loss(
    *,
    pred_clean: torch.Tensor,
    target_clean: torch.Tensor,
    pad_mask: torch.Tensor,
    motion_rep,
    gammas: Sequence[float],
    input_is_normalized: bool,
) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    gammas = _validate_gammas(gammas)
    aux_gammas = [0.0, 0.0, 0.0, 0.0, 0.0, gammas[5], gammas[6]]
    loss_dict = compute_kimodo_loss(
        pred_x0=pred_clean,
        gt_x0=target_clean,
        motion_rep=motion_rep,
        pad_mask=pad_mask,
        gammas=aux_gammas,
        input_is_normalized=input_is_normalized,
    )
    terms = {
        "foot_contacts": loss_dict["weighted_foot_contacts"],
        "fk_consistency": loss_dict["weighted_fk_consistency"],
    }
    return loss_dict["total"], terms


def main() -> None:
    args = parse_args()
    rank, local_rank, world_size, is_distributed = setup_distributed()
    device = build_device(local_rank)

    logging.basicConfig(
        level=logging.INFO if is_main_process(rank) else logging.WARNING,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    cfg = OmegaConf.load(args.config)
    os.environ.setdefault("KIMODO_DISABLE_ROOT_SMOOTH", "1")
    seed_everything(int(cfg.training.seed) + rank, deterministic=bool(cfg.training.deterministic))

    output_dir = Path(cfg.training.output_dir)
    ckpt_dir = output_dir / "checkpoints"
    log_path = output_dir / "train_log.jsonl"
    if is_main_process(rank):
        output_dir.mkdir(parents=True, exist_ok=True)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        OmegaConf.save(cfg, output_dir / "resolved_config.yaml")
    wandb_run = maybe_init_wandb(cfg, rank=rank, output_dir=output_dir)

    teacher_cfg = OmegaConf.to_container(cfg.model.teacher_denoiser, resolve=True)
    student_cfg = OmegaConf.to_container(cfg.model.student_denoiser, resolve=True)
    teacher = instantiate_from_dict(teacher_cfg).to(device)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False
    student = instantiate_from_dict(student_cfg).to(device)
    if bool(cfg.distillation.get("student_init_from_teacher", True)):
        copied, total = _copy_matching_params(student, teacher)
        init_mode = str(cfg.distillation.get("student_init_mode", "matching_params")).lower()
        if init_mode == "teacher_layer_map":
            teacher_layer_indices = cfg.distillation.get("teacher_layer_indices_for_student", None)
            if teacher_layer_indices is None:
                raise ValueError(
                    "distillation.student_init_mode='teacher_layer_map' requires "
                    "distillation.teacher_layer_indices_for_student."
                )
            layer_copied, layer_total = copy_teacher_layers_to_student(
                student,
                teacher,
                teacher_layer_indices_for_student=[int(x) for x in teacher_layer_indices],
            )
            if is_main_process(rank):
                log.info(
                    "Applied teacher_layer_map overlay: copied %d/%d layer params",
                    layer_copied,
                    layer_total,
                )
        elif init_mode != "matching_params":
            raise ValueError(
                f"Unsupported distillation.student_init_mode={init_mode!r}; "
                "use 'matching_params' or 'teacher_layer_map'."
            )
        if is_main_process(rank):
            log.info("Warm-started student from teacher: copied %d/%d params", copied, total)

    diffusion = Diffusion(num_base_steps=int(cfg.model.num_base_steps)).to(device)
    _patch_continuous_timestep_embedding(teacher, diffusion.num_base_steps)
    _patch_continuous_timestep_embedding(student, diffusion.num_base_steps)
    sampler = DDIMSampler(diffusion)
    dataset_motion_rep = build_dataset_motion_rep(cfg)
    text_encoder = load_text_encoder(cfg, device)
    dataset = build_dataset(cfg, motion_rep=dataset_motion_rep, rank=rank)
    scheduler = create_keyframe_scheduler(cfg)
    loader, sampler_dist = create_dataloader(dataset, cfg, is_distributed=is_distributed)

    optimizer = build_optimizer(cfg, student)
    autocast_ctx, use_scaler = make_autocast(cfg.training.mixed_precision, device)
    scaler = torch.cuda.amp.GradScaler(enabled=use_scaler)
    ema = EMA(student, decay=float(cfg.training.ema_decay))

    train_model: nn.Module = student
    if is_distributed:
        train_model = DDP(
            student,
            device_ids=[local_rank] if device.type == "cuda" else None,
            output_device=local_rank if device.type == "cuda" else None,
            find_unused_parameters=False,
        )

    total_steps = int(cfg.training.total_steps)
    log_every = int(cfg.training.log_every)
    save_every = int(cfg.training.save_every)
    max_ckpt_keep = int(cfg.training.get("max_checkpoints_to_keep", 0))
    grad_clip = float(cfg.training.gradient_clip)
    cfg_text_drop = float(cfg.training.cfg_dropout_text_prob)
    cfg_motion_drop = float(cfg.training.cfg_dropout_motion_prob)
    skip_text_encode_errors = bool(cfg.training.get("skip_text_encode_errors", False))
    max_text_encode_retries = int(cfg.training.get("max_text_encode_retries", 8))
    teacher_steps = int(cfg.distillation.teacher_steps)
    student_steps = int(cfg.distillation.student_steps)
    tau_per_sample = max(1, int(cfg.flow_matching.get("tau_per_sample", 1)))
    teacher_gammas = _validate_gammas(cfg.loss.get("teacher_gammas", list(DEFAULT_GAMMAS)))
    gt_gammas = _validate_gammas(cfg.loss.get("gt_gammas", list(DEFAULT_GAMMAS)))
    if student_steps != 20 or teacher_steps != 100:
        log.warning("Config requests teacher_steps=%d, student_steps=%d (recommended: teacher100->student20 for this setup).", teacher_steps, student_steps)
    if is_main_process(rank):
        log.info("Using tau_per_sample=%d", tau_per_sample)

    start_step = 0
    epoch = 0
    resume_path = args.resume or cfg.training.get("resume", None)
    if resume_path:
        ckpt = load_resume_checkpoint(
            Path(resume_path),
            student=student,
            optimizer=optimizer,
            scaler=scaler if use_scaler else None,
            ema=ema,
            device=device,
        )
        start_step = int(ckpt.get("step", -1)) + 1
        epoch = int(ckpt.get("epoch", 0))
        if is_main_process(rank):
            log.info("Resumed from %s (start_step=%d, epoch=%d)", resume_path, start_step, epoch)

    iterator = iter(loader)
    if is_distributed and sampler_dist is not None:
        sampler_dist.set_epoch(epoch)
    student.train(True)

    for global_step in range(start_step, total_steps):
        if scheduler is not None:
            if should_use_phase2(global_step, cfg):
                dataset.set_phase2_num_keyframes(int(scheduler(global_step)))
            else:
                dataset.set_phase2_num_keyframes(0)

        text_encode_exc: Optional[Exception] = None
        for encode_attempt in range(1, max_text_encode_retries + 2):
            batch, iterator, epoch = get_batch(iterator, loader, sampler_dist, epoch, is_distributed)
            gt_x0 = batch["motion"].to(device=device, dtype=torch.float32)
            pad_mask = batch["pad_mask"].to(device=device, dtype=torch.bool)

            observed_motion = maybe_to_device(batch["observed_motion"], device)
            motion_mask = maybe_to_device(batch["motion_mask"], device)
            if observed_motion is not None:
                observed_motion = observed_motion.to(dtype=gt_x0.dtype)
            if motion_mask is not None:
                motion_mask = motion_mask.to(dtype=gt_x0.dtype)

            try:
                text_feat, text_pad_mask = encode_text_batch(text_encoder, batch["text"], device=device)
            except Exception as exc:  # noqa: BLE001
                text_encode_exc = exc
                if (not skip_text_encode_errors) or (encode_attempt > max_text_encode_retries):
                    raise
                if is_main_process(rank):
                    log.warning(
                        "Skipping batch at step=%d due to text encoder error (%s: %s), retry %d/%d.",
                        global_step,
                        type(exc).__name__,
                        exc,
                        encode_attempt,
                        max_text_encode_retries + 1,
                    )
                continue

            text_feat, text_pad_mask, observed_motion, motion_mask = apply_cfg_dropout(
                text_feat=text_feat,
                text_pad_mask=text_pad_mask,
                observed_motion=observed_motion,
                motion_mask=motion_mask,
                text_dropout_prob=cfg_text_drop,
                motion_dropout_prob=cfg_motion_drop,
            )
            text_encode_exc = None
            break

        if text_encode_exc is not None:
            raise RuntimeError(
                f"Text encoding failed for step={global_step} after {max_text_encode_retries + 1} attempts."
            ) from text_encode_exc

        first_heading_angle = None
        if bool(cfg.model.student_denoiser.get("input_first_heading_angle", False)):
            first_heading_angle = compute_first_heading_angle(
                x0=gt_x0,
                motion_rep=student.motion_rep,
                input_is_normalized=bool(cfg.data.dataset.to_normalize),
            )

        # Continuous FM convention in this repo:
        #   tau=1 -> Gaussian noise / high Kimodo timestep
        #   tau=0 -> clean motion / low Kimodo timestep
        # Straight-line path:
        #   x_tau = (1 - tau) * x_clean + tau * z
        #   v = dx/dtau = z - x_clean
        tau = torch.rand((gt_x0.shape[0] * tau_per_sample, 1, 1), device=device, dtype=gt_x0.dtype)

        with torch.no_grad():
            with autocast_ctx():
                z_teacher, teacher_clean = _teacher_clean_sample(
                    teacher=teacher,
                    diffusion=diffusion,
                    sampler=sampler,
                    teacher_steps=teacher_steps,
                    gt_x0=gt_x0,
                    pad_mask=pad_mask,
                    text_feat=text_feat,
                    text_pad_mask=text_pad_mask,
                    observed_motion=observed_motion,
                    motion_mask=motion_mask,
                    first_heading_angle=first_heading_angle,
                )

        gt_x0_tau = _repeat_batch(gt_x0, tau_per_sample)
        pad_mask_tau = _repeat_batch(pad_mask, tau_per_sample)
        text_feat_tau = _repeat_batch(text_feat, tau_per_sample)
        text_pad_mask_tau = _repeat_batch(text_pad_mask, tau_per_sample)
        observed_motion_tau = _repeat_batch(observed_motion, tau_per_sample)
        motion_mask_tau = _repeat_batch(motion_mask, tau_per_sample)
        first_heading_angle_tau = _repeat_batch(first_heading_angle, tau_per_sample)

        teacher_clean_tau = _repeat_batch(teacher_clean, tau_per_sample)
        z_teacher_tau = _repeat_batch(z_teacher, tau_per_sample)
        x_teacher_tau = (1.0 - tau) * teacher_clean_tau + tau * z_teacher_tau
        v_teacher = z_teacher_tau - teacher_clean_tau

        # GT branch target: linear Gaussian path (standard FM baseline).
        z = torch.randn_like(gt_x0_tau)
        x_tau = (1.0 - tau) * gt_x0_tau + tau * z
        v_gt = z - gt_x0_tau

        # Continuous-time input mapped to base-step index domain [0, num_base_steps-1].
        t_cont = tau[:, 0, 0] * float(diffusion.num_base_steps - 1)

        active_lr = set_lr_for_step(optimizer, cfg, global_step, total_steps)
        optimizer.zero_grad(set_to_none=True)
        with autocast_ctx():
            # Student predicts continuous-time velocity field (same tensor shape as x).
            timesteps_teacher = t_cont.to(device=device, dtype=gt_x0.dtype)
            pred_v_teacher = train_model(
                x=x_teacher_tau,
                x_pad_mask=pad_mask_tau,
                text_feat=text_feat_tau,
                text_feat_pad_mask=text_pad_mask_tau,
                timesteps=timesteps_teacher,
                first_heading_angle=first_heading_angle_tau,
                motion_mask=motion_mask_tau,
                observed_motion=observed_motion_tau,
            )

            timesteps_tau = t_cont.to(device=device, dtype=gt_x0.dtype)
            pred_v_gt = train_model(
                x=x_tau,
                x_pad_mask=pad_mask_tau,
                text_feat=text_feat_tau,
                text_feat_pad_mask=text_pad_mask_tau,
                timesteps=timesteps_tau,
                first_heading_angle=first_heading_angle_tau,
                motion_mask=motion_mask_tau,
                observed_motion=observed_motion_tau,
            )

            loss_teacher, terms_teacher = _weighted_velocity_field_mse(
                pred_v=pred_v_teacher,
                target_v=v_teacher,
                pad_mask=pad_mask_tau,
                motion_rep=student.motion_rep,
                gammas=teacher_gammas,
            )
            loss_gt, terms_gt = _weighted_velocity_field_mse(
                pred_v=pred_v_gt,
                target_v=v_gt,
                pad_mask=pad_mask_tau,
                motion_rep=student.motion_rep,
                gammas=gt_gammas,
            )
            teacher_clean_pred = x_teacher_tau - tau * pred_v_teacher
            gt_clean_pred = x_tau - tau * pred_v_gt
            teacher_aux, teacher_aux_terms = _clean_contact_fk_aux_loss(
                pred_clean=teacher_clean_pred,
                target_clean=teacher_clean_tau,
                pad_mask=pad_mask_tau,
                motion_rep=student.motion_rep,
                gammas=teacher_gammas,
                input_is_normalized=bool(cfg.data.dataset.to_normalize),
            )
            gt_aux, gt_aux_terms = _clean_contact_fk_aux_loss(
                pred_clean=gt_clean_pred,
                target_clean=gt_x0_tau,
                pad_mask=pad_mask_tau,
                motion_rep=student.motion_rep,
                gammas=gt_gammas,
                input_is_normalized=bool(cfg.data.dataset.to_normalize),
            )
            loss_teacher = loss_teacher + teacher_aux
            loss_gt = loss_gt + gt_aux
            terms_teacher.update(teacher_aux_terms)
            terms_gt.update(gt_aux_terms)
            raw_teacher_mse = _masked_mse(pred_v_teacher, v_teacher, pad_mask_tau)
            raw_gt_mse = _masked_mse(pred_v_gt, v_gt, pad_mask_tau)
            raw_teacher_rmse = _masked_rmse(pred_v_teacher, v_teacher, pad_mask_tau)
            raw_gt_rmse = _masked_rmse(pred_v_gt, v_gt, pad_mask_tau)
            tw, gw = _linear_weight_schedule(cfg, global_step=global_step, total_steps=total_steps)
            loss = float(tw) * loss_teacher + float(gw) * loss_gt

        if use_scaler:
            scaler.scale(loss).backward()
            if grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(student.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(student.parameters(), grad_clip)
            optimizer.step()

        if (global_step + 1) % int(cfg.training.ema_every) == 0:
            ema.update(student)

        if (global_step + 1) % log_every == 0 or global_step == 0:
            reduced_total = reduce_scalar(loss.detach(), world_size, is_distributed)
            reduced_teacher = reduce_scalar(loss_teacher.detach(), world_size, is_distributed)
            reduced_gt = reduce_scalar(loss_gt.detach(), world_size, is_distributed)
            reduced_raw_teacher_mse = reduce_scalar(raw_teacher_mse.detach(), world_size, is_distributed)
            reduced_raw_gt_mse = reduce_scalar(raw_gt_mse.detach(), world_size, is_distributed)
            reduced_raw_teacher_rmse = reduce_scalar(raw_teacher_rmse.detach(), world_size, is_distributed)
            reduced_raw_gt_rmse = reduce_scalar(raw_gt_rmse.detach(), world_size, is_distributed)
            if is_main_process(rank):
                metric = {
                    "step": global_step,
                    "epoch": epoch,
                    "num_keyframes": int(dataset.get_phase2_num_keyframes()),
                    "tau_per_sample": int(tau_per_sample),
                    "motion_batch_size": int(gt_x0.shape[0]),
                    "effective_tau_batch_size": int(gt_x0_tau.shape[0]),
                    "loss_total": float(reduced_total.item()),
                    "loss_teacher_flow": float(reduced_teacher.item()),
                    "loss_gt_flow": float(reduced_gt.item()),
                    "raw_teacher_v_mse": float(reduced_raw_teacher_mse.item()),
                    "raw_gt_v_mse": float(reduced_raw_gt_mse.item()),
                    "raw_teacher_v_rmse": float(reduced_raw_teacher_rmse.item()),
                    "raw_gt_v_rmse": float(reduced_raw_gt_rmse.item()),
                    "teacher_weight": float(tw),
                    "gt_weight": float(gw),
                    "lr": float(active_lr),
                    "time": time.time(),
                }
                for name in LOSS_NAMES:
                    metric[f"teacher_{name}"] = float(terms_teacher[name].detach().item())
                    metric[f"gt_{name}"] = float(terms_gt[name].detach().item())
                log.info(
                    "step=%d loss=%.6f teacher_flow=%.6f gt_flow=%.6f raw_rmse=%.4f/%.4f tw=%.3f gw=%.3f lr=%.3e",
                    metric["step"],
                    metric["loss_total"],
                    metric["loss_teacher_flow"],
                    metric["loss_gt_flow"],
                    metric["raw_teacher_v_rmse"],
                    metric["raw_gt_v_rmse"],
                    metric["teacher_weight"],
                    metric["gt_weight"],
                    metric["lr"],
                )
                with log_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(metric, ensure_ascii=False) + "\n")
                if wandb_run is not None:
                    wandb_run.log(metric, step=global_step)

        if is_main_process(rank) and (((global_step + 1) % save_every == 0) or (global_step + 1 == total_steps)):
            ckpt_path = ckpt_dir / f"step_{global_step + 1:08d}.pt"
            save_checkpoint(
                ckpt_path,
                step=global_step,
                epoch=epoch,
                student=student,
                optimizer=optimizer,
                scaler=scaler if use_scaler else None,
                ema=ema,
                cfg=cfg,
            )
            cleanup_old_checkpoints(ckpt_dir, max_keep=max_ckpt_keep)
            log.info("Saved checkpoint to %s", ckpt_path)

    if is_main_process(rank):
        ema_path = output_dir / "ema_final.pt"
        torch.save({"ema": ema.state_dict(), "step": total_steps - 1}, ema_path)
        log.info("Saved final EMA to %s", ema_path)
        if wandb_run is not None:
            wandb_run.finish()

    cleanup_distributed(is_distributed)


if __name__ == "__main__":
    main()
