#!/usr/bin/env python3
"""DAgger-style diffusion distillation for Kimodo G1 with GT branch and self-acc regularization.

This fine-tunes a previously distilled diffusion student by:
1. rolling out the current student for 20 DDIM steps,
2. sampling one or more visited denoising states per rollout,
3. asking the 100-step teacher denoiser to relabel those off-policy states,
4. training the student with teacher + GT losses plus optional self-acc regularization.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import time
from pathlib import Path
from typing import Any, Dict, Iterator, Optional

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
    cleanup_distributed,
    cleanup_old_checkpoints,
    compute_first_heading_angle,
    copy_matching_params,
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
from kimodo.distillation.loss import compute_constraint_loss, constraint_feature_acceleration_loss
from kimodo.training.loss import LOSS_NAMES, compute_kimodo_loss
from kimodo.model.diffusion import DDIMSampler, Diffusion
from kimodo.model.loading import instantiate_from_dict
from kimodo.tools import seed_everything
from kimodo.training.ema import EMA
from kimodo.training.optimizers import build_optimizer

log = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="DAgger fine-tune 100->20 Kimodo G1 with GT branch and self-acc regularization."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="kimodo/distillation/configs/distill_g1_100_to_20_dagger_teacher_gt03_50k_bs4x4_k3_cosine_selfacc10.yaml",
        help="Path to DAgger distillation config yaml.",
    )
    parser.add_argument("--resume", type=str, default=None, help="Optional training checkpoint to resume.")
    return parser.parse_args()


def make_student_schedule(diffusion: Diffusion, student_steps: int) -> tuple[torch.Tensor, torch.Tensor]:
    use_timesteps, map_tensor = diffusion.space_timesteps(int(student_steps))
    diffusion.calc_diffusion_vars(use_timesteps)
    return use_timesteps, map_tensor


def _resolve_student_state_dict(ckpt_path: str) -> tuple[dict[str, torch.Tensor], str]:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if not isinstance(ckpt, dict):
        raise ValueError(f"Unsupported student checkpoint format: {ckpt_path}")

    if "student" in ckpt and isinstance(ckpt["student"], dict):
        return ckpt["student"], "student"
    if "ema" in ckpt and isinstance(ckpt["ema"], dict) and "shadow" in ckpt["ema"]:
        shadow = ckpt["ema"]["shadow"]
        if not isinstance(shadow, dict):
            raise ValueError(f"Invalid ema.shadow format in checkpoint: {ckpt_path}")
        return shadow, "ema.shadow"
    if all(isinstance(k, str) for k in ckpt.keys()) and any(isinstance(v, torch.Tensor) for v in ckpt.values()):
        return ckpt, "state_dict"
    raise ValueError(
        "Student checkpoint must contain one of {'student': state_dict}, "
        "{'ema': {'shadow': state_dict}}, or be a plain state_dict. "
        f"Got keys={list(ckpt.keys())[:10]} from {ckpt_path}"
    )


def load_student_weights(student: nn.Module, ckpt_path: str) -> str:
    state, source = _resolve_student_state_dict(ckpt_path)
    missing, unexpected = student.load_state_dict(state, strict=False)
    if unexpected:
        raise ValueError(f"Unexpected keys when loading {source} from {ckpt_path}: first 10={unexpected[:10]}")
    if missing:
        raise ValueError(f"Missing keys when loading {source} from {ckpt_path}: first 10={missing[:10]}")
    return source


@torch.no_grad()
def rollout_student_states(
    *,
    student: nn.Module,
    diffusion: Diffusion,
    sampler: DDIMSampler,
    use_timesteps: torch.Tensor,
    map_tensor: torch.Tensor,
    student_steps: int,
    gt_x0: torch.Tensor,
    pad_mask: torch.Tensor,
    text_feat: torch.Tensor,
    text_pad_mask: torch.Tensor,
    observed_motion: Optional[torch.Tensor],
    motion_mask: Optional[torch.Tensor],
    first_heading_angle: Optional[torch.Tensor],
) -> torch.Tensor:
    """Return visited states indexed by DDIM schedule index [0..student_steps-1].

    states[i] is the noisy input that the denoiser would see at schedule index i.
    states[student_steps - 1] is the initial Gaussian noise; states[0] is the
    input to the final denoising step, not the post-step x0 sample.
    """
    bsz = gt_x0.shape[0]
    x = torch.randn_like(gt_x0)
    states = gt_x0.new_empty((int(student_steps), *gt_x0.shape))
    states[int(student_steps) - 1] = x

    for schedule_idx in reversed(range(1, int(student_steps))):
        t_idx = torch.full((bsz,), schedule_idx, device=gt_x0.device, dtype=torch.long)
        t_map = map_tensor[t_idx]
        pred_x0 = student(
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
        states[schedule_idx - 1] = x
    return states


def sample_rollout_training_points(
    states: torch.Tensor,
    *,
    samples_per_rollout: int,
    include_t0: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample K schedule indices per rollout and gather states.

    Returns x_t [B*K,T,D] and schedule_idx [B*K].
    """
    student_steps, bsz = int(states.shape[0]), int(states.shape[1])
    low = 0 if include_t0 else 1
    if low >= student_steps:
        raise ValueError(f"No valid schedule indices: include_t0={include_t0}, student_steps={student_steps}")

    schedule_idx = torch.randint(
        low,
        student_steps,
        (bsz, int(samples_per_rollout)),
        device=states.device,
        dtype=torch.long,
    )
    batch_idx = torch.arange(bsz, device=states.device)[:, None].expand_as(schedule_idx)
    x_t = states[schedule_idx, batch_idx]
    return x_t.reshape(bsz * int(samples_per_rollout), *states.shape[2:]), schedule_idx.reshape(-1)


def repeat_optional(x: Optional[torch.Tensor], repeats: int) -> Optional[torch.Tensor]:
    if x is None:
        return None
    return x.repeat_interleave(int(repeats), dim=0)


def get_dataset_num_keyframes(dataset: Any) -> int:
    getter = getattr(dataset, "get_phase2_num_keyframes", None)
    if callable(getter):
        return int(getter())
    return 0


def masked_second_difference_l1(pred: torch.Tensor, pad_mask: torch.Tensor) -> torch.Tensor:
    if pred.shape[1] < 3:
        return pred.new_zeros(())
    acc = pred[:, 2:] - 2.0 * pred[:, 1:-1] + pred[:, :-2]
    valid = pad_mask[:, 2:] & pad_mask[:, 1:-1] & pad_mask[:, :-2]
    while valid.ndim < acc.ndim:
        valid = valid.unsqueeze(-1)
    valid = valid.expand_as(acc).to(dtype=acc.dtype)
    return (acc.abs() * valid).sum() / valid.sum().clamp(min=1.0)


def self_acceleration_loss(
    pred_x0: torch.Tensor,
    *,
    motion_rep: Any,
    pad_mask: torch.Tensor,
    input_is_normalized: bool,
) -> torch.Tensor:
    """Simple self-acceleration regularizer on predicted clean motion features."""
    if pred_x0.shape[1] < 3:
        return pred_x0.new_zeros(())

    pred = pred_x0
    if input_is_normalized:
        pred = motion_rep.unnormalize(pred)

    foot_slice = getattr(motion_rep, "slice_dict", {}).get("foot_contacts", None)
    if foot_slice is not None:
        keep = torch.ones(pred.shape[-1], dtype=torch.bool, device=pred.device)
        keep[foot_slice] = False
        pred = pred[..., keep]

    return masked_second_difference_l1(pred, pad_mask)


def feature_slice_acceleration_loss(
    pred_x0: torch.Tensor,
    *,
    motion_rep: Any,
    pad_mask: torch.Tensor,
    input_is_normalized: bool,
    feature_names: tuple[str, ...],
) -> torch.Tensor:
    """Self-acceleration regularizer for the first available named feature slice."""
    slice_dict = getattr(motion_rep, "slice_dict", {})
    feature_slice = None
    for feature_name in feature_names:
        feature_slice = slice_dict.get(feature_name, None)
        if feature_slice is not None:
            break
    if feature_slice is None:
        return pred_x0.new_zeros(())

    pred = pred_x0
    if input_is_normalized:
        pred = motion_rep.unnormalize(pred)
    return masked_second_difference_l1(pred[..., feature_slice], pad_mask)


def set_lr_for_step(optimizer: torch.optim.Optimizer, cfg: DictConfig, global_step: int, total_steps: int) -> float:
    """Apply optional warmup+cosine LR decay and return the active LR."""
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


def maybe_load_initial_student(cfg: DictConfig, student: nn.Module, teacher: nn.Module, rank: int) -> None:
    dagger_cfg = cfg.get("dagger", None)
    initial_ckpt = None if dagger_cfg is None else dagger_cfg.get("initial_student_ckpt", None)
    if initial_ckpt:
        source = load_student_weights(student, str(initial_ckpt))
        if is_main_process(rank):
            log.info("Loaded initial student from %s (%s)", initial_ckpt, source)
        return

    if bool(cfg.distillation.get("student_init_from_teacher", True)):
        copied, total = copy_matching_params(student, teacher)
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
                log.info("Applied teacher_layer_map overlay: copied %d/%d layer params", layer_copied, layer_total)
        elif init_mode != "matching_params":
            raise ValueError(
                f"Unsupported distillation.student_init_mode={init_mode!r}; "
                "use 'matching_params' or 'teacher_layer_map'."
            )
        if is_main_process(rank):
            log.info("Warm-started student from teacher: copied %d/%d params", copied, total)


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
    os.environ.setdefault("KIMODO_LLM2VEC_DISABLE_MP", "1")
    seed_everything(int(cfg.training.seed) + rank, deterministic=bool(cfg.training.deterministic))

    output_dir = Path(cfg.training.output_dir)
    initial_ckpt = Path(str(cfg.get("dagger", {}).get("initial_student_ckpt", "")))
    if initial_ckpt and initial_ckpt.parent.resolve() == output_dir.resolve():
        raise ValueError("training.output_dir must differ from dagger.initial_student_ckpt parent to avoid overwrites.")

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
    if is_main_process(rank):
        log.info("loading teacher done")

    student = instantiate_from_dict(student_cfg).to(device)
    maybe_load_initial_student(cfg, student, teacher, rank)
    if is_main_process(rank):
        log.info("loading student done")

    diffusion = Diffusion(num_base_steps=int(cfg.model.num_base_steps)).to(device)
    sampler = DDIMSampler(diffusion)
    motion_rep = student.motion_rep

    dataset_motion_rep = build_dataset_motion_rep(cfg)
    if is_main_process(rank):
        log.info("loading text encoder ...")
    text_encoder = load_text_encoder(cfg, device)
    if is_main_process(rank):
        log.info("loading text encoder done")
        log.info("building dataset ...")
    dataset = build_dataset(cfg, motion_rep=dataset_motion_rep, rank=rank)
    if is_main_process(rank):
        log.info("building dataset done")
    scheduler = create_keyframe_scheduler(cfg)
    if is_main_process(rank):
        log.info("building dataloader ...")
    loader, sampler_dist = create_dataloader(dataset, cfg, is_distributed=is_distributed)
    if is_main_process(rank):
        log.info("building dataloader done")

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
    max_ckpt_keep = int(cfg.training.max_checkpoints_to_keep)
    grad_clip = float(cfg.training.gradient_clip)
    cfg_text_drop = float(cfg.training.cfg_dropout_text_prob)
    cfg_motion_drop = float(cfg.training.cfg_dropout_motion_prob)
    skip_text_encode_errors = bool(cfg.training.get("skip_text_encode_errors", False))
    max_text_encode_retries = int(cfg.training.get("max_text_encode_retries", 8))

    student_steps = int(cfg.distillation.student_steps)
    use_timesteps, map_tensor = make_student_schedule(diffusion, student_steps)
    dagger_cfg = cfg.get("dagger", {})
    samples_per_rollout = int(dagger_cfg.get("samples_per_rollout", 4))
    include_t0 = bool(dagger_cfg.get("include_t0", True))

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

    iterator: Iterator[Dict[str, Any]] = iter(loader)
    if is_distributed and sampler_dist is not None:
        sampler_dist.set_epoch(epoch)

    student.train(True)
    if is_main_process(rank):
        log.info("starting step loop")
    if is_main_process(rank):
        log.info(
            "Starting teacher-only DAgger distillation on %s | world_size=%d | total_steps=%d | "
            "rollout_batch=%d | samples_per_rollout=%d | effective_supervision_batch=%d | student_steps=%d",
            device,
            world_size,
            total_steps,
            int(cfg.training.batch_size),
            samples_per_rollout,
            int(cfg.training.batch_size) * samples_per_rollout,
            student_steps,
        )

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
                motion_rep=motion_rep,
                input_is_normalized=bool(cfg.data.dataset.to_normalize),
            )

        with torch.no_grad():
            with autocast_ctx():
                states = rollout_student_states(
                    student=student,
                    diffusion=diffusion,
                    sampler=sampler,
                    use_timesteps=use_timesteps,
                    map_tensor=map_tensor,
                    student_steps=student_steps,
                    gt_x0=gt_x0,
                    pad_mask=pad_mask,
                    text_feat=text_feat,
                    text_pad_mask=text_pad_mask,
                    observed_motion=observed_motion,
                    motion_mask=motion_mask,
                    first_heading_angle=first_heading_angle,
                )
                x_t, schedule_idx = sample_rollout_training_points(
                    states,
                    samples_per_rollout=samples_per_rollout,
                    include_t0=include_t0,
                )
                timesteps = map_tensor[schedule_idx]

                pad_mask_rep = pad_mask.repeat_interleave(samples_per_rollout, dim=0)
                text_feat_rep = text_feat.repeat_interleave(samples_per_rollout, dim=0)
                text_pad_mask_rep = text_pad_mask.repeat_interleave(samples_per_rollout, dim=0)
                gt_x0_rep = gt_x0.repeat_interleave(samples_per_rollout, dim=0)
                first_heading_rep = (
                    None
                    if first_heading_angle is None
                    else first_heading_angle.repeat_interleave(samples_per_rollout, dim=0)
                )
                observed_motion_rep = repeat_optional(observed_motion, samples_per_rollout)
                motion_mask_rep = repeat_optional(motion_mask, samples_per_rollout)

                teacher_x0 = teacher(
                    x=x_t,
                    x_pad_mask=pad_mask_rep,
                    text_feat=text_feat_rep,
                    text_feat_pad_mask=text_pad_mask_rep,
                    timesteps=timesteps,
                    first_heading_angle=first_heading_rep,
                    motion_mask=motion_mask_rep,
                    observed_motion=observed_motion_rep,
                )

        active_lr = set_lr_for_step(optimizer, cfg, global_step, total_steps)
        optimizer.zero_grad(set_to_none=True)
        with autocast_ctx():
            student_x0 = train_model(
                x=x_t,
                x_pad_mask=pad_mask_rep,
                text_feat=text_feat_rep,
                text_feat_pad_mask=text_pad_mask_rep,
                timesteps=timesteps,
                first_heading_angle=first_heading_rep,
                motion_mask=motion_mask_rep,
                observed_motion=observed_motion_rep,
            )
            teacher_loss_dict = compute_kimodo_loss(
                pred_x0=student_x0,
                gt_x0=teacher_x0,
                motion_rep=motion_rep,
                pad_mask=pad_mask_rep,
                gammas=cfg.loss.teacher_gammas,
                input_is_normalized=bool(cfg.data.dataset.to_normalize),
            )
            gt_loss_dict = compute_kimodo_loss(
                pred_x0=student_x0,
                gt_x0=gt_x0_rep,
                motion_rep=motion_rep,
                pad_mask=pad_mask_rep,
                gammas=cfg.loss.gt_gammas,
                input_is_normalized=bool(cfg.data.dataset.to_normalize),
            )
            teacher_weight = float(cfg.loss.get("teacher_weight", 1.0))
            gt_weight = float(cfg.loss.get("gt_weight", 0.0))
            legacy_constraint_loss_weight = float(cfg.loss.get("constraint_loss_weight", 0.0))
            teacher_constraint_loss_weight = float(
                cfg.loss.get("teacher_constraint_loss_weight", legacy_constraint_loss_weight)
            )
            gt_constraint_loss_weight = float(cfg.loss.get("gt_constraint_loss_weight", legacy_constraint_loss_weight))
            constraint_loss_weight = (
                teacher_weight * teacher_constraint_loss_weight
                + gt_weight * gt_constraint_loss_weight
            )
            loss = teacher_loss_dict["total"] * teacher_weight + gt_loss_dict["total"] * gt_weight
            constraint_loss_dict = {
                "total": student_x0.new_zeros(()),
                "position": student_x0.new_zeros(()),
                "rotation": student_x0.new_zeros(()),
                "root": student_x0.new_zeros(()),
                "observed_count": student_x0.new_zeros(()),
            }
            if teacher_constraint_loss_weight > 0.0 or gt_constraint_loss_weight > 0.0:
                constraint_loss_dict = compute_constraint_loss(
                    pred_x0=student_x0,
                    observed_motion=observed_motion_rep,
                    motion_mask=motion_mask_rep,
                    motion_rep=motion_rep,
                    pad_mask=pad_mask_rep,
                    input_is_normalized=bool(cfg.data.dataset.to_normalize),
                    hand_constraint_weight=float(cfg.loss.get("hand_constraint_weight", 3.0)),
                    foot_constraint_weight=float(cfg.loss.get("foot_constraint_weight", 1.5)),
                    root_constraint_weight=float(cfg.loss.get("root_constraint_weight", 1.0)),
                )
                loss = loss + teacher_weight * teacher_constraint_loss_weight * constraint_loss_dict["total"]
                loss = loss + gt_weight * gt_constraint_loss_weight * constraint_loss_dict["total"]
            self_acc_weight = float(cfg.loss.get("self_acc_weight", 0.0))
            root_acc_weight = float(cfg.loss.get("root_acc_weight", 0.0))
            heading_acc_weight = float(cfg.loss.get("heading_acc_weight", 0.0))
            constraint_root_acc_weight = float(cfg.loss.get("constraint_root_acc_weight", 0.0))
            constraint_heading_acc_weight = float(cfg.loss.get("constraint_heading_acc_weight", 0.0))
            constraint_hand_pos_acc_weight = float(cfg.loss.get("constraint_hand_pos_acc_weight", 0.0))
            constraint_hand_rot_acc_weight = float(cfg.loss.get("constraint_hand_rot_acc_weight", 0.0))
            self_acc_loss = student_x0.new_zeros(())
            root_acc_loss = student_x0.new_zeros(())
            heading_acc_loss = student_x0.new_zeros(())
            constraint_root_acc_loss = student_x0.new_zeros(())
            constraint_heading_acc_loss = student_x0.new_zeros(())
            constraint_hand_pos_acc_loss = student_x0.new_zeros(())
            constraint_hand_rot_acc_loss = student_x0.new_zeros(())
            if self_acc_weight > 0.0:
                self_acc_loss = self_acceleration_loss(
                    student_x0,
                    motion_rep=motion_rep,
                    pad_mask=pad_mask_rep,
                    input_is_normalized=bool(cfg.data.dataset.to_normalize),
                )
                loss = loss + self_acc_weight * self_acc_loss
            if root_acc_weight > 0.0:
                root_acc_loss = feature_slice_acceleration_loss(
                    student_x0,
                    motion_rep=motion_rep,
                    pad_mask=pad_mask_rep,
                    input_is_normalized=bool(cfg.data.dataset.to_normalize),
                    feature_names=("root_pos", "smooth_root_pos"),
                )
                loss = loss + root_acc_weight * root_acc_loss
            if heading_acc_weight > 0.0:
                heading_acc_loss = feature_slice_acceleration_loss(
                    student_x0,
                    motion_rep=motion_rep,
                    pad_mask=pad_mask_rep,
                    input_is_normalized=bool(cfg.data.dataset.to_normalize),
                    feature_names=("global_root_heading",),
                )
                loss = loss + heading_acc_weight * heading_acc_loss
            if constraint_root_acc_weight > 0.0:
                constraint_root_acc_dims = tuple(int(x) for x in cfg.loss.get("constraint_root_acc_dims", [0, 2]))
                constraint_root_acc_loss = constraint_feature_acceleration_loss(
                    pred_x0=student_x0,
                    observed_motion=observed_motion_rep,
                    motion_mask=motion_mask_rep,
                    motion_rep=motion_rep,
                    pad_mask=pad_mask_rep,
                    input_is_normalized=bool(cfg.data.dataset.to_normalize),
                    feature_name="smooth_root_pos",
                    feature_dims=constraint_root_acc_dims,
                )
                loss = loss + constraint_root_acc_weight * constraint_root_acc_loss
            if constraint_heading_acc_weight > 0.0:
                constraint_heading_acc_loss = constraint_feature_acceleration_loss(
                    pred_x0=student_x0,
                    observed_motion=observed_motion_rep,
                    motion_mask=motion_mask_rep,
                    motion_rep=motion_rep,
                    pad_mask=pad_mask_rep,
                    input_is_normalized=bool(cfg.data.dataset.to_normalize),
                    feature_name="global_root_heading",
                )
                loss = loss + constraint_heading_acc_weight * constraint_heading_acc_loss
            if constraint_hand_pos_acc_weight > 0.0:
                constraint_hand_pos_acc_dims = tuple(
                    int(x) for x in cfg.loss.get("constraint_hand_pos_acc_dims", [])
                )
                constraint_hand_pos_acc_loss = constraint_feature_acceleration_loss(
                    pred_x0=student_x0,
                    observed_motion=observed_motion_rep,
                    motion_mask=motion_mask_rep,
                    motion_rep=motion_rep,
                    pad_mask=pad_mask_rep,
                    input_is_normalized=bool(cfg.data.dataset.to_normalize),
                    feature_name="local_joints_positions",
                    feature_dims=constraint_hand_pos_acc_dims or None,
                )
                loss = loss + constraint_hand_pos_acc_weight * constraint_hand_pos_acc_loss
            if constraint_hand_rot_acc_weight > 0.0:
                constraint_hand_rot_acc_dims = tuple(
                    int(x) for x in cfg.loss.get("constraint_hand_rot_acc_dims", [])
                )
                constraint_hand_rot_acc_loss = constraint_feature_acceleration_loss(
                    pred_x0=student_x0,
                    observed_motion=observed_motion_rep,
                    motion_mask=motion_mask_rep,
                    motion_rep=motion_rep,
                    pad_mask=pad_mask_rep,
                    input_is_normalized=bool(cfg.data.dataset.to_normalize),
                    feature_name="global_rot_data",
                    feature_dims=constraint_hand_rot_acc_dims or None,
                )
                loss = loss + constraint_hand_rot_acc_weight * constraint_hand_rot_acc_loss

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

        ema_every = int(cfg.training.get("ema_every", 1))
        if (global_step + 1) % ema_every == 0:
            ema.update(student)

        if (global_step + 1) % log_every == 0 or global_step == 0:
            reduced_total = reduce_scalar(loss.detach(), world_size, is_distributed)
            reduced_teacher_total = reduce_scalar(teacher_loss_dict["total"].detach(), world_size, is_distributed)
            reduced_gt_total = reduce_scalar(gt_loss_dict["total"].detach(), world_size, is_distributed)
            reduced_constraint_total = reduce_scalar(
                constraint_loss_dict["total"].detach(),
                world_size,
                is_distributed,
            )
            reduced_constraint_pos = reduce_scalar(
                constraint_loss_dict["position"].detach(),
                world_size,
                is_distributed,
            )
            reduced_constraint_rot = reduce_scalar(
                constraint_loss_dict["rotation"].detach(),
                world_size,
                is_distributed,
            )
            reduced_constraint_root = reduce_scalar(
                constraint_loss_dict["root"].detach(),
                world_size,
                is_distributed,
            )
            reduced_self_acc = reduce_scalar(self_acc_loss.detach(), world_size, is_distributed)
            reduced_root_acc = reduce_scalar(root_acc_loss.detach(), world_size, is_distributed)
            reduced_heading_acc = reduce_scalar(heading_acc_loss.detach(), world_size, is_distributed)
            reduced_constraint_root_acc = reduce_scalar(
                constraint_root_acc_loss.detach(),
                world_size,
                is_distributed,
            )
            reduced_constraint_heading_acc = reduce_scalar(
                constraint_heading_acc_loss.detach(),
                world_size,
                is_distributed,
            )
            reduced_constraint_hand_pos_acc = reduce_scalar(
                constraint_hand_pos_acc_loss.detach(),
                world_size,
                is_distributed,
            )
            reduced_constraint_hand_rot_acc = reduce_scalar(
                constraint_hand_rot_acc_loss.detach(),
                world_size,
                is_distributed,
            )
            mean_schedule_idx = reduce_scalar(schedule_idx.float().mean(), world_size, is_distributed)
            reduced_teacher_terms = {
                name: reduce_scalar(teacher_loss_dict[name].detach(), world_size, is_distributed)
                for name in LOSS_NAMES
            }
            reduced_teacher_weighted_terms = {
                name: reduce_scalar(teacher_loss_dict[f"weighted_{name}"].detach(), world_size, is_distributed)
                for name in LOSS_NAMES
            }
            reduced_gt_terms = {
                name: reduce_scalar(gt_loss_dict[name].detach(), world_size, is_distributed)
                for name in LOSS_NAMES
            }
            reduced_gt_weighted_terms = {
                name: reduce_scalar(gt_loss_dict[f"weighted_{name}"].detach(), world_size, is_distributed)
                for name in LOSS_NAMES
            }

            if is_main_process(rank):
                metric = {
                    "step": global_step,
                    "epoch": epoch,
                    "num_keyframes": get_dataset_num_keyframes(dataset),
                    "rollout_batch_size": int(gt_x0.shape[0]),
                    "samples_per_rollout": samples_per_rollout,
                    "effective_supervision_batch": int(x_t.shape[0]),
                    "mean_schedule_idx": float(mean_schedule_idx.item()),
                    "loss_total": float(reduced_total.item()),
                    "loss_teacher_total": float(reduced_teacher_total.item()),
                    "loss_gt_total": float(reduced_gt_total.item()),
                    "loss_constraint_total": float(reduced_constraint_total.item()),
                    "loss_constraint_position": float(reduced_constraint_pos.item()),
                    "loss_constraint_rotation": float(reduced_constraint_rot.item()),
                    "loss_constraint_root": float(reduced_constraint_root.item()),
                    "teacher_weight": float(teacher_weight),
                    "gt_weight": float(gt_weight),
                    "constraint_loss_weight": float(constraint_loss_weight),
                    "legacy_constraint_loss_weight": float(legacy_constraint_loss_weight),
                    "teacher_constraint_loss_weight": float(teacher_constraint_loss_weight),
                    "gt_constraint_loss_weight": float(gt_constraint_loss_weight),
                    "teacher_constraint_weighted": float(reduced_constraint_total.item())
                    * float(teacher_weight)
                    * float(teacher_constraint_loss_weight),
                    "gt_constraint_weighted": float(reduced_constraint_total.item())
                    * float(gt_weight)
                    * float(gt_constraint_loss_weight),
                    "constraint_observed_count": float(constraint_loss_dict["observed_count"].detach().item()),
                    "self_acc_weight": float(cfg.loss.get("self_acc_weight", 0.0)),
                    "self_acc_loss": float(reduced_self_acc.item()),
                    "self_acc_weighted": float(reduced_self_acc.item()) * float(cfg.loss.get("self_acc_weight", 0.0)),
                    "root_acc_weight": float(cfg.loss.get("root_acc_weight", 0.0)),
                    "root_acc_loss": float(reduced_root_acc.item()),
                    "root_acc_weighted": float(reduced_root_acc.item()) * float(cfg.loss.get("root_acc_weight", 0.0)),
                    "heading_acc_weight": float(cfg.loss.get("heading_acc_weight", 0.0)),
                    "heading_acc_loss": float(reduced_heading_acc.item()),
                    "heading_acc_weighted": float(reduced_heading_acc.item())
                    * float(cfg.loss.get("heading_acc_weight", 0.0)),
                    "constraint_root_acc_weight": float(cfg.loss.get("constraint_root_acc_weight", 0.0)),
                    "constraint_root_acc_loss": float(reduced_constraint_root_acc.item()),
                    "constraint_root_acc_weighted": float(reduced_constraint_root_acc.item())
                    * float(cfg.loss.get("constraint_root_acc_weight", 0.0)),
                    "constraint_heading_acc_weight": float(cfg.loss.get("constraint_heading_acc_weight", 0.0)),
                    "constraint_heading_acc_loss": float(reduced_constraint_heading_acc.item()),
                    "constraint_heading_acc_weighted": float(reduced_constraint_heading_acc.item())
                    * float(cfg.loss.get("constraint_heading_acc_weight", 0.0)),
                    "constraint_hand_pos_acc_weight": float(cfg.loss.get("constraint_hand_pos_acc_weight", 0.0)),
                    "constraint_hand_pos_acc_loss": float(reduced_constraint_hand_pos_acc.item()),
                    "constraint_hand_pos_acc_weighted": float(reduced_constraint_hand_pos_acc.item())
                    * float(cfg.loss.get("constraint_hand_pos_acc_weight", 0.0)),
                    "constraint_hand_rot_acc_weight": float(cfg.loss.get("constraint_hand_rot_acc_weight", 0.0)),
                    "constraint_hand_rot_acc_loss": float(reduced_constraint_hand_rot_acc.item()),
                    "constraint_hand_rot_acc_weighted": float(reduced_constraint_hand_rot_acc.item())
                    * float(cfg.loss.get("constraint_hand_rot_acc_weight", 0.0)),
                    "lr": float(active_lr),
                    "time": time.time(),
                }
                for name in LOSS_NAMES:
                    metric[f"teacher_{name}"] = float(reduced_teacher_terms[name].item())
                    metric[f"teacher_weighted_{name}"] = float(reduced_teacher_weighted_terms[name].item())
                    metric[f"gt_{name}"] = float(reduced_gt_terms[name].item())
                    metric[f"gt_weighted_{name}"] = float(reduced_gt_weighted_terms[name].item())
                log.info(
                    "step=%d kf=%d loss=%.6f teacher=%.6f gt=%.6f constraint=%.6f "
                    "tw=%.3f gw=%.3f tcw=%.3f gcw=%.3f ceff=%.3f "
                    "self_acc=%.6f saw=%.3f root_acc=%.6f raw=%.3f heading_acc=%.6f haw=%.3f "
                    "constraint_root_acc=%.6f craw=%.3f hand_pos_acc=%.6f hpaw=%.3f hand_rot_acc=%.6f hraw=%.3f "
                    "rollout_bs=%d k=%d eff_bs=%d mean_sched=%.2f lr=%.3e",
                    metric["step"],
                    metric["num_keyframes"],
                    metric["loss_total"],
                    metric["loss_teacher_total"],
                    metric["loss_gt_total"],
                    metric["loss_constraint_total"],
                    metric["teacher_weight"],
                    metric["gt_weight"],
                    metric["teacher_constraint_loss_weight"],
                    metric["gt_constraint_loss_weight"],
                    metric["constraint_loss_weight"],
                    metric["self_acc_loss"],
                    metric["self_acc_weight"],
                    metric["root_acc_loss"],
                    metric["root_acc_weight"],
                    metric["heading_acc_loss"],
                    metric["heading_acc_weight"],
                    metric["constraint_root_acc_loss"],
                    metric["constraint_root_acc_weight"],
                    metric["constraint_hand_pos_acc_loss"],
                    metric["constraint_hand_pos_acc_weight"],
                    metric["constraint_hand_rot_acc_loss"],
                    metric["constraint_hand_rot_acc_weight"],
                    metric["rollout_batch_size"],
                    metric["samples_per_rollout"],
                    metric["effective_supervision_batch"],
                    metric["mean_schedule_idx"],
                    metric["lr"],
                )
                with log_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(metric, ensure_ascii=False) + "\n")
                if wandb_run is not None:
                    wandb_run.log(metric, step=global_step)

        del (
            states,
            x_t,
            schedule_idx,
            timesteps,
            pad_mask_rep,
            text_feat_rep,
            text_pad_mask_rep,
            gt_x0_rep,
            first_heading_rep,
            observed_motion_rep,
            motion_mask_rep,
            teacher_x0,
            student_x0,
            teacher_loss_dict,
            gt_loss_dict,
            constraint_loss_dict,
            loss,
            self_acc_loss,
            root_acc_loss,
            heading_acc_loss,
            constraint_root_acc_loss,
            constraint_heading_acc_loss,
            constraint_hand_pos_acc_loss,
            constraint_hand_rot_acc_loss,
        )
        if device.type == "cuda" and bool(cfg.training.get("empty_cache_every_step", False)):
            torch.cuda.empty_cache()

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
