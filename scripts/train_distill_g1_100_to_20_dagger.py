#!/usr/bin/env python3
"""DAgger-style diffusion distillation for Kimodo G1.

This fine-tunes a previously distilled diffusion student by:
1. rolling out the current student for 20 DDIM steps,
2. sampling one or more visited denoising states per rollout,
3. asking the 100-step teacher denoiser to relabel those off-policy states,
4. training the student with teacher + GT auxiliary losses.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, Iterator, Optional

import torch
import torch.distributed as dist
from omegaconf import DictConfig, OmegaConf
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP

from kimodo.distillation.loss import DistillationKimodoLoss
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
    get_distill_loss_weights,
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
from kimodo.training.ema import EMA
from kimodo.training.optimizers import build_optimizer

log = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DAgger fine-tune 100->20 Kimodo G1 diffusion distillation.")
    parser.add_argument(
        "--config",
        type=str,
        default="kimodo/distillation/configs/distill_g1_100_to_20_dagger_10k_bs16x4.yaml",
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

    student = instantiate_from_dict(student_cfg).to(device)
    maybe_load_initial_student(cfg, student, teacher, rank)

    diffusion = Diffusion(num_base_steps=int(cfg.model.num_base_steps)).to(device)
    sampler = DDIMSampler(diffusion)
    motion_rep = student.motion_rep

    dataset_motion_rep = build_dataset_motion_rep(cfg)
    text_encoder = load_text_encoder(cfg, device)
    dataset = build_dataset(cfg, motion_rep=dataset_motion_rep, rank=rank)
    scheduler = create_keyframe_scheduler(cfg)
    loader, sampler_dist = create_dataloader(dataset, cfg, is_distributed=is_distributed)

    optimizer = build_optimizer(cfg, student)
    autocast_ctx, use_scaler = make_autocast(cfg.training.mixed_precision, device)
    scaler = torch.cuda.amp.GradScaler(enabled=use_scaler)

    loss_fn = DistillationKimodoLoss(
        motion_rep=motion_rep,
        teacher_gammas=cfg.loss.teacher_gammas,
        gt_gammas=cfg.loss.gt_gammas,
        teacher_weight=float(cfg.loss.teacher_weight),
        gt_weight=float(cfg.loss.gt_weight),
        constraint_loss_weight=float(cfg.loss.get("constraint_loss_weight", 0.0)),
        hand_constraint_weight=float(cfg.loss.get("hand_constraint_weight", 3.0)),
        foot_constraint_weight=float(cfg.loss.get("foot_constraint_weight", 1.5)),
        root_constraint_weight=float(cfg.loss.get("root_constraint_weight", 1.0)),
        input_is_normalized=bool(cfg.data.dataset.to_normalize),
    )
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
        log.info(
            "Starting DAgger distillation on %s | world_size=%d | total_steps=%d | "
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

        optimizer.zero_grad(set_to_none=True)
        teacher_weight, gt_weight = get_distill_loss_weights(
            cfg,
            global_step=global_step,
            total_steps=total_steps,
        )
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
            loss_dict = loss_fn(
                pred_x0=student_x0,
                teacher_x0=teacher_x0,
                gt_x0=gt_x0_rep,
                pad_mask=pad_mask_rep,
                observed_motion=observed_motion_rep,
                motion_mask=motion_mask_rep,
                teacher_weight=teacher_weight,
                gt_weight=gt_weight,
            )
            loss = loss_dict["total"]

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
            reduced_teacher_total = reduce_scalar(loss_dict["loss_teacher_total"].detach(), world_size, is_distributed)
            reduced_gt_total = reduce_scalar(loss_dict["loss_gt_total"].detach(), world_size, is_distributed)
            reduced_constraint_total = reduce_scalar(
                loss_dict["loss_constraint_total"].detach(),
                world_size,
                is_distributed,
            )
            reduced_constraint_pos = reduce_scalar(
                loss_dict["loss_constraint_position"].detach(),
                world_size,
                is_distributed,
            )
            reduced_constraint_rot = reduce_scalar(
                loss_dict["loss_constraint_rotation"].detach(),
                world_size,
                is_distributed,
            )
            reduced_constraint_root = reduce_scalar(
                loss_dict["loss_constraint_root"].detach(),
                world_size,
                is_distributed,
            )
            mean_schedule_idx = reduce_scalar(schedule_idx.float().mean(), world_size, is_distributed)

            if is_main_process(rank):
                metric = {
                    "step": global_step,
                    "epoch": epoch,
                    "num_keyframes": int(dataset.get_phase2_num_keyframes()),
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
                    "teacher_weight": float(loss_dict["teacher_weight"].detach().item()),
                    "gt_weight": float(loss_dict["gt_weight"].detach().item()),
                    "constraint_loss_weight": float(loss_dict["constraint_loss_weight"].detach().item()),
                    "constraint_observed_count": float(loss_dict["constraint_observed_count"].detach().item()),
                    "lr": float(optimizer.param_groups[0]["lr"]),
                    "time": time.time(),
                }
                log.info(
                    "step=%d kf=%d loss=%.6f teacher=%.6f gt=%.6f constraint=%.6f "
                    "tw=%.3f gw=%.3f cw=%.3f "
                    "rollout_bs=%d k=%d eff_bs=%d mean_sched=%.2f lr=%.3e",
                    metric["step"],
                    metric["num_keyframes"],
                    metric["loss_total"],
                    metric["loss_teacher_total"],
                    metric["loss_gt_total"],
                    metric["loss_constraint_total"],
                    metric["teacher_weight"],
                    metric["gt_weight"],
                    metric["constraint_loss_weight"],
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
