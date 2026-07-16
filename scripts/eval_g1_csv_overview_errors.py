#!/usr/bin/env python3
"""Evaluate G1 CSV training clips with overview prompts and EE constraint errors."""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
import torch
from omegaconf import OmegaConf
from scipy.spatial.transform import Rotation as R

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = Path(__file__).resolve().parent
for _p in (REPO_ROOT, SCRIPTS_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from kimodo import load_model
from kimodo.constraints import load_constraints_lst
from kimodo.exports.mujoco import MujocoQposConverter
from kimodo.model.kimodo_model import Kimodo
from kimodo.model.loading import instantiate_from_dict
from kimodo.training.g1_csv import load_g1_csv_motion
from generate_g1_with_first_heading import resolve_num_frames
from npz_to_ee_pose_constraints import mujoco_xyz_to_kimodo, rot_mujoco_to_kimodo, select_source_indices, yaw_from_rot_kimodo


FIELD_TO_BODY = {
    "left_hand_pose": "left_wrist_yaw_link",
    "right_hand_pose": "right_wrist_yaw_link",
    "left_foot_pose": "left_ankle_roll_link",
    "right_foot_pose": "right_ankle_roll_link",
}
HAND_FIELDS = {"left_hand_pose", "right_hand_pose"}
DEFAULT_G1_MODEL = "Kimodo-G1-RP-v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate G1 training CSV clips by generating from overview prompts and reporting per-task/overall EE errors."
        )
    )
    parser.add_argument("--csv_root", default="./dataset/g1/csv", help="Root folder of G1 training CSV files.")
    parser.add_argument("--timelines_jsonl", default="./dataset/timelines.jsonl", help="Timeline JSONL path.")
    parser.add_argument("--csv_pattern", default="*.csv", help="CSV pattern under csv_root.")
    parser.add_argument("--sample_ratio", type=float, default=1.0, help="Random sample ratio in (0, 1].")
    parser.add_argument("--max_tasks", type=int, default=None, help="Optional cap on sampled task count.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling.")
    parser.add_argument("--model", default=DEFAULT_G1_MODEL, help="Default official model name.")
    parser.add_argument(
        "--distill_config",
        default=None,
        help="Optional distillation YAML. If set with --distill_ckpt, evaluate distilled student.",
    )
    parser.add_argument(
        "--distill_ckpt",
        default=None,
        help="Optional distillation checkpoint (.pt). Must be paired with --distill_config.",
    )
    parser.add_argument("--source_fps", type=float, default=120.0, help="Source CSV FPS.")
    parser.add_argument("--target_fps", type=float, default=30.0, help="Constraint FPS.")
    parser.add_argument(
        "--max_eval_frames",
        type=int,
        default=300,
        help="Max evaluated frames at target_fps (default: 300, i.e., first 10s at 30fps).",
    )
    parser.add_argument("--keyframe_step", type=int, default=30, help="Keyframe stride on source frames.")
    parser.add_argument("--diffusion_steps", type=int, default=100, help="Generation diffusion steps.")
    parser.add_argument(
        "--num_runs_per_task",
        type=int,
        default=5,
        help="How many independent generation/eval runs per task, then average per-task metrics (default: 5).",
    )
    parser.add_argument("--xml", default="kimodo/assets/skeletons/g1skel34/xml/g1.xml", help="MuJoCo XML path.")
    parser.add_argument(
        "--output_json",
        default="scripts/eval_g1_csv_overview_errors_summary.json",
        help="Output summary JSON path.",
    )
    parser.add_argument(
        "--constraints_out_dir",
        default=None,
        help="Optional directory to persist generated 5-point constraint JSON files.",
    )
    parser.add_argument(
        "--progress_jsonl",
        default=None,
        help="Optional JSONL path to append one result row per completed task.",
    )
    parser.add_argument(
        "--videos_out_dir",
        default=None,
        help="Optional directory to save first-run qpos CSV and constraint-overlay MP4 for each task.",
    )
    parser.add_argument("--video_fps", type=int, default=30, help="Output video FPS when --videos_out_dir is set.")
    parser.add_argument("--video_width", type=int, default=1280, help="Output video width when --videos_out_dir is set.")
    parser.add_argument("--video_height", type=int, default=720, help="Output video height when --videos_out_dir is set.")
    parser.add_argument("--fail_fast", action="store_true", help="Stop immediately on one task failure.")
    parser.add_argument("--verbose", action="store_true", help="Print per-task details.")
    return parser.parse_args()


def _load_qpos_csv(path: Path) -> np.ndarray:
    """Load BONES/SEED-style training CSV and convert to MuJoCo qpos [T, 36].

    Expected CSV columns:
    - root_translate{X,Y,Z} (centimeters)
    - root_rotate{X,Y,Z} (degrees, Euler xyz)
    - 29 *_joint_dof columns (degrees)
    """
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if not rows:
        raise ValueError(f"Empty CSV: {path}")

    root_pos_cols = ["root_translateX", "root_translateY", "root_translateZ"]
    root_rot_cols = ["root_rotateX", "root_rotateY", "root_rotateZ"]
    dof_cols = [c for c in rows[0].keys() if c and c.endswith("_joint_dof")]
    if len(dof_cols) != 29:
        raise ValueError(f"Expected 29 *_joint_dof columns, got {len(dof_cols)} in {path}")

    qpos = np.zeros((len(rows), 36), dtype=np.float64)
    for i, row in enumerate(rows):
        root_xyz_cm = np.asarray([float(row[c]) for c in root_pos_cols], dtype=np.float64)
        root_rpy_deg = np.asarray([float(row[c]) for c in root_rot_cols], dtype=np.float64)
        dof_deg = np.asarray([float(row[c]) for c in dof_cols], dtype=np.float64)

        # Training CSV root translation uses centimeters; MuJoCo qpos uses meters.
        qpos[i, :3] = root_xyz_cm / 100.0
        quat_wxyz = R.from_euler("xyz", root_rpy_deg, degrees=True).as_quat(scalar_first=True)
        qpos[i, 3:7] = quat_wxyz
        qpos[i, 7:] = np.deg2rad(dof_deg)
    return qpos


def _build_ee_constraints_from_motion(
    local_joint_rots: torch.Tensor,
    root_positions: torch.Tensor,
    *,
    skeleton,
    source_fps: float,
    target_fps: float,
    keyframe_step: int,
) -> dict[str, Any]:
    """Build ee-pose constraints in Kimodo coordinate system from training-aligned motion tensors."""
    skel_device = skeleton.joint_parents.device
    local_joint_rots = local_joint_rots.to(device=skel_device)
    root_positions = root_positions.to(device=skel_device)
    num_frames = int(local_joint_rots.shape[0])
    src_idx = select_source_indices(num_frames=num_frames, step=max(1, int(keyframe_step)))
    dst_idx = np.round(src_idx * float(target_fps) / float(source_fps)).astype(np.int64)

    uniq_mask = np.ones(len(dst_idx), dtype=bool)
    uniq_mask[1:] = dst_idx[1:] != dst_idx[:-1]
    src_idx = src_idx[uniq_mask]
    dst_idx = dst_idx[uniq_mask]

    global_rots, global_positions, _ = skeleton.fk(local_joint_rots, root_positions)
    global_rots_np = global_rots.detach().cpu().numpy()
    global_positions_np = global_positions.detach().cpu().numpy()
    root_positions_np = root_positions.detach().cpu().numpy()

    ee_joint_map = {
        "left_hand_pose": "left_wrist_yaw_skel",
        "right_hand_pose": "right_wrist_yaw_skel",
        "left_foot_pose": "left_ankle_roll_skel",
        "right_foot_pose": "right_ankle_roll_skel",
    }
    ee_joint_idx = {field: skeleton.bone_order_names.index(joint_name) for field, joint_name in ee_joint_map.items()}

    item: dict[str, Any] = {"type": "ee-pose", "frame_indices": dst_idx.tolist(), "root_xyzyaw": []}
    for field in ee_joint_map:
        item[field] = []

    for t in src_idx:
        root_rot_k = global_rots_np[t, skeleton.root_idx]
        root_xyz_k = root_positions_np[t]
        root_yaw_k = yaw_from_rot_kimodo(root_rot_k)
        item["root_xyzyaw"].append([float(root_xyz_k[0]), float(root_xyz_k[1]), float(root_xyz_k[2]), float(root_yaw_k)])

        for field, joint_idx in ee_joint_idx.items():
            ee_xyz_k = global_positions_np[t, joint_idx]
            ee_rot_k = global_rots_np[t, joint_idx]
            ee_rpy_k = R.from_matrix(ee_rot_k).as_euler("xyz")
            item[field].append(
                [
                    float(ee_xyz_k[0]),
                    float(ee_xyz_k[1]),
                    float(ee_xyz_k[2]),
                    float(ee_rpy_k[0]),
                    float(ee_rpy_k[1]),
                    float(ee_rpy_k[2]),
                ]
            )
    return item


def _first_heading_from_motion(local_joint_rots: torch.Tensor, root_positions: torch.Tensor, skeleton) -> float:
    skel_device = skeleton.joint_parents.device
    local_joint_rots = local_joint_rots.to(device=skel_device)
    root_positions = root_positions.to(device=skel_device)
    global_rots, _, _ = skeleton.fk(local_joint_rots, root_positions)
    root_rot_k = global_rots[0, skeleton.root_idx].detach().cpu().numpy()
    return float(yaw_from_rot_kimodo(root_rot_k))


def _load_timeline_map(path: Path) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            filename = str(obj.get("filename", "")).strip()
            if filename:
                records[filename] = obj
    return records


def _resolve_prompt(clip_stem: str, timeline_map: dict[str, dict[str, Any]]) -> str:
    rec = timeline_map.get(clip_stem)
    if rec is None:
        return clip_stem
    overview = str(rec.get("overview_description", "")).strip()
    if overview:
        return overview
    events = rec.get("events", [])
    if events:
        first = events[0]
        desc = str(first.get("description", "")).strip()
        if desc:
            return desc
    return clip_stem


def _first_heading_from_qpos(qpos: np.ndarray) -> float:
    if qpos.shape[1] < 7:
        raise ValueError(f"Expected qpos with >=7 dims, got shape={qpos.shape}.")
    root_quat_wxyz = qpos[0, 3:7]
    root_rot_m = R.from_quat(root_quat_wxyz, scalar_first=True).as_matrix()
    root_rot_k = rot_mujoco_to_kimodo(root_rot_m)
    return yaw_from_rot_kimodo(root_rot_k)


def _build_ee_constraints_from_qpos(
    qpos: np.ndarray,
    *,
    mj_model: mujoco.MjModel,
    source_fps: float,
    target_fps: float,
    keyframe_step: int,
) -> dict[str, Any]:
    src_idx = select_source_indices(num_frames=int(qpos.shape[0]), step=max(1, int(keyframe_step)))
    dst_idx = np.round(src_idx * float(target_fps) / float(source_fps)).astype(np.int64)

    uniq_mask = np.ones(len(dst_idx), dtype=bool)
    uniq_mask[1:] = dst_idx[1:] != dst_idx[:-1]
    src_idx = src_idx[uniq_mask]
    dst_idx = dst_idx[uniq_mask]

    mj_data = mujoco.MjData(mj_model)
    item: dict[str, Any] = {"type": "ee-pose", "frame_indices": dst_idx.tolist(), "root_xyzyaw": []}
    for field in FIELD_TO_BODY:
        item[field] = []

    body_ids = {
        field: mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, body_name) for field, body_name in FIELD_TO_BODY.items()
    }
    for field, bid in body_ids.items():
        if bid < 0:
            raise ValueError(f"Body not found in XML for field={field}: {FIELD_TO_BODY[field]}")

    for t in src_idx:
        mj_data.qpos[:] = qpos[t]
        mujoco.mj_forward(mj_model, mj_data)

        root_xyz_m = np.asarray(qpos[t, :3], dtype=np.float64)
        root_rot_m = R.from_quat(np.asarray(qpos[t, 3:7], dtype=np.float64), scalar_first=True).as_matrix()
        root_rot_k = rot_mujoco_to_kimodo(root_rot_m)
        root_xyz_k = mujoco_xyz_to_kimodo(root_xyz_m)
        root_yaw_k = yaw_from_rot_kimodo(root_rot_k)
        item["root_xyzyaw"].append([float(root_xyz_k[0]), float(root_xyz_k[1]), float(root_xyz_k[2]), float(root_yaw_k)])

        for field, bid in body_ids.items():
            ee_xyz_m = np.asarray(mj_data.xpos[bid], dtype=np.float64)
            ee_rot_m = np.asarray(mj_data.xmat[bid], dtype=np.float64).reshape(3, 3)
            ee_xyz_k = mujoco_xyz_to_kimodo(ee_xyz_m)
            ee_rot_k = rot_mujoco_to_kimodo(ee_rot_m)
            ee_rpy_k = R.from_matrix(ee_rot_k).as_euler("xyz")
            item[field].append(
                [
                    float(ee_xyz_k[0]),
                    float(ee_xyz_k[1]),
                    float(ee_xyz_k[2]),
                    float(ee_rpy_k[0]),
                    float(ee_rpy_k[1]),
                    float(ee_rpy_k[2]),
                ]
            )
    return item


def _kimodo_to_mujoco_xyz(xyz_k: np.ndarray) -> np.ndarray:
    return np.asarray([xyz_k[2], xyz_k[0], xyz_k[1]], dtype=np.float64)


def _angle_abs_diff(a: float, b: float) -> float:
    return float(abs((float(a) - float(b) + np.pi) % (2.0 * np.pi) - np.pi))


def _rotation_angle_error(rot_a: np.ndarray, rot_b: np.ndarray) -> float:
    rel = np.asarray(rot_a, dtype=np.float64).T @ np.asarray(rot_b, dtype=np.float64)
    return float(R.from_matrix(rel).magnitude())


def _evaluate_qpos_errors(qpos: np.ndarray, ee_constraint_item: dict[str, Any], mj_model: mujoco.MjModel) -> dict[str, Any]:
    frame_indices = ee_constraint_item.get("frame_indices", [])
    mj_data = mujoco.MjData(mj_model)
    five_point_pos_errors: list[float] = []
    ee_pos_errors: list[float] = []
    hand_errors: list[float] = []
    foot_errors: list[float] = []
    root_errors: list[float] = []
    ee_rot_errors: list[float] = []
    hand_rot_errors: list[float] = []
    foot_rot_errors: list[float] = []
    root_yaw_errors: list[float] = []
    per_point_pos: dict[str, list[float]] = {
        "root": [],
        "left_hand": [],
        "right_hand": [],
        "left_foot": [],
        "right_foot": [],
    }
    per_point_rot: dict[str, list[float]] = {
        "root_yaw": [],
        "left_hand": [],
        "right_hand": [],
        "left_foot": [],
        "right_foot": [],
    }
    field_to_point = {
        "left_hand_pose": "left_hand",
        "right_hand_pose": "right_hand",
        "left_foot_pose": "left_foot",
        "right_foot_pose": "right_foot",
    }

    for local_i, frame_idx in enumerate(frame_indices):
        if frame_idx < 0 or frame_idx >= len(qpos):
            continue
        mj_data.qpos[:] = qpos[frame_idx]
        mujoco.mj_forward(mj_model, mj_data)

        root_xyzyaw = ee_constraint_item.get("root_xyzyaw")
        if root_xyzyaw and local_i < len(root_xyzyaw):
            expected_root_k = np.asarray(root_xyzyaw[local_i][:3], dtype=np.float64)
            expected_root_m = _kimodo_to_mujoco_xyz(expected_root_k)
            actual_root_m = np.asarray(qpos[frame_idx, :3], dtype=np.float64)
            root_err = float(np.linalg.norm(actual_root_m - expected_root_m))
            root_errors.append(root_err)
            five_point_pos_errors.append(root_err)
            per_point_pos["root"].append(root_err)

            expected_yaw = float(root_xyzyaw[local_i][3])
            actual_root_rot_m = R.from_quat(np.asarray(qpos[frame_idx, 3:7], dtype=np.float64), scalar_first=True).as_matrix()
            actual_root_rot_k = rot_mujoco_to_kimodo(actual_root_rot_m)
            actual_yaw = yaw_from_rot_kimodo(actual_root_rot_k)
            yaw_err = _angle_abs_diff(actual_yaw, expected_yaw)
            root_yaw_errors.append(yaw_err)
            per_point_rot["root_yaw"].append(yaw_err)

        for field, body_name in FIELD_TO_BODY.items():
            poses = ee_constraint_item.get(field)
            if not poses:
                continue
            expected_k = np.asarray(poses[local_i][:3], dtype=np.float64)
            expected = _kimodo_to_mujoco_xyz(expected_k)
            body_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, body_name)
            actual = np.asarray(mj_data.xpos[body_id], dtype=np.float64)
            err = float(np.linalg.norm(actual - expected))
            point_name = field_to_point[field]
            ee_pos_errors.append(err)
            five_point_pos_errors.append(err)
            per_point_pos[point_name].append(err)
            if field in HAND_FIELDS:
                hand_errors.append(err)
            else:
                foot_errors.append(err)

            if len(poses[local_i]) >= 6:
                expected_rot_k = R.from_euler("xyz", np.asarray(poses[local_i][3:6], dtype=np.float64)).as_matrix()
                actual_rot_m = np.asarray(mj_data.xmat[body_id], dtype=np.float64).reshape(3, 3)
                actual_rot_k = rot_mujoco_to_kimodo(actual_rot_m)
                rot_err = _rotation_angle_error(actual_rot_k, expected_rot_k)
                ee_rot_errors.append(rot_err)
                per_point_rot[point_name].append(rot_err)
                if field in HAND_FIELDS:
                    hand_rot_errors.append(rot_err)
                else:
                    foot_rot_errors.append(rot_err)

    return {
        "five_point_pos_errors": five_point_pos_errors,
        "ee_pos_errors": ee_pos_errors,
        "hand_pos_errors": hand_errors,
        "foot_pos_errors": foot_errors,
        "root_pos_errors": root_errors,
        "ee_rot_errors_rad": ee_rot_errors,
        "hand_rot_errors_rad": hand_rot_errors,
        "foot_rot_errors_rad": foot_rot_errors,
        "root_yaw_errors_rad": root_yaw_errors,
        "per_point_pos_errors": per_point_pos,
        "per_point_rot_errors_rad": per_point_rot,
    }


def _safe_stats(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "mean_error_m": None, "max_error_m": None, "min_error_m": None}
    arr = np.asarray(values, dtype=np.float64)
    return {
        "count": int(arr.size),
        "mean_error_m": float(arr.mean()),
        "max_error_m": float(arr.max()),
        "min_error_m": float(arr.min()),
    }


def _safe_mean(values: list[float]) -> float | None:
    if not values:
        return None
    arr = np.asarray(values, dtype=np.float64)
    return float(arr.mean())


def _safe_percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def _sync_if_cuda(device: str) -> None:
    if str(device).startswith("cuda"):
        torch.cuda.synchronize()


def _save_first_run_video(
    *,
    qpos: np.ndarray,
    constraints_path: Path,
    videos_out_dir: Path,
    rel_parent: Path,
    clip_stem: str,
    xml_path: str,
    fps: int,
    width: int,
    height: int,
) -> dict[str, str]:
    from render_mujoco_constraints_overlay import render as render_constraint_overlay

    task_video_dir = videos_out_dir / rel_parent
    task_video_dir.mkdir(parents=True, exist_ok=True)
    qpos_path = task_video_dir / f"{clip_stem}.first_run.qpos.csv"
    video_path = task_video_dir / f"{clip_stem}.first_run.constraints_overlay.mp4"
    preview_path = task_video_dir / f"{clip_stem}.first_run.constraints_overlay.png"
    np.savetxt(qpos_path, np.asarray(qpos, dtype=np.float64), delimiter=",")
    render_constraint_overlay(
        csv_path=qpos_path,
        constraints_path=constraints_path,
        output=video_path,
        preview=preview_path,
        xml_path=Path(xml_path),
        fps=int(fps),
        width=int(width),
        height=int(height),
        trail=True,
    )
    return {
        "qpos_path": str(qpos_path),
        "video_path": str(video_path),
        "preview_path": str(preview_path),
    }


def _load_eval_model(args: argparse.Namespace, device: str):
    if bool(args.distill_config) != bool(args.distill_ckpt):
        raise ValueError("--distill_config and --distill_ckpt must be provided together.")
    if args.distill_config and args.distill_ckpt:
        return _build_model_from_distill_flexible(
            distill_config_path=args.distill_config,
            distill_ckpt_path=args.distill_ckpt,
            device=device,
        ), "distill-student"
    return load_model(args.model, device=device, default_family="Kimodo", return_resolved_name=True)


def _resolve_student_state_dict_from_ckpt(ckpt_path: str) -> tuple[dict[str, torch.Tensor], str]:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if not isinstance(ckpt, dict):
        raise ValueError(f"Unsupported distill checkpoint format: {ckpt_path}")

    if "student" in ckpt and isinstance(ckpt["student"], dict):
        return ckpt["student"], "student"

    if "ema" in ckpt and isinstance(ckpt["ema"], dict) and "shadow" in ckpt["ema"]:
        shadow = ckpt["ema"]["shadow"]
        if not isinstance(shadow, dict):
            raise ValueError(f"Invalid ema.shadow format in checkpoint: {ckpt_path}")
        return shadow, "ema.shadow"

    # Fallback: sometimes checkpoint itself is a plain state_dict.
    if all(isinstance(k, str) for k in ckpt.keys()):
        if any(isinstance(v, torch.Tensor) for v in ckpt.values()):
            return ckpt, "state_dict"

    raise ValueError(
        "Distill checkpoint must contain one of: {'student': state_dict} or "
        "{'ema': {'shadow': state_dict}} or be a plain state_dict. "
        f"Got keys={list(ckpt.keys())[:10]} from {ckpt_path}"
    )


def _build_model_from_distill_flexible(*, distill_config_path: str, distill_ckpt_path: str, device: str) -> Kimodo:
    cfg = OmegaConf.load(distill_config_path)
    student_cfg = OmegaConf.to_container(cfg.model.student_denoiser, resolve=True)
    text_encoder_cfg = OmegaConf.to_container(cfg.text_encoder, resolve=True)

    denoiser = instantiate_from_dict(student_cfg).to(device)
    text_encoder = instantiate_from_dict(text_encoder_cfg)
    text_encoder.to(device)
    text_encoder.eval()

    loaded_state, source = _resolve_student_state_dict_from_ckpt(distill_ckpt_path)
    missing, unexpected = denoiser.load_state_dict(loaded_state, strict=False)
    if unexpected:
        raise ValueError(
            f"Unexpected keys when loading {source} from {distill_ckpt_path}: first 10={unexpected[:10]}"
        )
    if missing:
        # Keep strict enough for reliability; if only buffers are missing, this still flags.
        raise ValueError(
            f"Missing keys when loading {source} from {distill_ckpt_path}: first 10={missing[:10]}"
        )
    denoiser.eval()

    model = Kimodo(
        denoiser=denoiser,
        text_encoder=text_encoder,
        num_base_steps=int(cfg.model.num_base_steps),
        device=device,
        cfg_type="separated",
    )
    return model


def _sample_tasks(all_paths: list[Path], *, ratio: float, max_tasks: int | None, seed: int) -> list[Path]:
    if not (0.0 < ratio <= 1.0):
        raise ValueError(f"sample_ratio must be in (0,1], got {ratio}")
    rng = random.Random(seed)
    paths = list(all_paths)
    rng.shuffle(paths)
    keep = max(1, int(round(len(paths) * ratio)))
    sampled = paths[:keep]
    if max_tasks is not None:
        sampled = sampled[: max(1, int(max_tasks))]
    return sorted(sampled)


def main() -> None:
    args = parse_args()
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

    csv_root = Path(args.csv_root)
    if not csv_root.exists():
        raise FileNotFoundError(f"csv_root not found: {csv_root}")
    timeline_path = Path(args.timelines_jsonl)
    if not timeline_path.exists():
        raise FileNotFoundError(f"timelines_jsonl not found: {timeline_path}")

    all_csv_paths = sorted(csv_root.rglob(args.csv_pattern))
    if not all_csv_paths:
        raise FileNotFoundError(f"No CSV files found under {csv_root} with pattern {args.csv_pattern!r}")
    csv_paths = _sample_tasks(all_csv_paths, ratio=float(args.sample_ratio), max_tasks=args.max_tasks, seed=int(args.seed))

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model, resolved_model = _load_eval_model(args, device=device)
    converter = MujocoQposConverter(model.skeleton)
    mj_model = mujoco.MjModel.from_xml_path(args.xml)
    timeline_map = _load_timeline_map(timeline_path)
    constraints_out_dir = Path(args.constraints_out_dir) if args.constraints_out_dir else None
    if constraints_out_dir is not None:
        constraints_out_dir.mkdir(parents=True, exist_ok=True)
    progress_jsonl = Path(args.progress_jsonl) if args.progress_jsonl else None
    if progress_jsonl is not None:
        progress_jsonl.parent.mkdir(parents=True, exist_ok=True)
    videos_out_dir = Path(args.videos_out_dir) if args.videos_out_dir else None
    if videos_out_dir is not None:
        videos_out_dir.mkdir(parents=True, exist_ok=True)

    if args.verbose:
        print(f"Loaded model: {resolved_model} on {device}")
        print(f"Total CSV files={len(all_csv_paths)} | sampled={len(csv_paths)}")

    per_task = []
    failures: list[dict[str, str]] = []
    all_five_point_errors: list[float] = []
    all_ee_errors: list[float] = []
    all_hand_errors: list[float] = []
    all_foot_errors: list[float] = []
    all_root_errors: list[float] = []
    all_ee_rot_errors: list[float] = []
    all_hand_rot_errors: list[float] = []
    all_foot_rot_errors: list[float] = []
    all_root_yaw_errors: list[float] = []
    all_generation_latencies: list[float] = []

    with tempfile.TemporaryDirectory(prefix="kimodo_eval_g1_csv_") as tmpdir:
        tmpdir_path = Path(tmpdir)
        for idx, csv_path in enumerate(csv_paths, start=1):
            task_name = csv_path.relative_to(csv_root).as_posix()
            clip_stem = csv_path.stem
            print(f"[{idx}/{len(csv_paths)}] {task_name} ...", flush=True)
            try:
                prompt = _resolve_prompt(clip_stem, timeline_map)
                motion = load_g1_csv_motion(
                    csv_path,
                    source_coord_system="mujoco",
                    root_euler_order="xyz",
                    root_angle_unit="degrees",
                    joint_angle_unit="degrees",
                    root_position_unit="centimeters",
                    root_position_scale=1.0,
                    device="cpu",
                    dtype=torch.float32,
                )
                local_joint_rots = motion["local_joint_rots"]
                root_positions = motion["root_positions"]
                if int(args.max_eval_frames) > 0:
                    source_cap = int(round(int(args.max_eval_frames) * float(args.source_fps) / float(args.target_fps)))
                    source_cap = max(1, source_cap)
                    if local_joint_rots.shape[0] > source_cap:
                        local_joint_rots = local_joint_rots[:source_cap]
                        root_positions = root_positions[:source_cap]

                ee_constraint = _build_ee_constraints_from_motion(
                    local_joint_rots,
                    root_positions,
                    skeleton=model.skeleton,
                    source_fps=float(args.source_fps),
                    target_fps=float(args.target_fps),
                    keyframe_step=int(args.keyframe_step),
                )

                constraints_path = tmpdir_path / f"{idx:06d}_{clip_stem}.constraints.json"
                constraints_path.write_text(json.dumps([ee_constraint], indent=2), encoding="utf-8")
                persisted_constraints_path = None
                if constraints_out_dir is not None:
                    rel_parent = csv_path.relative_to(csv_root).parent
                    task_constraints_dir = constraints_out_dir / rel_parent
                    task_constraints_dir.mkdir(parents=True, exist_ok=True)
                    persisted_constraints_path = task_constraints_dir / f"{clip_stem}.five_pose_constraints.json"
                    persisted_constraints_path.write_text(json.dumps([ee_constraint], indent=2), encoding="utf-8")

                duration = float(local_joint_rots.shape[0]) / float(args.source_fps)
                num_frames, _ = resolve_num_frames(duration, float(model.fps), str(constraints_path))
                constraint_lst = load_constraints_lst(str(constraints_path), model.skeleton)
                first_heading = _first_heading_from_motion(local_joint_rots, root_positions, model.skeleton)

                run_means: list[float] = []
                run_maxes: list[float] = []
                run_mins: list[float] = []
                run_num_points: list[int] = []
                run_latencies: list[float] = []
                clip_five_point_errors_all_runs: list[float] = []
                clip_ee_errors_all_runs: list[float] = []
                clip_hand_errors_all_runs: list[float] = []
                clip_foot_errors_all_runs: list[float] = []
                clip_root_errors_all_runs: list[float] = []
                clip_ee_rot_errors_all_runs: list[float] = []
                clip_hand_rot_errors_all_runs: list[float] = []
                clip_foot_rot_errors_all_runs: list[float] = []
                clip_root_yaw_errors_all_runs: list[float] = []
                video_artifacts: dict[str, str] = {}

                for run_idx in range(int(args.num_runs_per_task)):
                    _sync_if_cuda(device)
                    t0 = time.perf_counter()
                    output = model(
                        prompt,
                        num_frames,
                        constraint_lst=constraint_lst,
                        num_denoising_steps=int(args.diffusion_steps),
                        num_samples=1,
                        multi_prompt=False,
                        post_processing=False,
                        return_numpy=True,
                        first_heading_angle=torch.tensor([first_heading], dtype=torch.float32, device=device),
                    )
                    _sync_if_cuda(device)
                    run_latencies.append(time.perf_counter() - t0)

                    pred_qpos = converter.dict_to_qpos(output, device)
                    pred_qpos = np.asarray(pred_qpos, dtype=np.float64)
                    if pred_qpos.ndim == 3:
                        pred_qpos = pred_qpos[0]

                    if videos_out_dir is not None and run_idx == 0:
                        rel_parent = csv_path.relative_to(csv_root).parent
                        video_artifacts = _save_first_run_video(
                            qpos=pred_qpos,
                            constraints_path=constraints_path,
                            videos_out_dir=videos_out_dir,
                            rel_parent=rel_parent,
                            clip_stem=clip_stem,
                            xml_path=args.xml,
                            fps=int(args.video_fps),
                            width=int(args.video_width),
                            height=int(args.video_height),
                        )

                    err = _evaluate_qpos_errors(pred_qpos, ee_constraint, mj_model)
                    run_five = err["five_point_pos_errors"]
                    run_stats = _safe_stats(run_five)
                    if run_stats["mean_error_m"] is not None:
                        run_means.append(float(run_stats["mean_error_m"]))
                    if run_stats["max_error_m"] is not None:
                        run_maxes.append(float(run_stats["max_error_m"]))
                    if run_stats["min_error_m"] is not None:
                        run_mins.append(float(run_stats["min_error_m"]))
                    run_num_points.append(int(run_stats["count"]))
                    clip_five_point_errors_all_runs.extend(run_five)
                    clip_ee_errors_all_runs.extend(err["ee_pos_errors"])
                    clip_hand_errors_all_runs.extend(err["hand_pos_errors"])
                    clip_foot_errors_all_runs.extend(err["foot_pos_errors"])
                    clip_root_errors_all_runs.extend(err["root_pos_errors"])
                    clip_ee_rot_errors_all_runs.extend(err["ee_rot_errors_rad"])
                    clip_hand_rot_errors_all_runs.extend(err["hand_rot_errors_rad"])
                    clip_foot_rot_errors_all_runs.extend(err["foot_rot_errors_rad"])
                    clip_root_yaw_errors_all_runs.extend(err["root_yaw_errors_rad"])

                clip_stats = {
                    "count": int(np.mean(run_num_points)) if run_num_points else 0,
                    "mean_error_m": _safe_mean(run_means),
                    "max_error_m": _safe_mean(run_maxes),
                    "min_error_m": _safe_mean(run_mins),
                }
                task_row = {
                    "task": task_name,
                    "clip": clip_stem,
                    "prompt": prompt,
                    "constraints_path": str(persisted_constraints_path) if persisted_constraints_path else None,
                    "num_frames": int(num_frames),
                    "duration_sec": float(duration),
                    "num_keyframes": int(len(ee_constraint.get("frame_indices", []))),
                    "num_constraint_points": clip_stats["count"],
                    "five_point_mean_error_m": clip_stats["mean_error_m"],
                    "five_point_max_error_m": clip_stats["max_error_m"],
                    "five_point_min_error_m": clip_stats["min_error_m"],
                    "five_point_p95_error_m": _safe_percentile(clip_five_point_errors_all_runs, 95),
                    "ee4_mean_error_m": _safe_mean(clip_ee_errors_all_runs),
                    "hand2_mean_error_m": _safe_mean(clip_hand_errors_all_runs),
                    "foot2_mean_error_m": _safe_mean(clip_foot_errors_all_runs),
                    "root_mean_error_m": _safe_mean(clip_root_errors_all_runs),
                    "ee4_mean_rot_error_rad": _safe_mean(clip_ee_rot_errors_all_runs),
                    "hand2_mean_rot_error_rad": _safe_mean(clip_hand_rot_errors_all_runs),
                    "foot2_mean_rot_error_rad": _safe_mean(clip_foot_rot_errors_all_runs),
                    "root_mean_yaw_error_rad": _safe_mean(clip_root_yaw_errors_all_runs),
                    "generation_latency_mean_sec": _safe_mean(run_latencies),
                    "generation_latency_p50_sec": _safe_percentile(run_latencies, 50),
                    "generation_latency_p95_sec": _safe_percentile(run_latencies, 95),
                    "num_runs": int(args.num_runs_per_task),
                    **video_artifacts,
                }
                per_task.append(task_row)
                if progress_jsonl is not None:
                    with progress_jsonl.open("a", encoding="utf-8") as f:
                        f.write(json.dumps(task_row, ensure_ascii=False) + "\n")
                all_five_point_errors.extend(clip_five_point_errors_all_runs)
                all_ee_errors.extend(clip_ee_errors_all_runs)
                all_hand_errors.extend(clip_hand_errors_all_runs)
                all_foot_errors.extend(clip_foot_errors_all_runs)
                all_root_errors.extend(clip_root_errors_all_runs)
                all_ee_rot_errors.extend(clip_ee_rot_errors_all_runs)
                all_hand_rot_errors.extend(clip_hand_rot_errors_all_runs)
                all_foot_rot_errors.extend(clip_foot_rot_errors_all_runs)
                all_root_yaw_errors.extend(clip_root_yaw_errors_all_runs)
                all_generation_latencies.extend(run_latencies)
                if args.verbose:
                    print(
                        f"  five_mean={clip_stats['mean_error_m']:.6f} "
                        f"max={clip_stats['max_error_m']:.6f} "
                        f"latency={task_row['generation_latency_mean_sec']:.3f}s"
                    )
            except Exception as exc:  # noqa: BLE001
                failures.append({"task": task_name, "error": str(exc)})
                if args.verbose:
                    print(f"  FAILED: {exc}")
                else:
                    print(f"[{idx}/{len(csv_paths)}] FAILED: {task_name}")
                if args.fail_fast:
                    raise

    overall_five_point = _safe_stats(all_five_point_errors)
    overall_all_ee = _safe_stats(all_ee_errors)
    overall_hands = _safe_stats(all_hand_errors)
    overall_feet = _safe_stats(all_foot_errors)
    overall_root = _safe_stats(all_root_errors)
    overall_ee_rot = _safe_stats(all_ee_rot_errors)
    overall_hand_rot = _safe_stats(all_hand_rot_errors)
    overall_foot_rot = _safe_stats(all_foot_rot_errors)
    overall_root_yaw = _safe_stats(all_root_yaw_errors)
    per_task_means = [
        float(x["five_point_mean_error_m"]) for x in per_task if x.get("five_point_mean_error_m") is not None
    ]
    overall_mean_of_task_means = _safe_mean(per_task_means)

    result = {
        "settings": {
            "csv_root": str(csv_root),
            "timelines_jsonl": str(timeline_path),
            "csv_pattern": args.csv_pattern,
            "sample_ratio": float(args.sample_ratio),
            "max_tasks": args.max_tasks,
            "seed": int(args.seed),
            "model": args.model,
            "distill_config": args.distill_config,
            "distill_ckpt": args.distill_ckpt,
            "source_fps": float(args.source_fps),
            "target_fps": float(args.target_fps),
            "keyframe_step": int(args.keyframe_step),
            "max_eval_frames": int(args.max_eval_frames),
            "diffusion_steps": int(args.diffusion_steps),
            "num_runs_per_task": int(args.num_runs_per_task),
            "xml": args.xml,
            "constraints_out_dir": args.constraints_out_dir,
            "progress_jsonl": args.progress_jsonl,
            "videos_out_dir": args.videos_out_dir,
            "video_fps": int(args.video_fps),
            "video_width": int(args.video_width),
            "video_height": int(args.video_height),
        },
        "total_csv": len(all_csv_paths),
        "sampled_tasks": len(csv_paths),
        "success_tasks": len(per_task),
        "failed_tasks": len(failures),
        "per_task": per_task,
        "overall_five_point": overall_five_point,
        "overall_all_ee": overall_all_ee,
        "overall_hands_only": overall_hands,
        "overall_feet_only": overall_feet,
        "overall_root_only": overall_root,
        "overall_ee_rotation_rad": overall_ee_rot,
        "overall_hand_rotation_rad": overall_hand_rot,
        "overall_foot_rotation_rad": overall_foot_rot,
        "overall_root_yaw_rad": overall_root_yaw,
        "overall_mean_of_task_means_m": overall_mean_of_task_means,
        "generation_latency_sec": {
            "mean": _safe_mean(all_generation_latencies),
            "p50": _safe_percentile(all_generation_latencies, 50),
            "p95": _safe_percentile(all_generation_latencies, 95),
            "min": min(all_generation_latencies) if all_generation_latencies else None,
            "max": max(all_generation_latencies) if all_generation_latencies else None,
        },
        "failures": failures,
    }

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(
        "Done: "
        f"success={len(per_task)}/{len(csv_paths)}, failed={len(failures)}, "
        f"overall_five_point_mean_error_m={overall_five_point['mean_error_m']}, "
        f"overall_mean_of_task_means_m={overall_mean_of_task_means}, json={output_path}"
    )


if __name__ == "__main__":
    main()
