#!/usr/bin/env python3
"""Evaluate generated G1 output folders.

For each folder, this script reads:
  - g1_generated.csv
  - constraints_ee_pose.json

It reports sparse keyframe end-effector tracking error plus qpos acceleration
and jerk p95 jitter metrics.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
from pathlib import Path
from typing import Any

import numpy as np


EE_FIELDS = (
    ("left_hand_pose", "left_wrist_yaw_link", "left_hand"),
    ("right_hand_pose", "right_wrist_yaw_link", "right_hand"),
    ("left_foot_pose", "left_ankle_roll_link", "left_foot"),
    ("right_foot_pose", "right_ankle_roll_link", "right_foot"),
)


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description=(
            "Compute EE mean/max error and self acc/jerk p95 for one or more "
            "Kimodo G1 pipeline output folders."
        )
    )
    parser.add_argument(
        "folders",
        nargs="+",
        help="Output folders containing g1_generated.csv and constraints_ee_pose.json.",
    )
    parser.add_argument("--csv-name", default="g1_generated.csv", help="CSV filename inside each folder.")
    parser.add_argument(
        "--constraints-name",
        default="constraints_ee_pose.json",
        help="Constraints JSON filename inside each folder.",
    )
    parser.add_argument(
        "--xml",
        default=str(repo_root / "kimodo/assets/skeletons/g1skel34/xml/g1.xml"),
        help="G1 MuJoCo XML path.",
    )
    parser.add_argument("--generated-fps", type=float, default=30.0, help="Generated CSV FPS.")
    parser.add_argument("--constraints-fps", type=float, default=30.0, help="Constraints frame-index FPS.")
    parser.add_argument(
        "--format",
        choices=("markdown", "csv", "json"),
        default="markdown",
        help="Output format.",
    )
    parser.add_argument(
        "--jitter-only",
        action="store_true",
        help="Only compute CSV self acc/jerk metrics; does not require MuJoCo.",
    )
    parser.add_argument("--output", default=None, help="Optional output file path.")
    return parser.parse_args()


def kimodo_xyz_to_mujoco(xyz_k: np.ndarray) -> np.ndarray:
    # Kimodo: y-up, z-forward; MuJoCo: z-up, x-forward.
    return np.asarray([xyz_k[2], xyz_k[0], xyz_k[1]], dtype=np.float64)


def load_qpos_csv(path: Path) -> np.ndarray:
    arr = np.loadtxt(str(path), delimiter=",", dtype=np.float64)
    if arr.ndim == 1:
        arr = arr[None, :]
    if arr.shape[1] != 36:
        raise ValueError(f"Expected qpos CSV with 36 columns, got {arr.shape} in {path}")
    return arr


def load_ee_constraint(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"Constraints JSON must be a list: {path}")
    for item in payload:
        if item.get("type") == "ee-pose":
            return item
    raise ValueError(f"No ee-pose item found in constraints: {path}")


def resolve_constraints_path(folder: Path, constraints_name: str) -> Path:
    path = folder / constraints_name
    if path.exists():
        return path

    # Convenience fallback for a common typo.
    typo_path = folder / "cinstraints_ee_pose.json"
    if constraints_name == "constraints_ee_pose.json" and typo_path.exists():
        return typo_path

    raise FileNotFoundError(f"Constraints JSON not found: {path}")


def build_constraint_keyframes(ee_item: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, list[str]]:
    frames = np.asarray(ee_item.get("frame_indices", []), dtype=np.int64)
    if frames.size == 0:
        raise ValueError("ee-pose constraint has empty frame_indices.")

    names: list[str] = []
    coords = []
    for field, _, out_name in EE_FIELDS:
        values = ee_item.get(field)
        if not values:
            raise ValueError(f"Constraint field missing/empty: {field}")
        if len(values) != len(frames):
            raise ValueError(f"Constraint field length mismatch: {field} vs frame_indices")
        xyz_k = np.asarray(values, dtype=np.float64)[:, :3]
        coords.append(np.stack([kimodo_xyz_to_mujoco(v) for v in xyz_k], axis=0))
        names.append(out_name)
    return frames, np.stack(coords, axis=1), names


def forward_ee_positions(mujoco: Any, model: Any, qpos: np.ndarray) -> tuple[np.ndarray, list[str]]:
    data = mujoco.MjData(model)
    body_ids = []
    names = []
    for _, body_name, out_name in EE_FIELDS:
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if body_id < 0:
            raise ValueError(f"Body not found in XML: {body_name}")
        body_ids.append(body_id)
        names.append(out_name)

    out = np.zeros((qpos.shape[0], len(body_ids), 3), dtype=np.float64)
    for frame_idx in range(qpos.shape[0]):
        data.qpos[:] = qpos[frame_idx]
        mujoco.mj_forward(model, data)
        for joint_idx, body_id in enumerate(body_ids):
            out[frame_idx, joint_idx] = data.xpos[body_id]
    return out, names


def forward_body_positions(mujoco: Any, model: Any, qpos: np.ndarray) -> tuple[np.ndarray, list[str]]:
    data = mujoco.MjData(model)
    body_ids = list(range(1, model.nbody))  # Skip world.
    names = []
    for body_id in body_ids:
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)
        names.append(str(name) if name else f"body_{body_id}")

    out = np.zeros((qpos.shape[0], len(body_ids), 3), dtype=np.float64)
    for frame_idx in range(qpos.shape[0]):
        data.qpos[:] = qpos[frame_idx]
        mujoco.mj_forward(model, data)
        for joint_idx, body_id in enumerate(body_ids):
            out[frame_idx, joint_idx] = data.xpos[body_id]
    return out, names


def align_generated_to_constraints(
    generated_xyz: np.ndarray,
    generated_fps: float,
    constraint_frames: np.ndarray,
    constraints_fps: float,
) -> tuple[np.ndarray, np.ndarray]:
    times = constraint_frames.astype(np.float64) / float(constraints_fps)
    mapped_frames = np.rint(times * float(generated_fps)).astype(np.int64)
    mapped_frames = np.clip(mapped_frames, 0, generated_xyz.shape[0] - 1)
    return generated_xyz[mapped_frames], mapped_frames


def compute_ee_metrics(
    generated_xyz: np.ndarray,
    joint_names: list[str],
    constraint_frames: np.ndarray,
    constraint_xyz: np.ndarray,
    generated_fps: float,
    constraints_fps: float,
) -> dict[str, Any]:
    pred, mapped_frames = align_generated_to_constraints(
        generated_xyz,
        generated_fps=generated_fps,
        constraint_frames=constraint_frames,
        constraints_fps=constraints_fps,
    )
    if pred.shape != constraint_xyz.shape:
        raise ValueError(f"Shape mismatch: pred {pred.shape} vs constraints {constraint_xyz.shape}")

    err = np.linalg.norm(pred - constraint_xyz, axis=-1)
    max_flat = int(np.argmax(err))
    max_keyframe_idx, max_joint_idx = np.unravel_index(max_flat, err.shape)
    return {
        "ee_count": int(err.size),
        "ee_mean_m": float(err.mean()),
        "ee_max_m": float(err.max()),
        "ee_min_m": float(err.min()),
        "max_joint": joint_names[max_joint_idx],
        "max_constraint_frame": int(constraint_frames[max_keyframe_idx]),
        "max_generated_frame": int(mapped_frames[max_keyframe_idx]),
    }


def compute_jitter_metrics(qpos: np.ndarray) -> dict[str, Any]:
    if qpos.shape[0] < 4:
        raise ValueError(f"Need at least 4 frames for jerk, got {qpos.shape[0]}")
    acc = np.diff(qpos, n=2, axis=0)
    jerk = np.diff(qpos, n=3, axis=0)
    acc_frame = np.mean(np.abs(acc), axis=1)
    jerk_frame = np.mean(np.abs(jerk), axis=1)
    acc_l2 = np.linalg.norm(acc, axis=1)
    jerk_l2 = np.linalg.norm(jerk, axis=1)
    return {
        "acc_p95": float(np.percentile(acc_frame, 95)),
        "jerk_p95": float(np.percentile(jerk_frame, 95)),
        "acc_l2_p95": float(np.percentile(acc_l2, 95)),
        "jerk_l2_p95": float(np.percentile(jerk_l2, 95)),
        "acc_max": float(acc_frame.max()),
        "jerk_max": float(jerk_frame.max()),
        "acc_max_frame": int(np.argmax(acc_frame) + 1),
        "jerk_max_frame": int(np.argmax(jerk_frame) + 1),
    }


def compute_body_position_metrics(body_xyz: np.ndarray, body_names: list[str]) -> dict[str, Any]:
    if body_xyz.shape[0] < 4:
        raise ValueError(f"Need at least 4 frames for body position jitter, got {body_xyz.shape[0]}")

    step = np.linalg.norm(np.diff(body_xyz, axis=0), axis=-1)
    acc = np.linalg.norm(np.diff(body_xyz, n=2, axis=0), axis=-1)
    jerk = np.linalg.norm(np.diff(body_xyz, n=3, axis=0), axis=-1)

    root_step = step[:, 0]
    root_acc = acc[:, 0]
    root_jerk = jerk[:, 0]

    max_step_flat = int(np.argmax(step))
    max_step_frame, max_step_body = np.unravel_index(max_step_flat, step.shape)
    max_acc_flat = int(np.argmax(acc))
    max_acc_frame, max_acc_body = np.unravel_index(max_acc_flat, acc.shape)
    max_jerk_flat = int(np.argmax(jerk))
    max_jerk_frame, max_jerk_body = np.unravel_index(max_jerk_flat, jerk.shape)

    return {
        "body_step_p95_m": float(np.percentile(step, 95)),
        "body_step_max_m": float(step.max()),
        "body_step_max_body": body_names[max_step_body],
        "body_step_max_frame": int(max_step_frame),
        "body_acc_p95_m": float(np.percentile(acc, 95)),
        "body_acc_max_m": float(acc.max()),
        "body_acc_max_body": body_names[max_acc_body],
        "body_acc_max_frame": int(max_acc_frame + 1),
        "body_jerk_p95_m": float(np.percentile(jerk, 95)),
        "body_jerk_max_m": float(jerk.max()),
        "body_jerk_max_body": body_names[max_jerk_body],
        "body_jerk_max_frame": int(max_jerk_frame + 1),
        "root_step_p95_m": float(np.percentile(root_step, 95)),
        "root_step_max_m": float(root_step.max()),
        "root_acc_p95_m": float(np.percentile(root_acc, 95)),
        "root_acc_max_m": float(root_acc.max()),
        "root_jerk_p95_m": float(np.percentile(root_jerk, 95)),
        "root_jerk_max_m": float(root_jerk.max()),
    }


def evaluate_folder(
    folder: Path,
    mujoco: Any,
    model: Any,
    *,
    csv_name: str,
    constraints_name: str,
    generated_fps: float,
    constraints_fps: float,
) -> dict[str, Any]:
    csv_path = folder / csv_name
    constraints_path = resolve_constraints_path(folder, constraints_name)
    if not csv_path.exists():
        raise FileNotFoundError(f"Generated CSV not found: {csv_path}")

    qpos = load_qpos_csv(csv_path)
    ee_item = load_ee_constraint(constraints_path)
    constraint_frames, constraint_xyz, constraint_joint_names = build_constraint_keyframes(ee_item)
    generated_xyz, joint_names = forward_ee_positions(mujoco, model, qpos)
    body_xyz, body_names = forward_body_positions(mujoco, model, qpos)
    if joint_names != constraint_joint_names:
        raise ValueError(f"Joint order mismatch: generated={joint_names}, constraints={constraint_joint_names}")

    result = {
        "run": folder.name,
        "folder": str(folder),
        "csv": str(csv_path),
        "constraints": str(constraints_path),
        "frames": int(qpos.shape[0]),
        "constraint_keyframes": int(constraint_frames.size),
    }
    result.update(
        compute_ee_metrics(
            generated_xyz,
            joint_names,
            constraint_frames,
            constraint_xyz,
            generated_fps=generated_fps,
            constraints_fps=constraints_fps,
        )
    )
    result.update(compute_jitter_metrics(qpos))
    result.update(compute_body_position_metrics(body_xyz, body_names))
    return result


def evaluate_folder_jitter_only(folder: Path, *, csv_name: str) -> dict[str, Any]:
    csv_path = folder / csv_name
    if not csv_path.exists():
        raise FileNotFoundError(f"Generated CSV not found: {csv_path}")
    qpos = load_qpos_csv(csv_path)
    result = {
        "run": folder.name,
        "folder": str(folder),
        "csv": str(csv_path),
        "constraints": "",
        "frames": int(qpos.shape[0]),
        "constraint_keyframes": 0,
        "ee_count": 0,
        "ee_mean_m": float("nan"),
        "ee_max_m": float("nan"),
        "ee_min_m": float("nan"),
        "max_joint": "",
        "max_constraint_frame": -1,
        "max_generated_frame": -1,
    }
    result.update(compute_jitter_metrics(qpos))
    result.update(
        {
            "body_step_p95_m": float("nan"),
            "body_step_max_m": float("nan"),
            "body_step_max_body": "",
            "body_step_max_frame": -1,
            "body_acc_p95_m": float("nan"),
            "body_acc_max_m": float("nan"),
            "body_acc_max_body": "",
            "body_acc_max_frame": -1,
            "body_jerk_p95_m": float("nan"),
            "body_jerk_max_m": float("nan"),
            "body_jerk_max_body": "",
            "body_jerk_max_frame": -1,
            "root_step_p95_m": float("nan"),
            "root_step_max_m": float("nan"),
            "root_acc_p95_m": float("nan"),
            "root_acc_max_m": float("nan"),
            "root_jerk_p95_m": float("nan"),
            "root_jerk_max_m": float("nan"),
        }
    )
    return result


def format_markdown(rows: list[dict[str, Any]]) -> str:
    lines = [
        "| run | frames | EE mean m | EE max m | max corresponding | acc L2 p95 | jerk L2 p95 | body step p95 m | body step max | root step p95 m |",
        "|---|---:|---:|---:|---|---:|---:|---:|---|---:|",
    ]
    for row in rows:
        has_ee = int(row["ee_count"]) > 0
        ee_mean = f"{row['ee_mean_m']:.5f}" if has_ee else "-"
        ee_max = f"{row['ee_max_m']:.5f}" if has_ee else "-"
        max_desc = f"{row['max_joint']}, frame {row['max_constraint_frame']}" if has_ee else "-"
        if has_ee and row["max_generated_frame"] != row["max_constraint_frame"]:
            max_desc += f" -> gen {row['max_generated_frame']}"
        body_step_p95 = f"{row['body_step_p95_m']:.5f}" if has_ee else "-"
        root_step_p95 = f"{row['root_step_p95_m']:.5f}" if has_ee else "-"
        body_step_max = (
            f"{row['body_step_max_m']:.5f}, {row['body_step_max_body']}, frame {row['body_step_max_frame']}"
            if has_ee
            else "-"
        )
        lines.append(
            "| {run} | {frames:d} | {ee_mean} | {ee_max} | {max_desc} | "
            "{acc_l2_p95:.5f} | {jerk_l2_p95:.5f} | {body_step_p95} | {body_step_max} | {root_step_p95} |".format(
                ee_mean=ee_mean,
                ee_max=ee_max,
                max_desc=max_desc,
                body_step_p95=body_step_p95,
                body_step_max=body_step_max,
                root_step_p95=root_step_p95,
                **row,
            )
        )
    return "\n".join(lines)


def format_csv(rows: list[dict[str, Any]]) -> str:
    headers = [
        "run",
        "folder",
        "frames",
        "constraint_keyframes",
        "ee_count",
        "ee_mean_m",
        "ee_max_m",
        "ee_min_m",
        "max_joint",
        "max_constraint_frame",
        "max_generated_frame",
        "acc_p95",
        "jerk_p95",
        "acc_l2_p95",
        "jerk_l2_p95",
        "acc_max",
        "jerk_max",
        "acc_max_frame",
        "jerk_max_frame",
        "body_step_p95_m",
        "body_step_max_m",
        "body_step_max_body",
        "body_step_max_frame",
        "body_acc_p95_m",
        "body_acc_max_m",
        "body_acc_max_body",
        "body_acc_max_frame",
        "body_jerk_p95_m",
        "body_jerk_max_m",
        "body_jerk_max_body",
        "body_jerk_max_frame",
        "root_step_p95_m",
        "root_step_max_m",
        "root_acc_p95_m",
        "root_acc_max_m",
        "root_jerk_p95_m",
        "root_jerk_max_m",
    ]
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=headers, extrasaction="ignore")
    writer.writeheader()
    for row in rows:
        writer.writerow(row)
    return buf.getvalue().rstrip("\n")


def main() -> None:
    args = parse_args()
    if args.jitter_only:
        rows = [evaluate_folder_jitter_only(Path(folder), csv_name=args.csv_name) for folder in args.folders]
    else:
        try:
            import mujoco
        except ImportError as exc:
            raise SystemExit(
                "mujoco is required for CSV+constraints EE evaluation. "
                "Run this script in the Kimodo/MuJoCo environment, or pass --jitter-only."
            ) from exc

        xml_path = Path(args.xml)
        if not xml_path.exists():
            raise FileNotFoundError(f"MuJoCo XML not found: {xml_path}")
        model = mujoco.MjModel.from_xml_path(str(xml_path))

        rows = [
            evaluate_folder(
                Path(folder),
                mujoco,
                model,
                csv_name=args.csv_name,
                constraints_name=args.constraints_name,
                generated_fps=float(args.generated_fps),
                constraints_fps=float(args.constraints_fps),
            )
            for folder in args.folders
        ]

    if args.format == "markdown":
        text = format_markdown(rows)
    elif args.format == "csv":
        text = format_csv(rows)
    else:
        text = json.dumps(rows, indent=2)

    if args.output:
        Path(args.output).write_text(text + "\n", encoding="utf-8")
    else:
        print(text)


if __name__ == "__main__":
    main()
