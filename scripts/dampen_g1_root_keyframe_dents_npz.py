#!/usr/bin/env python3
"""Dampen root/heading sawtooth dents only at constraint keyframes in a G1 NPZ."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dampen root/yaw keyframe dents in generated Kimodo G1 NPZ.")
    parser.add_argument("--input", required=True, help="Input g1_generated.npz.")
    parser.add_argument("--constraints", required=True, help="constraints_ee_pose.json with frame_indices.")
    parser.add_argument("--output", required=True, help="Output NPZ.")
    parser.add_argument("--strength", type=float, default=0.7, help="Dent removal strength in [0, 1]. Default: 0.7.")
    parser.add_argument("--root-threshold", type=float, default=0.003, help="Root dent threshold in meters.")
    parser.add_argument("--yaw-threshold", type=float, default=0.003, help="Yaw dent threshold in radians.")
    return parser.parse_args()


def load_keyframes(path: Path) -> np.ndarray:
    payload = json.loads(path.read_text(encoding="utf-8"))
    for item in payload:
        if item.get("type") == "ee-pose":
            return np.asarray(item.get("frame_indices", []), dtype=np.int64)
    raise ValueError(f"No ee-pose item found in {path}.")


def rot_y(angle: np.ndarray) -> np.ndarray:
    c = np.cos(angle)
    s = np.sin(angle)
    out = np.zeros((angle.shape[0], 3, 3), dtype=np.float64)
    out[:, 0, 0] = c
    out[:, 0, 2] = s
    out[:, 1, 1] = 1.0
    out[:, 2, 0] = -s
    out[:, 2, 2] = c
    return out


def main() -> None:
    args = parse_args()
    strength = float(np.clip(args.strength, 0.0, 1.0))
    arrays = {key: value for key, value in np.load(args.input, allow_pickle=True).items()}
    keyframes = load_keyframes(Path(args.constraints))

    root = np.asarray(arrays["root_positions"], dtype=np.float64)
    heading = np.asarray(arrays["global_root_heading"], dtype=np.float64)
    posed = np.asarray(arrays["posed_joints"], dtype=np.float64)
    angle = np.unwrap(np.arctan2(heading[:, 1], heading[:, 0]))

    valid = keyframes[(keyframes > 0) & (keyframes < root.shape[0] - 1)]
    new_root = root.copy()
    new_angle = angle.copy()
    root_changed = 0
    yaw_changed = 0

    for frame in valid:
        root_neighbor = 0.5 * (root[frame - 1] + root[frame + 1])
        root_dent = root[frame] - root_neighbor
        if np.linalg.norm(root_dent) > float(args.root_threshold):
            new_root[frame] = root[frame] - strength * root_dent
            root_changed += 1

        yaw_neighbor = 0.5 * (angle[frame - 1] + angle[frame + 1])
        yaw_dent = angle[frame] - yaw_neighbor
        if abs(yaw_dent) > float(args.yaw_threshold):
            new_angle[frame] = angle[frame] - strength * yaw_dent
            yaw_changed += 1

    delta_angle = new_angle - angle
    delta_rot = rot_y(delta_angle)
    rel = posed - root[:, None, :]
    new_posed = np.einsum("tij,tkj->tki", delta_rot, rel) + new_root[:, None, :]

    arrays["root_positions"] = new_root.astype(arrays["root_positions"].dtype, copy=False)
    if "smooth_root_pos" in arrays:
        arrays["smooth_root_pos"] = new_root.astype(arrays["smooth_root_pos"].dtype, copy=False)
    arrays["global_root_heading"] = np.stack([np.cos(new_angle), np.sin(new_angle)], axis=1).astype(
        arrays["global_root_heading"].dtype,
        copy=False,
    )
    arrays["posed_joints"] = new_posed.astype(arrays["posed_joints"].dtype, copy=False)

    if "global_rot_mats" in arrays:
        global_rot = np.asarray(arrays["global_rot_mats"], dtype=np.float64)
        arrays["global_rot_mats"] = np.einsum("tij,tkjl->tkil", delta_rot, global_rot).astype(
            arrays["global_rot_mats"].dtype,
            copy=False,
        )

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(str(out), **arrays)
    print(f"Saved dampened NPZ: {out}")
    print(f"root keyframes changed={root_changed}, yaw keyframes changed={yaw_changed}, strength={strength}")


if __name__ == "__main__":
    main()
