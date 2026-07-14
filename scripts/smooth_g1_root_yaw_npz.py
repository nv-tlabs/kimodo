#!/usr/bin/env python3
"""Smooth only root position and heading in a generated Kimodo G1 NPZ.

This is a diagnostic post-process for NPZ-only viewers. It preserves local
joint rotations and foot contacts, and applies the root/yaw delta as a rigid
world transform to posed_joints/global_rot_mats.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smooth root position/heading of a generated G1 NPZ.")
    parser.add_argument("--input", required=True, help="Input g1_generated.npz.")
    parser.add_argument("--output", required=True, help="Output smoothed NPZ.")
    parser.add_argument("--window", type=int, default=9, help="Odd smoothing window size. Default: 9.")
    parser.add_argument("--passes", type=int, default=1, help="Number of smoothing passes. Default: 1.")
    parser.add_argument("--no-root", action="store_true", help="Do not smooth root position.")
    parser.add_argument("--no-heading", action="store_true", help="Do not smooth global root heading.")
    return parser.parse_args()


def smooth_series(x: np.ndarray, window: int, passes: int) -> np.ndarray:
    if window <= 1 or passes <= 0:
        return x.copy()
    if window % 2 == 0:
        raise ValueError(f"--window must be odd, got {window}.")
    if window > x.shape[0]:
        raise ValueError(f"--window {window} exceeds frame count {x.shape[0]}.")

    kernel = np.ones(window, dtype=np.float64) / float(window)
    pad = window // 2
    out = x.astype(np.float64, copy=True)
    for _ in range(passes):
        padded = np.pad(out, [(pad, pad)] + [(0, 0)] * (out.ndim - 1), mode="edge")
        if out.ndim == 1:
            out = np.convolve(padded, kernel, mode="valid")
        else:
            out = np.stack(
                [np.convolve(padded[:, col], kernel, mode="valid") for col in range(out.shape[1])],
                axis=1,
            )
    return out


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
    inp = Path(args.input)
    data = np.load(str(inp), allow_pickle=True)
    arrays = {key: data[key] for key in data.files}

    required = {"root_positions", "global_root_heading", "posed_joints"}
    missing = required - set(arrays)
    if missing:
        raise ValueError(f"Missing required arrays in {inp}: {sorted(missing)}")

    root = np.asarray(arrays["root_positions"], dtype=np.float64)
    heading = np.asarray(arrays["global_root_heading"], dtype=np.float64)
    posed = np.asarray(arrays["posed_joints"], dtype=np.float64)
    if root.ndim != 2 or root.shape[1] != 3:
        raise ValueError(f"root_positions must have shape [T, 3], got {root.shape}.")
    if heading.ndim != 2 or heading.shape[1] != 2:
        raise ValueError(f"global_root_heading must have shape [T, 2], got {heading.shape}.")
    if posed.ndim != 3 or posed.shape[0] != root.shape[0] or posed.shape[2] != 3:
        raise ValueError(f"posed_joints must have shape [T, J, 3], got {posed.shape}.")

    old_angle = np.unwrap(np.arctan2(heading[:, 1], heading[:, 0]))
    new_root = root.copy() if args.no_root else smooth_series(root, args.window, args.passes)
    new_angle = old_angle.copy() if args.no_heading else smooth_series(old_angle, args.window, args.passes)

    delta_angle = new_angle - old_angle
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
        if global_rot.ndim == 4 and global_rot.shape[0] == root.shape[0] and global_rot.shape[-2:] == (3, 3):
            arrays["global_rot_mats"] = np.einsum("tij,tkjl->tkil", delta_rot, global_rot).astype(
                arrays["global_rot_mats"].dtype,
                copy=False,
            )

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(str(out), **arrays)
    print(f"Saved smoothed NPZ: {out}")
    print(f"frames={root.shape[0]}, window={args.window}, passes={args.passes}")
    print("Preserved local_rot_mats and foot_contacts; updated root/heading/posed_joints/global_rot_mats.")


if __name__ == "__main__":
    main()
