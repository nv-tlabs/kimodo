#!/usr/bin/env python3
"""Smooth only root xyz and yaw in a G1 MuJoCo qpos CSV.

This is a diagnostic post-process: qpos[:, 7:] joint angles are left unchanged.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smooth root xyz/yaw of a G1 generated qpos CSV.")
    parser.add_argument("--input", required=True, help="Input g1_generated.csv.")
    parser.add_argument("--output", required=True, help="Output smoothed CSV.")
    parser.add_argument("--window", type=int, default=9, help="Odd smoothing window size. Default: 9.")
    parser.add_argument("--passes", type=int, default=1, help="Number of smoothing passes. Default: 1.")
    parser.add_argument("--no-root-xyz", action="store_true", help="Do not smooth root xyz.")
    parser.add_argument("--no-yaw", action="store_true", help="Do not smooth root yaw.")
    return parser.parse_args()


def load_qpos(path: Path) -> np.ndarray:
    qpos = np.loadtxt(str(path), delimiter=",", dtype=np.float64)
    if qpos.ndim == 1:
        qpos = qpos[None, :]
    if qpos.shape[1] != 36:
        raise ValueError(f"Expected qpos CSV with 36 columns, got {qpos.shape}.")
    return qpos


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
            cols = [np.convolve(padded[:, i], kernel, mode="valid") for i in range(out.shape[1])]
            out = np.stack(cols, axis=1)
    return out


def quat_normalize(q: np.ndarray) -> np.ndarray:
    return q / np.linalg.norm(q, axis=-1, keepdims=True)


def quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aw, ax, ay, az = np.moveaxis(a, -1, 0)
    bw, bx, by, bz = np.moveaxis(b, -1, 0)
    return np.stack(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        axis=-1,
    )


def yaw_from_quat_wxyz(q: np.ndarray) -> np.ndarray:
    q = quat_normalize(q)
    w, x, y, z = np.moveaxis(q, -1, 0)
    return np.unwrap(np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))


def yaw_delta_quat(delta_yaw: np.ndarray) -> np.ndarray:
    half = 0.5 * delta_yaw
    q = np.zeros((delta_yaw.shape[0], 4), dtype=np.float64)
    q[:, 0] = np.cos(half)
    q[:, 3] = np.sin(half)
    return q


def main() -> None:
    args = parse_args()
    qpos = load_qpos(Path(args.input))
    out = qpos.copy()

    if not args.no_root_xyz:
        out[:, :3] = smooth_series(qpos[:, :3], window=int(args.window), passes=int(args.passes))

    if not args.no_yaw:
        yaw = yaw_from_quat_wxyz(qpos[:, 3:7])
        smooth_yaw = smooth_series(yaw, window=int(args.window), passes=int(args.passes))
        delta = smooth_yaw - yaw
        # MuJoCo root quaternion is wxyz. Pre-multiply by a world-z yaw delta,
        # preserving the original non-yaw tilt as much as possible.
        out[:, 3:7] = quat_normalize(quat_mul(yaw_delta_quat(delta), qpos[:, 3:7]))

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(str(output), out, delimiter=",")
    print(f"Saved smoothed CSV: {output}")
    print(f"frames={out.shape[0]}, window={args.window}, passes={args.passes}")
    print("Changed columns: root xyz" + ("" if args.no_yaw else " + root yaw") + "; joint qpos columns unchanged.")


if __name__ == "__main__":
    main()
