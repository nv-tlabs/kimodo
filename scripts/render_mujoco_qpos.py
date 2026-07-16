#!/usr/bin/env python3
"""Render a MuJoCo qpos CSV to MP4/PNG using the G1 XML."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kimodo.assets import skeleton_asset_path


def load_qpos(path: Path) -> np.ndarray:
    qpos = np.loadtxt(str(path), delimiter=",", dtype=np.float64)
    if qpos.ndim == 1:
        qpos = qpos[None, :]
    return qpos


def build_camera(qpos: np.ndarray) -> mujoco.MjvCamera:
    root = qpos[:, :3]
    root_min = root.min(axis=0)
    root_max = root.max(axis=0)
    span = float(np.linalg.norm(root_max[:2] - root_min[:2]))

    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = np.array(
        [
            0.5 * (root_min[0] + root_max[0]),
            0.5 * (root_min[1] + root_max[1]),
            max(0.7, float(np.median(root[:, 2]))),
        ],
        dtype=np.float64,
    )
    camera.distance = max(2.2, span + 2.0)
    camera.azimuth = 135.0
    camera.elevation = -18.0
    return camera


def render(csv_path: Path, output: Path, preview: Path, xml_path: Path, fps: int, width: int, height: int) -> None:
    qpos = load_qpos(csv_path)
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    if qpos.shape[1] != model.nq:
        raise ValueError(f"qpos has {qpos.shape[1]} columns but MuJoCo model.nq={model.nq}.")
    model.vis.global_.offwidth = int(width)
    model.vis.global_.offheight = int(height)

    data = mujoco.MjData(model)
    camera = build_camera(qpos)
    renderer = mujoco.Renderer(model, height=height, width=width)

    output.parent.mkdir(parents=True, exist_ok=True)
    preview.parent.mkdir(parents=True, exist_ok=True)

    writer = imageio.get_writer(str(output), fps=fps, codec="libx264", quality=8, macro_block_size=1)
    try:
        first_frame = None
        for frame_qpos in qpos:
            data.qpos[:] = frame_qpos
            mujoco.mj_forward(model, data)
            renderer.update_scene(data, camera=camera)
            frame = renderer.render()
            if first_frame is None:
                first_frame = frame.copy()
            writer.append_data(frame)
    finally:
        writer.close()
        renderer.close()

    if first_frame is not None:
        imageio.imwrite(str(preview), first_frame)

    print(f"Rendered {len(qpos)} frames")
    print(f"Video: {output}")
    print(f"Preview: {preview}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Render G1 MuJoCo qpos CSV to MP4.")
    parser.add_argument("--csv", required=True, type=Path, help="Input qpos CSV.")
    parser.add_argument("--output", required=True, type=Path, help="Output MP4 path.")
    parser.add_argument("--preview", type=Path, default=None, help="Output preview PNG path.")
    parser.add_argument("--xml", type=Path, default=Path(skeleton_asset_path("g1skel34", "xml", "g1.xml")))
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    args = parser.parse_args()

    preview = args.preview or args.output.with_suffix(".png")
    render(args.csv, args.output, preview, args.xml, args.fps, args.width, args.height)


if __name__ == "__main__":
    main()
