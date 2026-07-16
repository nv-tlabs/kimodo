#!/usr/bin/env python3
"""Render a G1 MuJoCo qpos CSV with sparse Kimodo constraint targets highlighted."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable

import imageio.v2 as imageio
import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kimodo.assets import skeleton_asset_path


POINTS = {
    "root_xyzyaw": {
        "label": "root",
        "rgba": np.array([1.0, 0.0, 0.9, 1.0], dtype=np.float32),
        "radius": 0.055,
    },
    "left_hand_pose": {
        "label": "left_hand",
        "rgba": np.array([0.0, 0.85, 1.0, 1.0], dtype=np.float32),
        "radius": 0.042,
    },
    "right_hand_pose": {
        "label": "right_hand",
        "rgba": np.array([1.0, 0.05, 0.05, 1.0], dtype=np.float32),
        "radius": 0.042,
    },
    "left_foot_pose": {
        "label": "left_foot",
        "rgba": np.array([1.0, 0.85, 0.0, 1.0], dtype=np.float32),
        "radius": 0.040,
    },
    "right_foot_pose": {
        "label": "right_foot",
        "rgba": np.array([0.05, 1.0, 0.15, 1.0], dtype=np.float32),
        "radius": 0.040,
    },
}


def load_qpos(path: Path) -> np.ndarray:
    qpos = np.loadtxt(str(path), delimiter=",", dtype=np.float64)
    if qpos.ndim == 1:
        qpos = qpos[None, :]
    return qpos


def kimodo_xyz_to_mujoco(xyz_k: Iterable[float]) -> np.ndarray:
    xyz_k = np.asarray(xyz_k, dtype=np.float64)
    return np.asarray([xyz_k[2], xyz_k[0], xyz_k[1]], dtype=np.float64)


def load_constraint_points(path: Path) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"Constraints JSON must be a list: {path}")

    out: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for item in payload:
        if item.get("type") != "ee-pose":
            continue
        frame_indices = np.asarray(item.get("frame_indices", []), dtype=np.int64)
        if frame_indices.size == 0:
            continue
        for field in POINTS:
            values = item.get(field)
            if not values:
                continue
            arr = np.asarray(values, dtype=np.float64)
            if arr.shape[0] != frame_indices.shape[0] or arr.shape[1] < 3:
                raise ValueError(
                    f"Constraint field {field} has shape {arr.shape}, "
                    f"expected [{frame_indices.shape[0]}, >=3]."
                )
            xyz_m = np.stack([kimodo_xyz_to_mujoco(v[:3]) for v in arr], axis=0)
            out[field] = (frame_indices, xyz_m)
    if not out:
        raise ValueError(f"No ee-pose constraint points found in: {path}")
    return out


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


def add_sphere(scene: mujoco.MjvScene, pos: np.ndarray, radius: float, rgba: np.ndarray) -> None:
    if scene.ngeom >= scene.maxgeom:
        return
    mujoco.mjv_initGeom(
        scene.geoms[scene.ngeom],
        mujoco.mjtGeom.mjGEOM_SPHERE,
        np.asarray([radius, radius, radius], dtype=np.float64),
        np.asarray(pos, dtype=np.float64),
        np.eye(3, dtype=np.float64).reshape(-1),
        rgba.astype(np.float32, copy=False),
    )
    scene.ngeom += 1


def add_capsule(scene: mujoco.MjvScene, start: np.ndarray, end: np.ndarray, radius: float, rgba: np.ndarray) -> None:
    if scene.ngeom >= scene.maxgeom:
        return
    mujoco.mjv_initGeom(
        scene.geoms[scene.ngeom],
        mujoco.mjtGeom.mjGEOM_CAPSULE,
        np.asarray([radius, 0.0, 0.0], dtype=np.float64),
        np.zeros(3, dtype=np.float64),
        np.eye(3, dtype=np.float64).reshape(-1),
        rgba.astype(np.float32, copy=False),
    )
    mujoco.mjv_connector(
        scene.geoms[scene.ngeom],
        mujoco.mjtGeom.mjGEOM_CAPSULE,
        radius,
        np.asarray(start, dtype=np.float64),
        np.asarray(end, dtype=np.float64),
    )
    scene.ngeom += 1


def nearest_index(frames: np.ndarray, frame: int) -> int:
    return int(np.argmin(np.abs(frames - int(frame))))


def add_constraint_overlay(
    scene: mujoco.MjvScene,
    points: dict[str, tuple[np.ndarray, np.ndarray]],
    frame: int,
    trail: bool,
) -> None:
    for field, (frames, xyz) in points.items():
        style = POINTS[field]
        rgba = style["rgba"].copy()
        radius = float(style["radius"])

        if trail:
            trail_rgba = rgba.copy()
            trail_rgba[3] = 0.32
            for p in xyz:
                add_sphere(scene, p, radius * 0.45, trail_rgba)
            line_rgba = rgba.copy()
            line_rgba[3] = 0.55
            for a, b in zip(xyz[:-1], xyz[1:]):
                add_capsule(scene, a, b, radius * 0.08, line_rgba)

        i = nearest_index(frames, frame)
        if int(frames[i]) == int(frame):
            cur_rgba = rgba
            cur_radius = radius * 1.35
        else:
            cur_rgba = rgba.copy()
            cur_rgba[3] = 0.70
            cur_radius = radius
        add_sphere(scene, xyz[i], cur_radius, cur_rgba)


def render(
    csv_path: Path,
    constraints_path: Path,
    output: Path,
    preview: Path,
    xml_path: Path,
    fps: int,
    width: int,
    height: int,
    trail: bool,
) -> None:
    qpos = load_qpos(csv_path)
    points = load_constraint_points(constraints_path)
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
        for frame_idx, frame_qpos in enumerate(qpos):
            data.qpos[:] = frame_qpos
            mujoco.mj_forward(model, data)
            renderer.update_scene(data, camera=camera)
            add_constraint_overlay(renderer.scene, points, frame_idx, trail=trail)
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
    print(f"Constraints: {constraints_path}")
    for field, (frames, _) in points.items():
        print(f"  {POINTS[field]['label']}: {len(frames)} sparse keyframes")
    print(f"Video: {output}")
    print(f"Preview: {preview}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Render G1 MuJoCo qpos with sparse constraint target overlay.")
    parser.add_argument("--csv", required=True, type=Path, help="Input qpos CSV.")
    parser.add_argument("--constraints", required=True, type=Path, help="Kimodo constraints JSON.")
    parser.add_argument("--output", required=True, type=Path, help="Output MP4 path.")
    parser.add_argument("--preview", type=Path, default=None, help="Output preview PNG path.")
    parser.add_argument("--xml", type=Path, default=Path(skeleton_asset_path("g1skel34", "xml", "g1.xml")))
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--no-trail", action="store_true", help="Show only nearest/current target point, no sparse trajectory.")
    args = parser.parse_args()

    preview = args.preview or args.output.with_suffix(".png")
    render(
        csv_path=args.csv,
        constraints_path=args.constraints,
        output=args.output,
        preview=preview,
        xml_path=args.xml,
        fps=args.fps,
        width=args.width,
        height=args.height,
        trail=not args.no_trail,
    )


if __name__ == "__main__":
    main()
