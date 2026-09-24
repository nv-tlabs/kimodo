# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Step (3) of evaluation pipeline.

This script recursively embeds generated motions, ground-truth motions, and text prompts from a test suite folder tree with the pre-trained TMR model.
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from kimodo.device import resolve_device
from kimodo.meta import parse_prompts_from_meta
from kimodo.model.load_model import load_model
from kimodo.tools import load_json


def discover_motion_folders(root: Path) -> list[Path]:
    root = root.resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Folder does not exist: {root}")
    out: list[Path] = []
    for meta_path in root.rglob("meta.json"):
        src_dir = meta_path.parent
        if (src_dir / "motion.npz").is_file() or (src_dir / "gt_motion.npz").is_file():
            out.append(src_dir)
    return sorted(out)


def _load_posed_joints(npz_path: Path, device: str) -> torch.Tensor:
    data = np.load(npz_path)
    if "posed_joints" not in data:
        raise SystemExit(f"NPZ must contain 'posed_joints': {npz_path}")
    posed_joints = data["posed_joints"]
    if posed_joints.ndim == 4:
        if posed_joints.shape[0] != 1:
            raise SystemExit(f"Expected batch size 1 for posed_joints, got {posed_joints.shape[0]} in {npz_path}")
        posed_joints = posed_joints[0]
    if posed_joints.ndim != 3:
        raise SystemExit(f"Expected posed_joints shape [T, J, 3], got {posed_joints.shape} in {npz_path}")
    return torch.from_numpy(posed_joints).float().to(device)


def main():
    parser = argparse.ArgumentParser(
        description="Recursively embed motion, gt_motion, and text; save motion_embedding.npy, gt_motion_embedding.npy, and text_embedding.npy when present.",
    )
    parser.add_argument(
        "folder",
        type=Path,
        help="Root folder to search recursively for meta.json and motion.npz and/or gt_motion.npz",
    )
    parser.add_argument(
        "--model",
        default="tmr-soma-rp",
        help="Model for encoding (e.g. TMR-SOMA-RP-v1, tmr-soma-rp). Default: tmr-soma-rp",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Device: auto, mps, cuda, or cpu (default: auto)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-embed even if embedding files already exist",
    )
    parser.add_argument(
        "--text_encoder_fp32",
        action="store_true",
        help="Uses fp32 for the text encoder rather than default bfloat16.",
    )
    args = parser.parse_args()

    folder = args.folder.resolve()
    if not folder.is_dir():
        raise SystemExit(f"Folder does not exist or is not a directory: {folder}")

    device = resolve_device(args.device)
    model = load_model(modelname=args.model, device=device, default_family="TMR", text_encoder_fp32=args.text_encoder_fp32)

    dirs = discover_motion_folders(folder)
    if not dirs:
        raise SystemExit(f"No directories with meta.json and (motion.npz or gt_motion.npz) found under {folder}")
    print(f"Discovered {len(dirs)} motion folders.")

    skipped_motion = 0
    skipped_gt = 0
    skipped_text = 0
    for sample_dir in tqdm(dirs, desc="Embedding"):
        meta_path = sample_dir / "meta.json"
        meta = load_json(meta_path)
        texts, _ = parse_prompts_from_meta(meta)
        if len(texts) != 1:
            raise SystemExit(f"Expected exactly one text per motion; got {len(texts)} in {meta_path}")
        text = texts[0]

        # Embed motion.npz -> motion_embedding.npy
        if (sample_dir / "motion.npz").is_file():
            if not args.overwrite and (sample_dir / "motion_embedding.npy").is_file():
                skipped_motion += 1
            else:
                npz_path = sample_dir / "motion.npz"
                posed_joints = _load_posed_joints(npz_path, device)
                with torch.inference_mode():
                    motion_emb = model.encode_motion(posed_joints, unit_vector=True)
                np.save(sample_dir / "motion_embedding.npy", motion_emb.cpu().numpy())

        # Embed gt_motion.npz -> gt_motion_embedding.npy
        if (sample_dir / "gt_motion.npz").is_file():
            if not args.overwrite and (sample_dir / "gt_motion_embedding.npy").is_file():
                skipped_gt += 1
            else:
                npz_path = sample_dir / "gt_motion.npz"
                posed_joints = _load_posed_joints(npz_path, device)
                with torch.inference_mode():
                    gt_motion_emb = model.encode_motion(posed_joints, unit_vector=True)
                np.save(sample_dir / "gt_motion_embedding.npy", gt_motion_emb.cpu().numpy())

        # Embed text -> text_embedding.npy
        if not args.overwrite and (sample_dir / "text_embedding.npy").is_file():
            skipped_text += 1
        else:
            with torch.inference_mode():
                text_emb = model.encode_raw_text([text], unit_vector=True)
            np.save(sample_dir / "text_embedding.npy", text_emb.cpu().numpy())

    total_skipped = skipped_motion + skipped_gt + skipped_text
    if total_skipped:
        print(f"Embedded {len(dirs)} folders; skipped some existing files (use --overwrite to re-embed).")
    else:
        print(f"Saved motion_embedding.npy, gt_motion_embedding.npy, and text_embedding.npy in {len(dirs)} folders.")


if __name__ == "__main__":
    main()
