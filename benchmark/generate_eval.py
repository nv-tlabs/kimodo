# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Step (2) of evaluation pipeline.

This script recursively generates motions using Kimodo from a test suite folder tree.
"""

import argparse
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from kimodo.constraints import load_constraints_lst
from kimodo.device import resolve_device
from kimodo.meta import parse_prompts_from_meta
from kimodo.model import DEFAULT_MODEL, load_model
from kimodo.tools import load_json, seed_everything


def parse_args():
    parser = argparse.ArgumentParser(description="Recursively generate motions from a testsuite folder tree")
    parser.add_argument(
        "--benchmark",
        type=str,
        default="testsuite",
        help="Root folder containing subfolders with meta.json (default: testsuite)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output root; directory hierarchy is mirrored here. If omitted, motions are generated in-place inside the testsuite folder.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for generating motions (default: 32)",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="DataLoader workers for loading meta/constraints paths (default: 4)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help="Name of the model (e.g. Kimodo-SOMA-RP-v1.1, kimodo-soma-rp, or SOMA).",
    )
    parser.add_argument(
        "--diffusion_steps",
        type=int,
        default=100,
        help="Number of diffusion steps (default: 100); overridden by meta.json if present",
    )
    parser.add_argument(
        "--postprocess",
        action="store_true",
        help="Apply motion post-processing to reduce foot skating",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Regenerate outputs even if motion.npz already exists",
    )
    parser.add_argument(
        "--text_encoder_fp32",
        action="store_true",
        help="Uses fp32 for instantiating the text encoder (if API is not already running) rather than default bfloat16.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="PyTorch device: auto, mps, cuda, or cpu (default: auto).",
    )
    return parser.parse_args()


def discover_example_folders(root: Path) -> list[tuple[Path, Path]]:
    """Discover leaf directories that contain meta.json.

    Returns list of (src_dir, rel_path).
    """
    root = root.resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Testsuite folder does not exist: {root}")
    out: list[tuple[Path, Path]] = []
    for meta_path in root.rglob("meta.json"):
        src_dir = meta_path.parent
        rel = src_dir.relative_to(root)
        out.append((src_dir, rel))
    return sorted(out, key=lambda x: str(x[1]))


def copy_source_files(src_dir: Path, out_dir: Path) -> None:
    """Copy meta.json, constraints.json, and gt_motion.npz (if present) from src_dir to out_dir."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in ("meta.json", "constraints.json", "gt_motion.npz"):
        src_file = src_dir / name
        if src_file.is_file():
            shutil.copy2(src_file, out_dir / name)


class EvalExampleDataset(Dataset):
    """Dataset of example folders: yields text, num_frame, constraints_path (and paths, meta).
    No torch/skeleton in workers so num_workers > 0 is safe with CUDA.
    """

    def __init__(
        self,
        examples: list[tuple[Path, Path]],
        testsuite_root: Path,
        generated_root: Path,
        fps: float,
    ):
        self.examples = examples
        self.testsuite_root = testsuite_root
        self.generated_root = generated_root
        self.fps = fps

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        src_dir, rel_path = self.examples[idx]
        out_dir = self.generated_root / rel_path
        meta_path = src_dir / "meta.json"
        meta = load_json(str(meta_path))
        assert meta.get("num_samples", 1) == 1, "Expected num_samples to be absent or 1 in meta.json"
        texts, durations_sec = parse_prompts_from_meta(meta)
        assert len(texts) == 1, "Expected exactly one prompt (len(texts)==1) per example"
        num_frames = [int(float(d) * self.fps) for d in durations_sec]
        assert len(num_frames) == 1, "Expected exactly one duration per example"
        constraints_path = src_dir / "constraints.json"
        cpath = str(constraints_path) if constraints_path.is_file() else None
        return {
            "rel_path": rel_path,
            "src_dir": str(src_dir),
            "out_dir": str(out_dir),
            "meta": meta,
            "text": texts[0],
            "num_frame": num_frames[0],
            "constraints_path": cpath,
        }


def collate_examples(batch: list[dict]) -> dict[str, Any]:
    """Collate list of example dicts; keep list fields as lists (no stacking)."""
    if not batch:
        return {}
    keys = batch[0].keys()
    out: dict[str, Any] = {}
    for k in keys:
        vals = [b[k] for b in batch]
        out[k] = vals
    return out


def group_by_parent(
    examples: list[tuple[Path, Path]],
) -> list[list[tuple[Path, Path]]]:
    """Group (src_dir, rel_path) by parent directory of rel_path for folder-by-folder processing."""
    from itertools import groupby

    def parent_key(item: tuple[Path, Path]) -> Path:
        rel = item[1]
        return rel.parent if len(rel.parts) > 1 else Path(".")

    sorted_examples = sorted(examples, key=parent_key)
    groups: list[list[tuple[Path, Path]]] = []
    for _key, group in groupby(sorted_examples, key=parent_key):
        groups.append(list(group))
    return groups


def _slice_output_at(output: dict[str, Any], index: int) -> dict[str, Any]:
    """Slice a (possibly nested) output dict at batch index for one sample."""
    out: dict[str, Any] = {}
    for k, v in output.items():
        if isinstance(v, dict):
            out[k] = _slice_output_at(v, index)
        elif isinstance(v, np.ndarray) and v.ndim > 0:
            out[k] = v[index]
        else:
            out[k] = v
    return out


def _crop_output(output: dict[str, Any], num_frames: int) -> dict[str, Any]:
    """Crop a single-sample output dict along the time dimension (axis 0)."""
    out: dict[str, Any] = {}
    for k, v in output.items():
        if isinstance(v, dict):
            out[k] = _crop_output(v, num_frames)
        elif isinstance(v, np.ndarray) and v.ndim >= 1:
            out[k] = v[:num_frames]
        else:
            out[k] = v
    return out


def main():
    args = parse_args()
    device = resolve_device(args.device)
    print(f"Using device: {device}")
    testsuite_root = Path(args.benchmark).resolve()
    if args.output is not None:
        generated_root = Path(args.output).resolve()
    else:
        generated_root = testsuite_root
    in_place = generated_root == testsuite_root

    examples = discover_example_folders(testsuite_root)
    if not examples:
        raise SystemExit(f"No folders with meta.json found under {testsuite_root}")
    print(f"Discovered {len(examples)} example folders.")

    model, resolved_name = load_model(
        args.model,
        device=device,
        default_family="Kimodo",
        return_resolved_name=True,
        text_encoder_fp32=args.text_encoder_fp32,
    )
    # v1.1 models are meant to be used for benchmark evaluation
    _deprecated_for_benchmark = {
        "kimodo-soma-rp-v1": "Kimodo-SOMA-RP-v1 was not trained to be compatible with the benchmark evaluation.",
        "kimodo-soma-seed-v1": "Kimodo-SOMA-SEED-v1 is not the latest model for benchmark evaluation.",
    }
    if resolved_name in _deprecated_for_benchmark:
        import warnings

        warnings.warn(
            f"Model '{args.model}' resolved to {resolved_name}: "
            f"{_deprecated_for_benchmark[resolved_name]} Consider using v1.1.",
            stacklevel=1,
        )
    print(f"Generating with model: {resolved_name}")
    fps = model.fps
    default_diffusion_steps = args.diffusion_steps

    groups = group_by_parent(examples)
    total_generated = 0
    total_skipped = 0

    total_examples = len(examples)
    for group in groups:
        rel_path_0 = group[0][1]
        if rel_path_0.parent != Path("."):
            folder_label = str(rel_path_0.parent)
        else:
            # Direct children of testsuite root: show root name (e.g. inbetweening)
            folder_label = testsuite_root.name
        num_in_folder = len(group)
        print(f"Generating folder: {folder_label} ({num_in_folder} motions)")

        dataset = EvalExampleDataset(
            group,
            testsuite_root,
            generated_root,
            fps=fps,
        )
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=collate_examples,
        )

        folder_generated = 0
        folder_skipped = 0
        for batch_idx, batch in enumerate(loader):
            rel_paths = batch["rel_path"]
            src_dirs = batch["src_dir"]
            out_dirs = batch["out_dir"]
            metas = batch["meta"]
            batch_texts = batch["text"]
            batch_num_frames = batch["num_frame"]
            constraints_paths = batch["constraints_path"]

            # Filter out samples that are already generated (unless --overwrite).
            if args.overwrite:
                selected_indices = list(range(len(rel_paths)))
            else:
                selected_indices = []
                for i, out_dir_str in enumerate(out_dirs):
                    motion_path = Path(out_dir_str) / "motion.npz"
                    if motion_path.is_file():
                        folder_skipped += 1
                        total_skipped += 1
                        continue
                    selected_indices.append(i)

            if not selected_indices:
                print(
                    f"\r  Generated {folder_generated} / {num_in_folder} (skipped: {folder_skipped}) "
                    f"(total: {total_generated + total_skipped} / {total_examples})",
                    end="",
                    flush=True,
                )
                continue

            rel_paths = [rel_paths[i] for i in selected_indices]
            src_dirs = [src_dirs[i] for i in selected_indices]
            out_dirs = [out_dirs[i] for i in selected_indices]
            metas = [metas[i] for i in selected_indices]
            batch_texts = [batch_texts[i] for i in selected_indices]
            batch_num_frames = [batch_num_frames[i] for i in selected_indices]
            constraints_paths = [constraints_paths[i] for i in selected_indices]

            # Load constraints in main process on model device (no torch in workers)
            device_t = torch.device(device)
            batch_constraints_lst = [
                load_constraints_lst(cpath, model.skeleton, device=device_t) if cpath else []
                for cpath in constraints_paths
            ]

            if not in_place:
                for i in range(len(rel_paths)):
                    copy_source_files(Path(src_dirs[i]), Path(out_dirs[i]))

            # Use first example's diffusion_steps and seed for the whole batch
            diffusion_steps = metas[0].get("diffusion_steps", default_diffusion_steps)
            seed = metas[0].get("seed", None)
            if seed is not None:
                seed_everything(seed)
            else:
                print("Warning: No seed found in meta.json, not seeding this batch.")

            # Single model call for the entire batch (count in bar title, bar clears when done)
            bar_desc = (
                f"  Generated {folder_generated} / {num_in_folder} "
                f"(skipped: {folder_skipped}) (total: {total_generated + total_skipped} / {total_examples})"
            )
            output = model(
                batch_texts,
                batch_num_frames,
                constraint_lst=batch_constraints_lst,
                num_denoising_steps=diffusion_steps,
                multi_prompt=False,
                post_processing=args.postprocess,
                return_numpy=True,
                progress_bar=lambda x: tqdm(x, leave=False, desc=bar_desc),
            )

            # Save each sample to its output dir
            B = len(batch_texts)
            for b in range(B):
                out_dir = Path(out_dirs[b])
                sample_output = _slice_output_at(output, b)
                sample_output = _crop_output(sample_output, batch_num_frames[b])
                motion_path = out_dir / "motion.npz"
                np.savez(motion_path, **sample_output)
                total_generated += 1
                folder_generated += 1

            print(
                f"\r  Generated {folder_generated} / {num_in_folder} (skipped: {folder_skipped}) "
                f"(total: {total_generated + total_skipped} / {total_examples})",
                end="",
                flush=True,
            )

        print()
        print(
            f"  Finished folder {folder_label} ({num_in_folder} motions, "
            f"generated: {folder_generated}, skipped: {folder_skipped})."
        )

    if in_place:
        print(f"Generated {total_generated} motions in-place under {testsuite_root}.")
    else:
        print(f"Generated {total_generated} motions under {generated_root}.")


if __name__ == "__main__":
    main()
