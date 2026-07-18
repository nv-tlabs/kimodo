# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Step (4) of evaluation pipeline.

This script recursively computes metrics for generated and ground-truth motions within a test suite folder tree. 
Saves metrics json files per test case and per group of test cases in the folder tree.
"""

import argparse
import json
from itertools import groupby
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm import tqdm

from kimodo.constraints import load_constraints_lst
from kimodo.device import resolve_device
from kimodo.meta import parse_prompts_from_meta
from kimodo.metrics import (
    ContraintFollow,
    FootContactConsistency,
    FootSkateFromContacts,
    FootSkateFromHeight,
    FootSkateRatio,
    TMR_EmbeddingMetric,
    aggregate_metrics,
    clear_metrics,
    compute_metrics,
    compute_tmr_per_sample_retrieval,
)
from kimodo.skeleton import build_skeleton
from kimodo.skeleton.definitions import SOMASkeleton30
from kimodo.tools import load_json, to_torch

DEFAULT_FPS = 30.0


def discover_motion_folders(root: Path) -> list[tuple[Path, Path]]:
    root = root.resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Folder does not exist: {root}")
    out: list[tuple[Path, Path]] = []
    for meta_path in root.rglob("meta.json"):
        sample_dir = meta_path.parent
        if (sample_dir / "motion.npz").is_file() and (sample_dir / "gt_motion.npz").is_file():
            rel = sample_dir.relative_to(root)
            out.append((sample_dir, rel))
    return sorted(out, key=lambda x: str(x[1]))


def group_by_parent(examples: list[tuple[Path, Path]]) -> list[list[tuple[Path, Path]]]:
    def parent_key(item: tuple[Path, Path]) -> Path:
        return item[1].parent if len(item[1].parts) > 1 else Path(".")

    sorted_examples = sorted(examples, key=parent_key)
    groups: list[list[tuple[Path, Path]]] = []
    for _key, group in groupby(sorted_examples, key=parent_key):
        groups.append(list(group))
    return groups


def _to_scalar(t: torch.Tensor) -> float:
    return float(t.mean().item()) if t.numel() > 0 else float(t.item())


def _to_p95(t: torch.Tensor) -> float:
    if t.numel() == 0:
        return float("nan")
    return float(torch.nanquantile(t, torch.tensor(0.95, device=t.device), dim=0).item())


def _per_sample_metrics_from_saved(metrics_list: list, n: int) -> list[dict[str, float]]:
    per_sample: list[dict[str, float]] = [{} for _ in range(n)]
    for metric in metrics_list:
        for key, lst in metric.saved_metrics.items():
            for i, t in enumerate(lst):
                if i >= n:
                    break
                per_sample[i][key] = _to_scalar(t)
    return per_sample


def _load_pair_embeddings(
    sample_dir: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None] | None:
    motion_emb_path = sample_dir / "motion_embedding.npy"
    text_emb_path = sample_dir / "text_embedding.npy"
    gt_motion_emb_path = sample_dir / "gt_motion_embedding.npy"
    if not (motion_emb_path.is_file() and text_emb_path.is_file()):
        return None

    motion_emb = np.load(motion_emb_path)
    text_emb = np.load(text_emb_path)
    if motion_emb.ndim == 3 and motion_emb.shape[0] == 1:
        motion_emb = motion_emb[0]
    if text_emb.ndim == 3 and text_emb.shape[0] == 1:
        text_emb = text_emb[0]

    gt_motion_emb = None
    if gt_motion_emb_path.is_file():
        gt_motion_emb = np.load(gt_motion_emb_path)
        if gt_motion_emb.ndim == 3 and gt_motion_emb.shape[0] == 1:
            gt_motion_emb = gt_motion_emb[0]

    return motion_emb, text_emb, gt_motion_emb


def _load_npz_motion(
    npz_path: Path,
    device: str,
    soma30_skel: SOMASkeleton30 | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Load posed_joints and foot_contacts from an NPZ, upscaling SOMA30 to SOMA77 if needed."""
    data = np.load(npz_path)
    posed_joints = to_torch(data["posed_joints"], device=device)
    foot_contacts = to_torch(data["foot_contacts"], device=device)

    if posed_joints.shape[-2] == 30 and soma30_skel is not None:
        local_rot_mats = to_torch(data["local_rot_mats"], device=device)
        root_positions = to_torch(data["root_positions"], device=device)
        out77 = soma30_skel.output_to_SOMASkeleton77(
            {"local_rot_mats": local_rot_mats, "root_positions": root_positions, "foot_contacts": foot_contacts}
        )
        posed_joints = out77["posed_joints"]
        foot_contacts = out77["foot_contacts"]

    return posed_joints, foot_contacts


def _run_eval_on_group(
    group: list[tuple[Path, Path]],
    skeleton: torch.nn.Module,
    metrics_list: list,
    device: str,
    group_name: str = "",
    soma30_skel: SOMASkeleton30 | None = None,
) -> tuple[
    list[dict[str, float]],
    list[dict[str, float]],
    dict[str, float],
    dict[str, float],
    dict[str, float],
    list[dict[str, Any]],
]:
    """Run two passes: gen (motion.npz + embeddings) and GT (gt_motion.npz only). Return
    per_sample_gen, per_sample_gt, aggregated_gen, aggregated_gt, tmr_metrics, tmr_per_sample.
    """
    n = len(group)
    sample_ids: list[str] = []
    texts: list[str] = []
    motion_embs: list[np.ndarray] = []
    text_embs: list[np.ndarray] = []

    # ----- Pass 1: generation (motion.npz + all embeddings) -----
    clear_metrics(metrics_list)
    desc = f"Samples ({group_name})" if group_name else "Samples"
    for sample_dir, rel_path in tqdm(group, desc=desc, unit="motion"):
        stem = rel_path.name
        sample_ids.append(stem)
        meta_path = sample_dir / "meta.json"
        meta = load_json(meta_path)
        texts_parsed, _ = parse_prompts_from_meta(meta)
        texts.append(texts_parsed[0] if texts_parsed else "")

        posed_joints, foot_contacts = _load_npz_motion(sample_dir / "motion.npz", device, soma30_skel)
        nframes = posed_joints.shape[0]
        lengths = torch.tensor(nframes, dtype=torch.long, device=device)
        constraints_path = sample_dir / "constraints.json"
        constraints_lst = (
            load_constraints_lst(str(constraints_path), skeleton=skeleton) if constraints_path.is_file() else []
        )
        metrics_in: dict[str, Any] = {
            "posed_joints": posed_joints,
            "foot_contacts": foot_contacts,
            "lengths": lengths,
            "constraints_lst": constraints_lst,
        }
        text_this = texts_parsed[0] if texts_parsed else ""
        embs = _load_pair_embeddings(sample_dir)
        if (text_this or "").strip() and embs is not None:
            motion_emb, text_emb, gt_motion_emb = embs
            metrics_in["motion_emb"] = motion_emb
            metrics_in["text_emb"] = text_emb
            if gt_motion_emb is not None:
                metrics_in["gt_motion_emb"] = gt_motion_emb
            motion_embs.append(motion_emb)
            text_embs.append(text_emb)

        compute_metrics(metrics_list, metrics_in)

    per_sample_gen = _per_sample_metrics_from_saved(metrics_list, n)
    raw_aggregated_gen = aggregate_metrics(metrics_list)
    aggregated_gen = {}
    tmr_metrics: dict[str, float] = {}
    has_text = len(motion_embs) == n and len(text_embs) == n
    for key, v in raw_aggregated_gen.items():
        val = _to_scalar(v)
        if key.startswith("TMR/"):
            if has_text:
                tmr_metrics[key] = val
        else:
            aggregated_gen[key] = val
    if "constraint_root2d_err" in raw_aggregated_gen:
        aggregated_gen["constraint_root2d_err_p95"] = _to_p95(raw_aggregated_gen["constraint_root2d_err"])

    tmr_per_sample: list[dict[str, Any]] = []
    if has_text and motion_embs and text_embs and len(motion_embs) == n and len(text_embs) == n:
        motion_emb_stack = np.stack(motion_embs, axis=0)
        text_emb_stack = np.stack(text_embs, axis=0)
        tmr_per_sample = compute_tmr_per_sample_retrieval(motion_emb_stack, text_emb_stack, sample_ids, texts, top_k=5)

    # ----- Pass 2: GT (gt_motion.npz only, no embeddings) -----
    clear_metrics(metrics_list)
    for sample_dir, rel_path in tqdm(group, desc=f"GT ({group_name})" if group_name else "GT", unit="motion"):
        posed_joints, foot_contacts = _load_npz_motion(sample_dir / "gt_motion.npz", device, soma30_skel)
        nframes = posed_joints.shape[0]
        lengths = torch.tensor(nframes, dtype=torch.long, device=device)
        constraints_path = sample_dir / "constraints.json"
        constraints_lst = (
            load_constraints_lst(str(constraints_path), skeleton=skeleton) if constraints_path.is_file() else []
        )
        metrics_in = {
            "posed_joints": posed_joints,
            "foot_contacts": foot_contacts,
            "lengths": lengths,
            "constraints_lst": constraints_lst,
        }
        compute_metrics(metrics_list, metrics_in)

    per_sample_gt = _per_sample_metrics_from_saved(metrics_list, n)
    raw_aggregated_gt = aggregate_metrics(metrics_list)
    aggregated_gt = {}
    for key, v in raw_aggregated_gt.items():
        if key.startswith("TMR/"):
            continue
        aggregated_gt[key] = _to_scalar(v)
    if "constraint_root2d_err" in raw_aggregated_gt:
        aggregated_gt["constraint_root2d_err_p95"] = _to_p95(raw_aggregated_gt["constraint_root2d_err"])

    return (
        per_sample_gen,
        per_sample_gt,
        aggregated_gen,
        aggregated_gt,
        tmr_metrics,
        tmr_per_sample,
    )


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(
        description="Recursively evaluate generated motions; write metrics.json per folder and <name>.json per parent.",
    )
    parser.add_argument(
        "folder",
        type=Path,
        help="Root folder to search recursively for meta.json + motion.npz + gt_motion.npz",
    )
    parser.add_argument("--device", default=None, help="auto/mps/cuda/cpu. Default: auto")
    args = parser.parse_args()

    folder = args.folder.resolve()
    if not folder.is_dir():
        raise SystemExit(f"Folder does not exist: {folder}")

    device = resolve_device(args.device)

    examples = discover_motion_folders(folder)
    if not examples:
        raise SystemExit(f"No directories with meta.json, motion.npz, and gt_motion.npz found under {folder}")
    print(f"Discovered {len(examples)} motion folders.")

    first_posed = np.load(examples[0][0] / "motion.npz")["posed_joints"]
    num_joints = first_posed.shape[-2]

    # SOMA models could generate 30-joint output; upscale to 77 for evaluation
    soma30_skel: SOMASkeleton30 | None = None
    if num_joints == 30:
        soma30_skel = SOMASkeleton30().to(device)
        _ = soma30_skel.somaskel77  # trigger lazy init
        soma30_skel.somaskel77.to(device)
        skeleton = soma30_skel.somaskel77
        print("Detected SOMA30 motions; will upscale to SOMA77 for evaluation.")
    else:
        skeleton = build_skeleton(num_joints).to(device)

    fps = DEFAULT_FPS
    kwargs = {"skeleton": skeleton, "fps": fps}
    metrics_list = [
        FootSkateFromHeight(**kwargs),
        FootSkateFromContacts(**kwargs),
        FootContactConsistency(**kwargs),
        FootSkateRatio(**kwargs),
        ContraintFollow(**kwargs),
        TMR_EmbeddingMetric(**kwargs),
    ]

    groups = group_by_parent(examples)
    for group in tqdm(groups, desc="Evaluating folders"):
        sample_dirs = [g[0] for g in group]
        folder_for_group = sample_dirs[0].parent
        folder_name = folder_for_group.name

        (
            per_sample_gen,
            per_sample_gt,
            aggregated_gen,
            aggregated_gt,
            tmr_metrics,
            tmr_per_sample,
        ) = _run_eval_on_group(group, skeleton, metrics_list, device, group_name=folder_name, soma30_skel=soma30_skel)

        texts = []
        for sample_dir, _ in group:
            meta = load_json(sample_dir / "meta.json")
            texts_parsed, _ = parse_prompts_from_meta(meta)
            texts.append(texts_parsed[0] if texts_parsed else "")

        for i, (sample_dir, _) in enumerate(group):
            metrics_path = sample_dir / "metrics.json"
            out = {
                "num_motions": 1,
                "folder": str(sample_dir),
                "per_motion_mean_gen": per_sample_gen[i] if i < len(per_sample_gen) else {},
                "per_motion_mean_gt": per_sample_gt[i] if i < len(per_sample_gt) else {},
            }
            if i < len(tmr_per_sample):
                out["tmr"] = {
                    "t2m_rank": tmr_per_sample[i]["rank"],
                    "text": texts[i] if i < len(texts) else "",
                    "top5_retrieved": tmr_per_sample[i]["top_k"],
                }
            _write_json(metrics_path, out)

        parent_json_path = folder_for_group.parent / f"{folder_name}.json"
        full_metrics = {
            "num_motions": len(group),
            "folder": str(folder_for_group),
            "per_motion_mean_gen": aggregated_gen,
            "per_motion_mean_gt": aggregated_gt,
        }
        if tmr_metrics:
            full_metrics["tmr"] = tmr_metrics
        _write_json(parent_json_path, full_metrics)

    print(f"Wrote metrics.json in each of {len(examples)} folders and folder-level JSONs for {len(groups)} groups.")


if __name__ == "__main__":
    main()
