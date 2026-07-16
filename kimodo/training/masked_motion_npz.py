# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Dataset for precomputed Kimodo feature NPZs with explicit condition masks."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional, Sequence, Union

import numpy as np
import torch
from torch.utils.data import Dataset

PathLike = Union[str, Path]


def resolve_masked_sample_path(npz_root: PathLike, sample_path: PathLike) -> Path:
    """Resolve relative entries, stale absolute paths, and samples/<basename> fallbacks."""
    npz_root = Path(npz_root)
    path = Path(sample_path)
    if path.is_file():
        return path

    if not path.is_absolute():
        candidate = npz_root / path
        if candidate.is_file():
            return candidate

    candidate = npz_root / "samples" / path.name
    if candidate.is_file():
        return candidate

    return path


class MaskedMotionNPZDataset(Dataset):
    """Read precomputed Kimodo samples that already include observed_motion/motion_mask."""

    def __init__(
        self,
        npz_root: PathLike,
        *,
        manifest_path: Optional[PathLike] = None,
        text_from: str = "clip_name",
        default_text: str = "",
        npz_paths: Optional[Sequence[PathLike]] = None,
        limit: Optional[int] = None,
        verify_shapes: bool = True,
        **_: Any,
    ) -> None:
        self.npz_root = Path(npz_root)
        self.text_from = str(text_from)
        self.default_text = str(default_text)
        self.verify_shapes = bool(verify_shapes)

        if npz_paths is not None:
            self.samples = [{"sample_path": str(Path(p))} for p in npz_paths]
        else:
            manifest = Path(manifest_path) if manifest_path is not None else self.npz_root / "manifest.jsonl"
            if manifest.is_file():
                self.samples = []
                with manifest.open("r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            self.samples.append(json.loads(line))
            else:
                paths = sorted((self.npz_root / "samples").glob("*.npz"))
                if not paths:
                    paths = sorted(self.npz_root.rglob("*.npz"))
                self.samples = [{"sample_path": str(p)} for p in paths]

        if limit is not None and int(limit) > 0:
            self.samples = self.samples[: int(limit)]
        if not self.samples:
            raise FileNotFoundError(f"No masked motion NPZ samples found under {self.npz_root}")

    def __len__(self) -> int:
        return len(self.samples)

    def _sample_path(self, item: dict[str, Any]) -> Path:
        return resolve_masked_sample_path(self.npz_root, item["sample_path"])

    def _text_for(self, item: dict[str, Any], npz_data: np.lib.npyio.NpzFile) -> str:
        if self.text_from == "none":
            return self.default_text
        if self.text_from in item:
            return str(item[self.text_from])
        if self.text_from in npz_data.files:
            value = npz_data[self.text_from]
            if value.shape == ():
                return str(value.item())
            return str(value)
        if self.text_from == "clip_name":
            if "clip_name" in item:
                return str(item["clip_name"])
            if "clip_name" in npz_data.files:
                return str(npz_data["clip_name"].item())
        return self.default_text

    def __getitem__(self, idx: int) -> dict[str, Any]:
        item = self.samples[int(idx)]
        path = self._sample_path(item)
        with np.load(path, allow_pickle=False) as z:
            required = ("motion", "observed_motion", "motion_mask", "pad_mask", "length")
            missing = [name for name in required if name not in z.files]
            if missing:
                raise ValueError(f"{path} missing required keys: {missing}")

            motion = torch.from_numpy(np.asarray(z["motion"], dtype=np.float32))
            observed_motion = torch.from_numpy(np.asarray(z["observed_motion"], dtype=np.float32))
            motion_mask = torch.from_numpy(np.asarray(z["motion_mask"], dtype=np.bool_))
            pad_mask = torch.from_numpy(np.asarray(z["pad_mask"], dtype=np.bool_))
            length = int(np.asarray(z["length"]).item())
            text = self._text_for(item, z)
            clip_name = str(z["clip_name"].item()) if "clip_name" in z.files and z["clip_name"].shape == () else path.stem

        if self.verify_shapes:
            if motion.ndim != 2:
                raise ValueError(f"{path}: motion must be [T,D], got {tuple(motion.shape)}")
            if observed_motion.shape != motion.shape:
                raise ValueError(
                    f"{path}: observed_motion shape {tuple(observed_motion.shape)} != motion {tuple(motion.shape)}"
                )
            if motion_mask.shape != motion.shape:
                raise ValueError(f"{path}: motion_mask shape {tuple(motion_mask.shape)} != motion {tuple(motion.shape)}")
            if pad_mask.shape != motion.shape[:1]:
                raise ValueError(
                    f"{path}: pad_mask shape {tuple(pad_mask.shape)} incompatible with motion {tuple(motion.shape)}"
                )

        return {
            "motion": motion,
            "length": torch.tensor(length, dtype=torch.long),
            "pad_mask": pad_mask,
            "text": text,
            "csv_path": str(item.get("source_csv", path)),
            "clip_name": clip_name,
            "frame_start": torch.tensor(int(item.get("frame_start", 0)), dtype=torch.long),
            "frame_end": torch.tensor(int(item.get("frame_end", length)), dtype=torch.long),
            "observed_motion": observed_motion,
            "motion_mask": motion_mask,
            "keyframe_indices": None,
            "constraint_mode": "precomputed_mask",
            "constraints": None,
        }
