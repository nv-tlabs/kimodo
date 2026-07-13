from __future__ import annotations

from typing import Optional

import torch


def torch_cuda_available(torch_mod=torch) -> bool:
    cuda_mod = getattr(torch_mod, "cuda", None)
    is_available = getattr(cuda_mod, "is_available", None)
    return bool(is_available()) if callable(is_available) else False


def torch_mps_available(torch_mod=torch) -> bool:
    backends = getattr(torch_mod, "backends", None)
    mps_backend = getattr(backends, "mps", None)
    if mps_backend is None:
        return False
    is_built = getattr(mps_backend, "is_built", None)
    is_available = getattr(mps_backend, "is_available", None)
    built_ok = bool(is_built()) if callable(is_built) else True
    avail_ok = bool(is_available()) if callable(is_available) else False
    return built_ok and avail_ok


def resolve_torch_device(device: Optional[str] = None, torch_mod=torch) -> str:
    requested = str(device or "auto").strip().lower()
    if requested == "auto":
        if torch_cuda_available(torch_mod):
            return "cuda"
        if torch_mps_available(torch_mod):
            return "mps"
        return "cpu"
    if requested == "cpu":
        return "cpu"
    if requested == "cuda":
        return "cuda" if torch_cuda_available(torch_mod) else "cpu"
    if requested.startswith("cuda:"):
        return requested if torch_cuda_available(torch_mod) else "cpu"
    if requested == "mps":
        return "mps" if torch_mps_available(torch_mod) else "cpu"
    if requested.isdigit():
        return f"cuda:{requested}" if torch_cuda_available(torch_mod) else "cpu"
    if "," in requested:
        return resolve_torch_device(requested.split(",", 1)[0], torch_mod=torch_mod)
    return requested


def preferred_text_encoder_dtype(device: Optional[str], override: Optional[str] = None, torch_mod=torch) -> str:
    if override:
        return override
    resolved_device = resolve_torch_device(device, torch_mod=torch_mod)
    if resolved_device == "mps":
        return "float16"
    return "bfloat16"
