# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Runtime device selection shared by Kimodo entry points."""

import os
from typing import Optional

import torch


def mps_is_available() -> bool:
    """Return whether this PyTorch build can use Apple's Metal backend."""
    backend = getattr(torch.backends, "mps", None)
    return bool(backend is not None and backend.is_available())


def get_default_device() -> str:
    """Prefer CUDA, then Apple Metal (MPS), and finally CPU."""
    if torch.cuda.is_available():
        return "cuda:0"
    if mps_is_available():
        return "mps"
    return "cpu"


def resolve_device(device: Optional[str] = None, *, env_var: str = "KIMODO_DEVICE") -> str:
    """Resolve and validate a device name.

    ``None`` and ``"auto"`` consult ``env_var`` before selecting the best
    available accelerator. Explicit CUDA and MPS requests fail early with a
    useful message when the requested backend is unavailable.
    """
    requested = device
    if requested is None or str(requested).strip().lower() == "auto":
        requested = os.environ.get(env_var) or "auto"

    requested = str(requested).strip().lower()
    if requested == "auto":
        return get_default_device()

    try:
        resolved = torch.device(requested)
    except (RuntimeError, ValueError) as error:
        raise ValueError(f"Invalid PyTorch device '{requested}'.") from error

    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but this PyTorch build cannot access CUDA.")
    if resolved.type == "mps" and not mps_is_available():
        backend = getattr(torch.backends, "mps", None)
        if backend is None or not backend.is_built():
            reason = "this PyTorch build has no MPS support"
        else:
            reason = "MPS is not available on this macOS/hardware combination"
        raise RuntimeError(f"MPS was requested, but {reason}.")

    return str(resolved)


__all__ = ["get_default_device", "mps_is_available", "resolve_device"]
