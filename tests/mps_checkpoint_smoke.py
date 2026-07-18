#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Download a Kimodo checkpoint and run a tiny end-to-end MPS generation."""

import argparse

import torch
from torch import nn

from kimodo import load_model
from kimodo.device import resolve_device


class StubTextEncoder(nn.Module):
    """Deterministic embedding source that avoids the gated 8B encoder download."""

    def forward(self, texts: list[str]):
        return torch.zeros(len(texts), 1, 4096, dtype=torch.float32), [1] * len(texts)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="kimodo-soma-rp")
    parser.add_argument("--device", default="mps")
    parser.add_argument("--frames", type=int, default=30)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--postprocess", action="store_true")
    args = parser.parse_args()

    device = resolve_device(args.device)
    torch.manual_seed(7)
    model = load_model(args.model, device=device, text_encoder=StubTextEncoder())
    output = model(
        "A person walks forward.",
        args.frames,
        num_denoising_steps=args.steps,
        cfg_type="nocfg",
        num_samples=1,
        post_processing=args.postprocess,
        progress_bar=lambda values: values,
    )

    required = {"posed_joints", "local_rot_mats", "global_rot_mats", "root_positions"}
    if missing := required.difference(output):
        raise RuntimeError(f"Missing output keys: {sorted(missing)}")
    for key in required:
        value = output[key]
        if not torch.isfinite(value).all():
            raise RuntimeError(f"Non-finite values in {key}")

    print(
        f"checkpoint smoke passed: device={device}, model={args.model}, "
        f"frames={args.frames}, steps={args.steps}, postprocess={args.postprocess}, "
        f"posed_joints={tuple(output['posed_joints'].shape)}"
    )


if __name__ == "__main__":
    main()
