# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: I001
import argparse

from kimodo.device import resolve_device
from kimodo.model import DEFAULT_MODEL
from kimodo.model.registry import resolve_model_name

from .app import Demo


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the kimodo demo UI.")
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help="Default model to load (e.g. Kimodo-SOMA-RP-v1, kimodo-soma-rp, or SOMA).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="PyTorch device: auto, mps, cuda, or cpu (default: auto).",
    )
    args = parser.parse_args()

    resolved = resolve_model_name(args.model, "Kimodo")
    demo = Demo(default_model_name=resolved, device=resolve_device(args.device))
    demo.run()


if __name__ == "__main__":
    main()
