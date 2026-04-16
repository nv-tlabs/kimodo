# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""LLM2Vec encoder wrapper for Kimodo text conditioning."""

import os

import numpy as np
import torch

from .llm2vec import LLM2Vec

# KIMODO_QUANTIZE options:
#   "4bit"  - NF4 4-bit quantization (~5GB VRAM for Llama-3-8B)
#   "8bit"  - INT8 8-bit quantization (~9GB VRAM for Llama-3-8B)
#   unset   - no quantization, full precision (~17GB VRAM)
QUANTIZE_PRESETS = {
    "4bit": {
        "load_in_4bit": True,
        "bnb_4bit_compute_dtype": "float16",
        "bnb_4bit_quant_type": "nf4",
        "bnb_4bit_use_double_quant": True,
    },
    "8bit": {
        "load_in_8bit": True,
    },
}


def _build_quantization_config():
    """Build BitsAndBytes quantization config from KIMODO_QUANTIZE env var."""
    quantize = os.environ.get("KIMODO_QUANTIZE", "").lower()
    if not quantize:
        return None
    if quantize not in QUANTIZE_PRESETS:
        available = ", ".join(sorted(QUANTIZE_PRESETS))
        raise ValueError(
            f"Unknown KIMODO_QUANTIZE='{quantize}'. Available: {available}"
        )
    from transformers import BitsAndBytesConfig
    kwargs = QUANTIZE_PRESETS[quantize].copy()
    if "bnb_4bit_compute_dtype" in kwargs:
        kwargs["bnb_4bit_compute_dtype"] = getattr(torch, kwargs["bnb_4bit_compute_dtype"])
    return BitsAndBytesConfig(**kwargs)


class LLM2VecEncoder:
    """LLM2Vec text embeddings."""

    def __init__(
        self,
        base_model_name_or_path: str,
        peft_model_name_or_path: str,
        dtype: str,
        llm_dim: int,
        device: str = "auto",
    ) -> None:
        torch_dtype = getattr(torch, dtype)
        self.llm_dim = llm_dim

        cache_dir = os.environ.get("HUGGINGFACE_CACHE_DIR")

        if "TEXT_ENCODERS_DIR" in os.environ:
            base_model_name_or_path = os.path.join(os.environ["TEXT_ENCODERS_DIR"], base_model_name_or_path)
            peft_model_name_or_path = os.path.join(os.environ["TEXT_ENCODERS_DIR"], peft_model_name_or_path)

        extra_kwargs = {}
        quantization_config = _build_quantization_config()
        if quantization_config is not None:
            extra_kwargs["quantization_config"] = quantization_config
            extra_kwargs["device_map"] = "auto"
            mode = os.environ.get("KIMODO_QUANTIZE", "").lower()
            print(f"[Kimodo] Using {mode} quantization for text encoder to reduce VRAM usage")

        self.model = LLM2Vec.from_pretrained(
            base_model_name_or_path=base_model_name_or_path,
            peft_model_name_or_path=peft_model_name_or_path,
            torch_dtype=torch_dtype,
            cache_dir=cache_dir,
            **extra_kwargs,
        )
        if device is not None:
            if device == "auto":
                device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model = self.model.to(device)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False
        self._quantized = quantization_config is not None

    def to(self, device: torch.device):
        if not self._quantized:
            self.model = self.model.to(device)
        return self

    def eval(self):
        self.model.eval()
        return self

    def get_device(self):
        return self.model.model.device

    def __call__(self, text: list[str] | str):
        is_string = False
        if isinstance(text, str):
            text = [text]
            is_string = True

        with torch.no_grad():
            encoded_text = self.model.encode(
                text,
                # IMPORTANT: different batch sizes unexpectedly change the output embeddings, so we always set it to 1
                #            here for repeatability no matter how many texts are being encoded. This
                #            is a fundamental issue with transformers, and is especially bad at lower
                #            precisions (https://github.com/huggingface/transformers/issues/25420#issuecomment-1775317535)
                #            note: this is an internal batch size used by llm2vec - the text list can still be of arbitrary length.
                batch_size=1,
                show_progress_bar=False,
            )

        assert len(encoded_text.shape)
        assert self.llm_dim == encoded_text.shape[-1]

        encoded_text = encoded_text[:, None]
        lengths = np.ones(len(encoded_text), dtype=int).tolist()

        if is_string:
            encoded_text = encoded_text[0]
            lengths = lengths[0]

        encoded_text = torch.tensor(encoded_text).to(self.get_device())
        return encoded_text, lengths
