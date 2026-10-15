# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""LLM2Vec encoder wrapper for Kimodo text conditioning."""

import os

import numpy as np
import torch

from .llm2vec import LLM2Vec


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

        self.model = LLM2Vec.from_pretrained(
            base_model_name_or_path=base_model_name_or_path,
            peft_model_name_or_path=peft_model_name_or_path,
            torch_dtype=torch_dtype,
            cache_dir=cache_dir,
        )

        env_device = os.environ.get("TEXT_ENCODER_DEVICE")
        if env_device:
            device = env_device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self._device = device
        if device is not None:
            self.model = self.model.to(device)

        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

    def to(self, device: torch.device):
        self.model = self.model.to(device)
        self._device = str(device) if not isinstance(device, str) else device
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
                device=self._device,
            )

        assert len(encoded_text.shape)
        assert self.llm_dim == encoded_text.shape[-1]

        encoded_text = encoded_text[:, None]
        lengths = np.ones(len(encoded_text), dtype=int).tolist()

        if is_string:
            encoded_text = encoded_text[0]
            lengths = lengths[0]

        encoded_text = torch.as_tensor(encoded_text, device=self._device)
        return encoded_text, lengths
