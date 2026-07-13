# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Generic Hugging Face embedding-model wrapper for Kimodo text conditioning."""

from __future__ import annotations

import os

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer


class AutoEmbeddingEncoder:
    """Single-vector embedding wrapper compatible with Kimodo text conditioning."""

    def __init__(
        self,
        model_name_or_path: str,
        dtype: str,
        llm_dim: int,
        query_prefix: str = "",
        normalize: bool = True,
    ) -> None:
        torch_dtype = getattr(torch, dtype)
        self.llm_dim = llm_dim
        self.query_prefix = query_prefix or ""
        self.normalize = normalize

        cache_dir = os.environ.get("HUGGINGFACE_CACHE_DIR")
        model_path = model_name_or_path
        if "TEXT_ENCODERS_DIR" in os.environ:
            model_path = os.path.join(os.environ["TEXT_ENCODERS_DIR"], model_name_or_path)

        self.tokenizer = AutoTokenizer.from_pretrained(model_path, cache_dir=cache_dir)
        if self.tokenizer.pad_token is None and self.tokenizer.eos_token is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModel.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            cache_dir=cache_dir,
        )
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad = False

    def to(self, device: torch.device):
        self.model = self.model.to(device)
        return self

    def eval(self):
        self.model.eval()
        return self

    def get_device(self):
        return next(self.model.parameters()).device

    def _prepare_texts(self, texts: list[str]) -> list[str]:
        if not self.query_prefix:
            return texts
        return [self.query_prefix + text for text in texts]

    def __call__(self, text: list[str] | str):
        is_string = False
        if isinstance(text, str):
            text = [text]
            is_string = True

        text = self._prepare_texts(text)
        device = self.get_device()
        tokenized = self.tokenizer(
            text,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        tokenized = {key: value.to(device) for key, value in tokenized.items()}

        with torch.no_grad():
            outputs = self.model(**tokenized)
            last_hidden_state = outputs.last_hidden_state
            eos_positions = tokenized["attention_mask"].sum(dim=1) - 1
            embeddings = last_hidden_state[torch.arange(len(text), device=device), eos_positions]
            if self.normalize:
                embeddings = F.normalize(embeddings, p=2, dim=1)

        assert embeddings.shape[-1] == self.llm_dim

        embeddings = embeddings[:, None]
        lengths = np.ones(len(embeddings), dtype=int).tolist()

        if is_string:
            embeddings = embeddings[0]
            lengths = lengths[0]

        embeddings = torch.as_tensor(embeddings, device=device)
        return embeddings, lengths
