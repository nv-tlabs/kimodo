# SPDX-FileCopyrightText: Copyright (c) 2024 McGill NLP
# SPDX-License-Identifier: MIT
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.


# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import logging
import os
from functools import partial
from typing import Dict, List, Optional, Union

import numpy as np
import torch
import torch.multiprocessing as mp
from peft import PeftModel
from torch import Tensor, device, nn
from tqdm.autonotebook import tqdm, trange
from transformers import (
    AutoConfig,
    AutoModel,
    AutoTokenizer,
    GemmaConfig,
    LlamaConfig,
    MistralConfig,
    PretrainedConfig,
    Qwen2Config,
)

from ...device_utils import resolve_torch_device

logger = logging.getLogger(__name__)


def batch_to_device(batch, target_device: device):
    """Send a pytorch batch to a device (CPU/GPU)"""
    for key in batch:
        if isinstance(batch[key], Tensor):
            batch[key] = batch[key].to(target_device)
    return batch


class LLM2Vec(nn.Module):
    def __init__(
        self,
        model: AutoModel,
        tokenizer: AutoTokenizer,
        pooling_mode: str = "mean",
        max_length: int = 512,
        doc_max_length: int = 400,
        skip_instruction: bool = True,
    ):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.pooling_mode = pooling_mode
        self.skip_instruction = skip_instruction
        self.max_length = max_length
        self.doc_max_length = doc_max_length
        self.config = model.config

    @classmethod
    def _get_model_class(cls, config_class_name, enable_bidirectional):
        if not enable_bidirectional:
            return AutoModel
        if config_class_name == "MistralConfig":
            from .models.bidirectional_mistral import MistralBiModel

            return MistralBiModel
        elif config_class_name == "LlamaConfig":
            from .models.bidirectional_llama import LlamaBiModel

            return LlamaBiModel
        elif config_class_name == "GemmaConfig":
            from .models.bidirectional_gemma import GemmaBiModel

            return GemmaBiModel
        elif config_class_name == "Qwen2Config":
            from .models.bidirectional_qwen2 import Qwen2BiModel

            return Qwen2BiModel
        else:
            raise ValueError(f"{config_class_name} is not supported yet with bidirectional models.")

    @classmethod
    def from_pretrained(
        cls,
        base_model_name_or_path,
        peft_model_name_or_path=None,
        merge_peft=False,
        enable_bidirectional=True,
        **kwargs,
    ):
        # pop out encoder args
        keys = ["pooling_mode", "max_length", "doc_max_length", "skip_instruction"]
        encoder_args = {key: kwargs.pop(key, None) for key in keys if kwargs.get(key) is not None}

        tokenizer = AutoTokenizer.from_pretrained(base_model_name_or_path)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"

        config = AutoConfig.from_pretrained(base_model_name_or_path)
        config_class_name = config.__class__.__name__

        model_class = cls._get_model_class(config_class_name, enable_bidirectional=enable_bidirectional)

        model = model_class.from_pretrained(base_model_name_or_path, **kwargs)

        if os.path.isdir(base_model_name_or_path) and os.path.exists(f"{base_model_name_or_path}/config.json"):
            with open(f"{base_model_name_or_path}/config.json", "r") as fIn:
                config_dict = json.load(fIn)
            config = PretrainedConfig.from_dict(config_dict)
            model.config._name_or_path = config._name_or_path

        # For special case where config.json and adapter weights are in the same directory
        if hasattr(model, "peft_config"):
            model = PeftModel.from_pretrained(
                model,
                base_model_name_or_path,
            )
            model = model.merge_and_unload()

        if peft_model_name_or_path is not None:
            model = PeftModel.from_pretrained(
                model,
                peft_model_name_or_path,
            )
            if merge_peft:
                model = model.merge_and_unload()

        config = {}
        config_addr = peft_model_name_or_path if peft_model_name_or_path is not None else base_model_name_or_path
        if os.path.exists(f"{config_addr}/llm2vec_config.json"):
            with open(f"{config_addr}/llm2vec_config.json", "r") as fIn:
                llm2vec_config = json.load(fIn)
            config.update(llm2vec_config)

        for key, value in encoder_args.items():
            config[key] = value

        return cls(model=model, tokenizer=tokenizer, **config)

    def prepare_for_tokenization(self, text):
        if self.model.config._name_or_path == "meta-llama/Meta-Llama-3-8B-Instruct":
            text = "<|start_header_id|>user<|end_header_id|>\n\n" + text.strip() + "<|eot_id|>"
            return text
        if self.model.config._name_or_path in [
            "mistralai/Mistral-7B-Instruct-v0.2",
            "meta-llama/Llama-2-7b-chat-hf",
        ]:
            text = "[INST] " + text.strip() + " [/INST]"
        if self.model.config._name_or_path in [
            "google/gemma-2-9b-it",
        ]:
            text = "<bos><start_of_turn>user\n" + text.strip() + "<end_of_turn>"
        if self.model.config._name_or_path in [
            "Qwen/Qwen2-1.5B-Instruct",
            "Qwen/Qwen2-7B-Instruct",
        ]:
            text = "<|im_start|>user\n" + text.strip() + "<|im_end|>"
        if self.pooling_mode == "eos_token":
            if self.model.config._name_or_path == "meta-llama/Meta-Llama-3-8B":
                text = text.strip() + "<|end_of_text|>"
            elif isinstance(self.model.config, LlamaConfig) or isinstance(self.model.config, MistralConfig):
                text = text.strip() + " </s>"
            elif isinstance(self.model.config, GemmaConfig):
                text = text.strip() + "<eos>"
            elif isinstance(self.model.config, Qwen2Config):
                text = text.strip() + "<|endoftext|>"
        return text

    def tokenize(self, texts):
        texts_2 = []
        original_texts = []
        for text in texts:
            t = text.split("!@#$%^&*()")
            texts_2.append(t[1] if len(t) > 1 else "")
            original_texts.append("".join(t))

        original = self.tokenizer(
            original_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        )
        embed_mask = None
        for t_i, t in enumerate(texts_2):
            ids = self.tokenizer(
                [t],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_length,
                add_special_tokens=False,
            )
            if embed_mask is None:
                e_m = torch.zeros_like(original["attention_mask"][t_i])
                if len(ids["input_ids"][0]) > 0:
                    e_m[-len(ids["input_ids"][0]) :] = torch.ones(len(ids["input_ids"][0]))
                embed_mask = e_m.unsqueeze(0)
            else:
                e_m = torch.zeros_like(original["attention_mask"][t_i])
                if len(ids["input_ids"][0]) > 0:
                    e_m[-len(ids["input_ids"][0]) :] = torch.ones(len(ids["input_ids"][0]))
                embed_mask = torch.cat((embed_mask, e_m.unsqueeze(0)), dim=0)

        original["embed_mask"] = embed_mask
        return original

    def _skip_instruction(self, sentence_feature):
        assert sentence_feature["attention_mask"].shape == sentence_feature["embed_mask"].shape
        sentence_feature["attention_mask"] = sentence_feature["embed_mask"]

    def forward(self, sentence_feature: Dict[str, Tensor]):
        embed_mask = None
        if "embed_mask" in sentence_feature:
            embed_mask = sentence_feature.pop("embed_mask")
        reps = self.model(**sentence_feature)
        sentence_feature["embed_mask"] = embed_mask

        return self.get_pooling(sentence_feature, reps.last_hidden_state)

    def get_pooling(self, features, last_hidden_states):  # All models padded from left
        assert self.tokenizer.padding_side == "left", "Pooling modes are implemented for padding from left."
        if self.skip_instruction:
            self._skip_instruction(features)
        seq_lengths = features["attention_mask"].sum(dim=-1)
        if self.pooling_mode == "mean":
            return torch.stack(
                [last_hidden_states[i, -length:, :].mean(dim=0) for i, length in enumerate(seq_lengths)],
                dim=0,
            )
        elif self.pooling_mode == "weighted_mean":
            bs, l, _ = last_hidden_states.shape
            complete_weights = torch.zeros(bs, l, device=last_hidden_states.device)
            for i, seq_l in enumerate(seq_lengths):
                if seq_l > 0:
                    complete_weights[i, -seq_l:] = torch.arange(seq_l) + 1
                    complete_weights[i] /= torch.clamp(complete_weights[i].sum(), min=1e-9)
            return torch.sum(last_hidden_states * complete_weights.unsqueeze(-1), dim=1)
        elif self.pooling_mode == "eos_token" or self.pooling_mode == "last_token":
            return last_hidden_states[:, -1]
        elif self.pooling_mode == "bos_token":
            return last_hidden_states[features["input_ids"] == self.tokenizer.bos_token_id]
        else:
            raise ValueError(f"{self.pooling_mode} is not implemented yet.")

    def _convert_to_str(self, instruction, text):
        tokenized_q = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
            add_special_tokens=False,
        )
        tokenized_q_length = len(tokenized_q["input_ids"][0])

        while tokenized_q_length > self.doc_max_length:
            reduction_ratio = self.doc_max_length / tokenized_q_length
            reduced_length = int(len(text.split()) * reduction_ratio)
            text = " ".join(text.split()[:reduced_length])
            tokenized_q = self.tokenizer(
                text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_length,
                add_special_tokens=False,
            )
            tokenized_q_length = len(tokenized_q["input_ids"][0])

        return f"{instruction.strip()} !@#$%^&*(){text}" if instruction else f"!@#$%^&*(){text}"

    def encode(
        self,
        sentences: Union[str, List[str]],
        batch_size: int = 32,
        show_progress_bar: bool = True,
        convert_to_numpy: bool = False,
        convert_to_tensor: bool = False,
        device: Optional[str] = None,
    ):
        """
        Encode a list of sentences to their respective embeddings. The sentences can be a list of strings or a string.
        Args:
            sentences: sentence or sentences to encode.
            batch_size: batch size for turning sentence tokens into embeddings.
            show_progress_bar: whether to show progress bars during encoding steps.
            convert_to_numpy: If true, return numpy arrays instead of torch tensors.
            convert_to_tensor: If true, return torch tensors (default).
            device: torch backend device identifier (e.g., 'cuda', 'cpu', 'mps', etc.). If not specified,
            the default is to prefer cuda when available, then mps, then cpu. Note that only the choice of
            'cuda' supports multiprocessing as currently implemented.

        Returns: embeddings of the sentences. Embeddings are detached and always on the CPU (see _encode implementation).

        """
        if isinstance(sentences[0], str) and isinstance(sentences[-1], int):
            sentences = [sentences]
        # required for MEDI version of MTEB
        if isinstance(sentences[0], str):
            sentences = [[""] + [sentence] for sentence in sentences]

        device = resolve_torch_device(device)

        concatenated_input_texts = []
        for sentence in sentences:
            assert isinstance(sentence[0], str)
            assert isinstance(sentence[1], str)
            concatenated_input_texts.append(self._convert_to_str(sentence[0], sentence[1]))
        sentences = concatenated_input_texts

        self.eval()

        if convert_to_tensor:
            convert_to_numpy = False

        length_sorted_idx = np.argsort([-self._text_length(sen) for sen in sentences])
        sentences_sorted = [sentences[idx] for idx in length_sorted_idx]
        all_embeddings = []

        if not str(device).startswith("cuda") or torch.cuda.device_count() <= 1:
            # This branch also support mps devices
            self.to(device)
            for start_index in trange(
                0,
                len(sentences),
                batch_size,
                desc="Batches",
                disable=not show_progress_bar,
            ):
                sentences_batch = sentences_sorted[start_index : start_index + batch_size]
                embeddings = self._encode(sentences_batch, device=device, convert_to_numpy=convert_to_numpy)
                all_embeddings.append(embeddings)
        else:
            num_proc = torch.cuda.device_count()
            cuda_compatible_multiprocess = mp.get_context("spawn")
            with cuda_compatible_multiprocess.Pool(num_proc) as p:
                sentences_batches = [
                    sentences_sorted[start_index : start_index + batch_size]
                    for start_index in range(0, len(sentences), batch_size)
                ]

                progress_bar = tqdm(
                    total=len(sentences_batches),
                    desc="Batches",
                    disable=not show_progress_bar,
                )
                results = []

                def update(*args):
                    progress_bar.update()

                for batch in sentences_batches:
                    results.append(
                        p.apply_async(
                            self._encode,
                            args=(batch, None, convert_to_numpy, True),
                            callback=update,
                        )
                    )

                all_embeddings = [result.get() for result in results]
                progress_bar.close()

        all_embeddings = torch.cat(all_embeddings, dim=0)
        all_embeddings = all_embeddings[np.argsort(length_sorted_idx)]
        all_embeddings = all_embeddings.to(torch.float32)
        if convert_to_numpy:
            all_embeddings = np.asarray([emb.numpy() for emb in all_embeddings])
        return all_embeddings

    def save(self, output_path, merge_before_save=False, save_config=True):
        if merge_before_save and isinstance(self.model, PeftModel):
            self.model = self.model.merge_and_unload()
            # Fixes the issue of saving - https://huggingface.co/McGill-NLP/LLM2Vec-Mistral-7B-Instruct-v2-mntp-unsup-simcse/discussions/1
            if hasattr(self.model, "_hf_peft_config_loaded"):
                self.model._hf_peft_config_loaded = False

        self.model.save_pretrained(output_path)
        self.tokenizer.save_pretrained(output_path)

        llm2vec_config = {
            "pooling_mode": self.pooling_mode,
            "max_length": self.max_length,
            "doc_max_length": self.doc_max_length,
            "skip_instruction": self.skip_instruction,
        }

        if save_config:
            os.makedirs(output_path, exist_ok=True)
            with open(f"{output_path}/llm2vec_config.json", "w") as fOut:
                json.dump(llm2vec_config, fOut, indent=4)

    def _encode(
        self,
        sentences_batch,
        device: Optional[str] = None,
        convert_to_numpy: bool = False,
        multiprocessing=False,
    ):
        if multiprocessing:
            # multiprocessing only supports CUDA devices at this time, so we ignore the value of device
            # and use cuda:rank for the device
            rank = mp.current_process()._identity[0]
            if device is None and torch.cuda.is_available():
                device = f"cuda:{rank % torch.cuda.device_count()}"

        self.to(device)
        features = self.tokenize([self.prepare_for_tokenization(sentence) for sentence in sentences_batch])
        features = batch_to_device(features, device)

        with torch.no_grad():
            embeddings = self.forward(features)
            embeddings = embeddings.detach()
            embeddings = embeddings.cpu()

        return embeddings

    def _text_length(self, text: Union[List[int], List[List[int]]]):
        """Help function to get the length for the input text.

        Text can be either a string (which means a single text) a list of ints (which means a single
        tokenized text), or a tuple of list of ints (representing several text inputs to the model).
        """
        if (
            isinstance(text, str) or (isinstance(text, list) and isinstance(text[0], int)) or len(text) == 0
        ):  # Single text, list of ints, or empty
            return len(text)
        if isinstance(text, dict):  # {key: value} case
            return len(next(iter(text.values())))
        elif not hasattr(text, "__len__"):  # Object has no len() method
            return 1
        else:
            return sum([len(t) for t in text])

    def resize_token_embeddings(
        self,
        new_num_tokens: Optional[int] = None,
        pad_to_multiple_of: Optional[int] = None,
    ) -> nn.Embedding:
        return self.model.resize_token_embeddings(new_num_tokens=new_num_tokens, pad_to_multiple_of=pad_to_multiple_of)

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=gradient_checkpointing_kwargs)
