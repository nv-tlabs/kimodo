# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import unittest
from unittest import mock

from kimodo.model.load_model import _build_local_text_encoder_conf


class TextEncoderConfigTest(unittest.TestCase):
    def test_dtype_override_does_not_mutate_preset(self):
        with mock.patch.dict(os.environ, {"TEXT_ENCODER_DTYPE": "float16"}, clear=False):
            self.assertEqual(_build_local_text_encoder_conf()["dtype"], "float16")

        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertEqual(_build_local_text_encoder_conf()["dtype"], "bfloat16")

    def test_fp32_flag_takes_precedence(self):
        with mock.patch.dict(os.environ, {"TEXT_ENCODER_DTYPE": "float16"}, clear=False):
            self.assertEqual(_build_local_text_encoder_conf(text_encoder_fp32=True)["dtype"], "float32")

    def test_model_device_is_forwarded_to_local_encoder(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertEqual(_build_local_text_encoder_conf(device="cpu")["device"], "cpu")

    def test_text_encoder_device_environment_takes_precedence(self):
        with mock.patch.dict(os.environ, {"TEXT_ENCODER_DEVICE": "mps"}, clear=True):
            self.assertEqual(_build_local_text_encoder_conf(device="cpu")["device"], "auto")


if __name__ == "__main__":
    unittest.main()
