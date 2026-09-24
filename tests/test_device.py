# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import unittest
from unittest import mock

from kimodo.device import get_default_device, resolve_device


class DeviceSelectionTest(unittest.TestCase):
    def test_auto_prefers_cuda(self):
        with mock.patch("kimodo.device.torch.cuda.is_available", return_value=True):
            self.assertEqual(get_default_device(), "cuda:0")

    def test_auto_uses_mps_before_cpu(self):
        with (
            mock.patch("kimodo.device.torch.cuda.is_available", return_value=False),
            mock.patch("kimodo.device.mps_is_available", return_value=True),
        ):
            self.assertEqual(get_default_device(), "mps")

    def test_auto_falls_back_to_cpu(self):
        with (
            mock.patch("kimodo.device.torch.cuda.is_available", return_value=False),
            mock.patch("kimodo.device.mps_is_available", return_value=False),
        ):
            self.assertEqual(get_default_device(), "cpu")

    def test_environment_override(self):
        with mock.patch.dict(os.environ, {"KIMODO_DEVICE": "cpu"}, clear=False):
            self.assertEqual(resolve_device("auto"), "cpu")

    def test_unavailable_mps_fails_early(self):
        with mock.patch("kimodo.device.mps_is_available", return_value=False):
            with self.assertRaisesRegex(RuntimeError, "MPS was requested"):
                resolve_device("mps")


if __name__ == "__main__":
    unittest.main()
