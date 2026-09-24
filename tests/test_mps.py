# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import unittest

import torch

from kimodo.device import mps_is_available
from kimodo.skeleton import G1Skeleton34, SMPLXSkeleton22, SOMASkeleton30, SOMASkeleton77


class MPSCompatibilityTest(unittest.TestCase):
    def test_skeleton_assets_use_accelerator_safe_dtypes(self):
        for skeleton_type in (G1Skeleton34, SMPLXSkeleton22, SOMASkeleton30, SOMASkeleton77):
            with self.subTest(skeleton=skeleton_type.__name__):
                skeleton = skeleton_type()
                floating_dtypes = {tensor.dtype for _, tensor in skeleton.named_buffers() if tensor.is_floating_point()}
                self.assertNotIn(torch.float64, floating_dtypes)

    @unittest.skipUnless(mps_is_available(), "MPS is not available")
    def test_skeletons_move_to_mps(self):
        for skeleton_type in (G1Skeleton34, SMPLXSkeleton22, SOMASkeleton30, SOMASkeleton77):
            with self.subTest(skeleton=skeleton_type.__name__):
                skeleton = skeleton_type().to("mps")
                self.assertTrue(all(tensor.device.type == "mps" for _, tensor in skeleton.named_buffers()))


if __name__ == "__main__":
    unittest.main()
