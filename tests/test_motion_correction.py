# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import base64
import io
import unittest
from pathlib import Path

import numpy as np

from tests.motion_correction_fixture import generate_fixture

REFERENCE_PATH = Path(__file__).parent / "data" / "motion_correction_x86_64.npz.b64"


class MotionCorrectionRegressionTest(unittest.TestCase):
    def test_fixed_fixture_matches_x86_reference(self):
        hips, rotations = generate_fixture()

        self.assertEqual(hips.shape, (1, 48, 3))
        self.assertEqual(rotations.shape, (1, 48, 12, 4))
        self.assertTrue(np.isfinite(hips).all())
        self.assertTrue(np.isfinite(rotations).all())

        # Golden arrays were produced by the original x86_64 AVX build under Rosetta.
        payload = base64.b64decode(REFERENCE_PATH.read_text(encoding="ascii"))
        with np.load(io.BytesIO(payload)) as reference:
            np.testing.assert_allclose(hips, reference["hips"], rtol=2e-5, atol=2e-5)
            np.testing.assert_allclose(rotations, reference["rotations"], rtol=2e-5, atol=2e-5)


if __name__ == "__main__":
    unittest.main()
