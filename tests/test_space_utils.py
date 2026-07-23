import unittest

from space_utils import MotionRequest, estimate_zero_gpu_duration, validate_motion_request


class MotionRequestTests(unittest.TestCase):
    def test_normalizes_valid_request(self):
        result = validate_motion_request("  walk   forward  ", 5, 42, 50, True)

        self.assertEqual(
            result,
            MotionRequest(
                prompt="walk forward",
                duration_seconds=5.0,
                seed=42,
                diffusion_steps=50,
                standard_tpose=True,
            ),
        )

    def test_rejects_invalid_values(self):
        invalid_requests = [
            ("", 5, 42, 50, True),
            ("walk", 0, 42, 50, True),
            ("walk", 11, 42, 50, True),
            ("walk", 5, -1, 50, True),
            ("walk", 5, 42, 101, True),
        ]

        for values in invalid_requests:
            with self.subTest(values=values), self.assertRaises(ValueError):
                validate_motion_request(*values)

    def test_duration_estimate_is_bounded_and_increases_with_work(self):
        short = estimate_zero_gpu_duration("walk", 1, 1, 10, True)
        long = estimate_zero_gpu_duration("walk", 10, 1, 100, True)

        self.assertGreaterEqual(short, 60)
        self.assertGreater(long, short)
        self.assertLessEqual(long, 180)


if __name__ == "__main__":
    unittest.main()
