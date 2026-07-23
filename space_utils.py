"""Pure helpers for the lightweight Hugging Face Space API."""

from __future__ import annotations

import math
from dataclasses import dataclass


MAX_PROMPT_CHARS = 1_000
MIN_DURATION_SECONDS = 1.0
MAX_DURATION_SECONDS = 10.0
MIN_DIFFUSION_STEPS = 10
MAX_DIFFUSION_STEPS = 100
MAX_SEED = 2**31 - 1


@dataclass(frozen=True)
class MotionRequest:
    """Validated inputs for a single Kimodo generation request."""

    prompt: str
    duration_seconds: float
    seed: int
    diffusion_steps: int
    standard_tpose: bool


def validate_motion_request(
    prompt: str,
    duration_seconds: float,
    seed: int,
    diffusion_steps: int,
    standard_tpose: bool,
) -> MotionRequest:
    """Validate and normalize user-controlled request values."""

    normalized_prompt = " ".join(str(prompt).split())
    if not normalized_prompt:
        raise ValueError("Prompt must not be empty.")
    if len(normalized_prompt) > MAX_PROMPT_CHARS:
        raise ValueError(f"Prompt must contain at most {MAX_PROMPT_CHARS} characters.")

    duration = float(duration_seconds)
    if not math.isfinite(duration) or not MIN_DURATION_SECONDS <= duration <= MAX_DURATION_SECONDS:
        raise ValueError(
            f"Duration must be between {MIN_DURATION_SECONDS:g} and {MAX_DURATION_SECONDS:g} seconds."
        )

    normalized_seed = int(seed)
    if not 0 <= normalized_seed <= MAX_SEED:
        raise ValueError(f"Seed must be between 0 and {MAX_SEED}.")

    steps = int(diffusion_steps)
    if not MIN_DIFFUSION_STEPS <= steps <= MAX_DIFFUSION_STEPS:
        raise ValueError(f"Diffusion steps must be between {MIN_DIFFUSION_STEPS} and {MAX_DIFFUSION_STEPS}.")

    return MotionRequest(
        prompt=normalized_prompt,
        duration_seconds=duration,
        seed=normalized_seed,
        diffusion_steps=steps,
        standard_tpose=bool(standard_tpose),
    )


def estimate_zero_gpu_duration(
    prompt: str,
    duration_seconds: float,
    seed: int,
    diffusion_steps: int,
    standard_tpose: bool,
    *args,
    **kwargs,
) -> int:
    """Estimate the scheduler reservation for one request.

    This is intentionally conservative for the first live deployment. Recalibrate it from
    measured Space timings after representative cold and warm requests.
    """

    del prompt, seed, standard_tpose, args, kwargs
    duration = min(MAX_DURATION_SECONDS, max(MIN_DURATION_SECONDS, float(duration_seconds)))
    steps = min(MAX_DIFFUSION_STEPS, max(MIN_DIFFUSION_STEPS, int(diffusion_steps)))
    return min(180, max(60, int(math.ceil(45 + duration * steps * 0.08))))
