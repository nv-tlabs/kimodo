"""Minimal Kimodo text-to-motion API for Hugging Face ZeroGPU."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("TEXT_ENCODER_MODE", "local")
os.environ.setdefault("TEXT_ENCODER_DEVICE", "cuda")

# ZeroGPU must patch CUDA before torch or any Kimodo module imports it.
import spaces
import gradio as gr
import torch

from kimodo import load_model
from kimodo.exports.bvh import save_motion_bvh
from kimodo.exports.motion_io import save_kimodo_npz
from kimodo.tools import seed_everything
from space_utils import estimate_zero_gpu_duration, validate_motion_request


MODEL_NAME = os.environ.get("KIMODO_MODEL", "Kimodo-SOMA-RP-v1.1")

# Eager module-scope placement is required by ZeroGPU's weight packing mechanism.
MODEL, RESOLVED_MODEL_NAME = load_model(
    MODEL_NAME,
    device="cuda",
    default_family="Kimodo",
    return_resolved_name=True,
)


def _without_progress(iterable):
    return iterable


def _single_sample(output: dict) -> dict:
    """Remove the leading batch dimension from a one-sample model result."""

    sample = {}
    for key, value in output.items():
        if hasattr(value, "shape") and len(value.shape) > 0 and int(value.shape[0]) == 1:
            sample[key] = value[0]
        else:
            sample[key] = value
    return sample


@spaces.GPU(duration=estimate_zero_gpu_duration, size="large")
def generate_motion(
    prompt: str,
    duration_seconds: float,
    seed: int,
    diffusion_steps: int,
    standard_tpose: bool,
) -> tuple[str, str, dict]:
    """Generate one text-conditioned human motion and return BVH, NPZ, and metadata."""

    request = validate_motion_request(
        prompt,
        duration_seconds,
        seed,
        diffusion_steps,
        standard_tpose,
    )
    seed_everything(request.seed)

    num_frames = max(1, int(round(request.duration_seconds * float(MODEL.fps))))
    output = MODEL(
        request.prompt,
        num_frames,
        num_denoising_steps=request.diffusion_steps,
        multi_prompt=False,
        constraint_lst=[],
        cfg_weight=[2.0, 2.0],
        num_samples=1,
        return_numpy=True,
        post_processing=False,
        progress_bar=_without_progress,
    )
    sample = _single_sample(output)

    output_dir = Path(tempfile.mkdtemp(prefix="kimodo-motion-"))
    bvh_path = output_dir / "motion.bvh"
    npz_path = output_dir / "motion.npz"
    save_kimodo_npz(str(npz_path), sample)

    skeleton = MODEL.output_skeleton
    local_rot_mats = torch.as_tensor(sample["local_rot_mats"], device="cuda")
    posed_joints = torch.as_tensor(sample["posed_joints"], device="cuda")
    root_positions = posed_joints[:, int(skeleton.root_idx), :]
    save_motion_bvh(
        bvh_path,
        local_rot_mats,
        root_positions,
        skeleton=skeleton,
        fps=float(MODEL.fps),
        standard_tpose=request.standard_tpose,
    )

    metadata = {
        "prompt": request.prompt,
        "duration_seconds": request.duration_seconds,
        "seed": request.seed,
        "diffusion_steps": request.diffusion_steps,
        "model": MODEL_NAME,
        "resolved_model": RESOLVED_MODEL_NAME,
        "fps": float(MODEL.fps),
        "frames": num_frames,
        "skeleton": str(skeleton.name),
        "standard_tpose": request.standard_tpose,
        "post_processing": False,
    }
    return str(bvh_path), str(npz_path), metadata


with gr.Blocks(title="Kimodo Motion API") as demo:
    gr.Markdown(
        """
        # Kimodo Motion API

        Generate a text-conditioned SOMA motion on ZeroGPU. Download the BVH for Blender
        or keep the NPZ for a later Kimodo workflow. The first call after an idle period may
        take longer while ZeroGPU restores model weights.
        """
    )
    prompt_input = gr.Textbox(
        label="Motion prompt",
        lines=3,
        max_length=1_000,
        value="A person walks forward cautiously, looks over the left shoulder, then stops.",
    )
    with gr.Row():
        duration_input = gr.Slider(1.0, 10.0, value=5.0, step=0.5, label="Duration (seconds)")
        seed_input = gr.Number(value=42, precision=0, minimum=0, maximum=2**31 - 1, label="Seed")
        steps_input = gr.Slider(10, 100, value=50, step=5, label="Diffusion steps")
    standard_tpose_input = gr.Checkbox(
        value=True,
        label="Export a standard T-pose rest skeleton",
        info="Recommended for Blender retargeting.",
    )
    generate_button = gr.Button("Generate motion", variant="primary")
    with gr.Row():
        bvh_output = gr.File(label="Blender BVH")
        npz_output = gr.File(label="Kimodo NPZ")
    metadata_output = gr.JSON(label="Generation metadata")

    generate_button.click(
        fn=generate_motion,
        inputs=[prompt_input, duration_input, seed_input, steps_input, standard_tpose_input],
        outputs=[bvh_output, npz_output, metadata_output],
        api_name="generate_motion",
        api_description="Generate a Kimodo motion as BVH and NPZ files.",
        concurrency_limit=1,
    )
    gr.Examples(
        examples=[
            ["A person takes three slow steps forward and waves with the right hand."],
            ["A person crouches, jumps upward, lands, and regains balance."],
            ["A person performs a short defensive boxing combination."],
        ],
        inputs=[prompt_input],
        cache_examples=False,
    )

demo.queue(default_concurrency_limit=1)

if __name__ == "__main__":
    demo.launch(mcp_server=True)
