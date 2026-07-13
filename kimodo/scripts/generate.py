# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import os
import shutil
from typing import Any, Dict, Optional

import torch

from kimodo import DEFAULT_MODEL, load_model
from kimodo.constraints import load_constraints_lst
from kimodo.device_utils import resolve_torch_device
from kimodo.exports.motion_io import save_kimodo_npz
from kimodo.meta import load_prompts_from_meta
from kimodo.model.cfg import CFG_TYPES
from kimodo.model.registry import get_model_info
from kimodo.tools import load_json, save_json, seed_everything


def parse_args():
    parser = argparse.ArgumentParser(description="Cmd line API for generation motions with kimodo")
    parser.add_argument(
        "prompt",
        nargs="?",
        type=str,
        default=None,
        help="Text prompt describing the motion to generate, or several prompts separated by periods.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help="Name of the model (e.g. Kimodo-SOMA-RP-v1, etc).",
    )
    parser.add_argument(
        "--duration",
        type=str,
        default="5.0",
        help="Duration in seconds (default: 5.0). Separate by spaces in a string for different durations per prompts",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=1,
        help="Number of samples to generate (default: 1)",
    )
    parser.add_argument(
        "--diffusion_steps",
        type=int,
        default=100,
        help="Number of diffusion steps (default: 100)",
    )
    parser.add_argument(
        "--num_transition_frames",
        type=int,
        default=5,
        help="Number of frames to help transitioning (default: 5)",
    )
    parser.add_argument(
        "--constraints",
        type=str,
        default=None,
        help="Saved constraint list",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Torch device to use: auto, cpu, cuda, cuda:0, mps, or a specific backend string.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output",
        help="Output stem name: with one sample writes a single file per format (e.g. test.npz, test.csv); with multiple samples creates a folder and writes test_00.npz, test_01.npz, ... inside it. Used for NPZ, AMASS NPZ, CSV, and BVH.",
    )
    parser.add_argument(
        "--save_example_dir",
        action="store_true",
        help=(
            "Save demo-compatible example directories (each contains motion.npz, constraints.json, meta.json). "
            "With one sample, writes <output>_example/. With multiple samples, writes "
            "<output>_examples/<output>_example_00/, <output>_example_01/, ..."
        ),
    )
    parser.add_argument(
        "--bvh",
        action="store_true",
        help="Also export BVH (SOMA models only); uses the same stem as --output.",
    )
    parser.add_argument(
        "--bvh_standard_tpose",
        action="store_true",
        help="If exporting BVH, export with the rest pose being the standard T-pose rather than the rest pose consistent with the BONES-SEED dataset.",
    )
    parser.add_argument(
        "--no-postprocess",
        action="store_true",
        help="Don't apply motion post-processing to reduce foot skating (ignored for G1)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Seed for reproducible results",
    )
    parser.add_argument(
        "--input_folder",
        type=str,
        default=None,
        help="Folder containing meta.json and optional constraints.json. If set, generation settings are loaded from meta.json.",
    )
    parser.add_argument(
        "--cfg_type",
        type=str,
        default=argparse.SUPPRESS,
        choices=CFG_TYPES,
        help=(
            "Classifier-free guidance mode: nocfg (no CFG), regular (single scale on cond vs uncond), "
            "or separated (custom: separate text and constraint scales). "
            "Use with --cfg_weight as required by the mode."
        ),
    )
    parser.add_argument(
        "--cfg_weight",
        type=float,
        nargs="*",
        default=argparse.SUPPRESS,
        help=(
            "CFG scale(s): one float for regular, or two floats [text_weight, constraint_weight] for separated. "
            "Omit with --cfg_type nocfg. If omitted, two floats alone imply separated; one float alone implies regular."
        ),
    )
    return parser.parse_args()


def get_texts_and_num_frames_from_prompt(prompt: str, duration: str, fps: float):
    # Get the texts
    texts = [text.strip() for text in prompt.split(".")]
    texts = [text + "." for text in texts if text]

    nb_prompts = len(texts)

    # Get the durations
    if " " not in duration:
        duration_sec = float(duration)
        # same for all the prompts
        num_frames = [int(duration_sec * fps)] * nb_prompts
    else:
        durations = duration.split(" ")
        assert len(durations) == len(texts), "The number of durations should match the number of prompts"
        num_frames = [int(float(duration.strip()) * fps) for duration in durations]
        assert len(num_frames) == nb_prompts, "The number of durations should be 1 or match the number of texts"

    return texts, num_frames


def _single_file_path(path: str, ext: str) -> str:
    """Return path for a single output file (no folder).

    Adds ext if missing; creates parent dirs if any.
    """
    if not path.endswith(ext):
        path = path.rstrip(os.sep) + ext
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    return path


def _output_dir_and_path(path: str, default_base: str, ext: str):
    """Create output folder from path and return (dir_path, path_for_file_with_suffix, base_name).

    If path has an extension, folder name is the path stem; else the path is the folder name.
    base_name is the folder basename for _00, _01, ... when n_samples > 1.
    """
    folder = os.path.splitext(path)[0] if os.path.splitext(path)[1] else path
    os.makedirs(folder, exist_ok=True)
    base_name = os.path.basename(folder.rstrip(os.sep))
    return folder, os.path.join(folder, default_base + ext), base_name


def resolve_cfg_kwargs(args: argparse.Namespace, meta: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Resolve cfg_type / cfg_weight for model(...).

    Precedence: explicit CLI (--cfg_type / --cfg_weight) overrides meta.json ``cfg``;
    if neither applies, returns {} so the model uses its own defaults.
    """
    ns = vars(args)
    has_type = "cfg_type" in ns
    has_wflag = "cfg_weight" in ns
    cli_type = ns.get("cfg_type")
    cli_w = ns.get("cfg_weight")

    if has_wflag:
        if cli_w is None or len(cli_w) == 0:
            raise ValueError("--cfg_weight requires one float (regular) or two floats (separated).")

    if has_type and cli_type == "nocfg":
        if has_wflag:
            raise ValueError("--cfg_weight is not used with --cfg_type nocfg.")
        return {"cfg_type": "nocfg"}

    if has_type or has_wflag:
        if has_type:
            eff_type = cli_type
            if has_wflag:
                if eff_type == "regular" and len(cli_w) != 1:
                    raise ValueError("--cfg_type regular requires exactly one --cfg_weight value.")
                if eff_type == "separated" and len(cli_w) != 2:
                    raise ValueError("--cfg_type separated requires exactly two --cfg_weight values.")
            else:
                if eff_type == "regular":
                    raise ValueError("--cfg_type regular requires --cfg_weight with one float.")
                if eff_type == "separated":
                    raise ValueError("--cfg_type separated requires --cfg_weight with two floats.")
        else:
            if len(cli_w) == 1:
                eff_type = "regular"
            elif len(cli_w) == 2:
                eff_type = "separated"
            else:
                raise ValueError("--cfg_weight expects 1 float (regular) or 2 floats (separated).")

        if eff_type == "regular":
            return {"cfg_type": "regular", "cfg_weight": float(cli_w[0])}
        return {"cfg_type": "separated", "cfg_weight": [float(cli_w[0]), float(cli_w[1])]}

    if meta and isinstance(meta.get("cfg"), dict):
        cfg = meta["cfg"]
        enabled = cfg.get("enabled", True)
        if not enabled:
            return {"cfg_type": "nocfg"}
        return {
            "cfg_type": "separated",
            "cfg_weight": [
                float(cfg.get("text_weight", 2.0)),
                float(cfg.get("constraint_weight", 2.0)),
            ],
        }

    return {}


def get_generation_inputs(args, fps: float):
    """Get texts/num_frames and parameter overrides from either CLI or input_folder."""
    if args.input_folder is None:
        if not args.prompt:
            raise ValueError("Either provide 'prompt' or '--input_folder'.")
        texts, num_frames = get_texts_and_num_frames_from_prompt(args.prompt, args.duration, fps)
        return {
            "texts": texts,
            "num_frames": num_frames,
            "num_samples": args.num_samples,
            "diffusion_steps": args.diffusion_steps,
            "seed": args.seed,
            "constraints_path": args.constraints,
            "meta": None,
        }

    meta_path = os.path.join(args.input_folder, "meta.json")
    meta = load_json(meta_path)
    texts, durations_sec = load_prompts_from_meta(meta_path)
    num_frames = [int(float(duration) * fps) for duration in durations_sec]

    constraints_path = args.constraints
    default_constraints_path = os.path.join(args.input_folder, "constraints.json")
    if constraints_path is None and os.path.exists(default_constraints_path):
        constraints_path = default_constraints_path

    return {
        "texts": texts,
        "num_frames": num_frames,
        "num_samples": meta.get("num_samples", args.num_samples),
        "diffusion_steps": meta.get("diffusion_steps", args.diffusion_steps),
        "seed": meta.get("seed", args.seed),
        "constraints_path": constraints_path,
        "meta": meta,
    }


def main():
    args = parse_args()
    device = resolve_torch_device(args.device)
    print(f"Using device: {device}")

    # Load model (resolution of name done inside load_model)
    model, resolved_model = load_model(
        args.model,
        device=device,
        default_family="Kimodo",
        return_resolved_name=True,
    )
    info = get_model_info(resolved_model)
    display = info.display_name if info else resolved_model
    print(f"Loaded model: {display} ({resolved_model})")

    # Get generation inputs
    generation_inputs = get_generation_inputs(args, model.fps)
    texts = generation_inputs["texts"]
    num_frames = generation_inputs["num_frames"]
    print("Will generate motions with the following prompts")
    for text, num_frame in zip(texts, num_frames):
        print(f"    '{text}' with {num_frame} frames")

    # Load constraints
    constraints_path = generation_inputs["constraints_path"]
    if constraints_path:
        constraint_lst = load_constraints_lst(constraints_path, model.skeleton)
    else:
        constraint_lst = []

    if constraint_lst:
        print(f"Using {len(constraint_lst)} set of constraints")
        for constraint in constraint_lst:
            print(f"    {constraint}")

    if generation_inputs["seed"] is not None:
        seed_everything(generation_inputs["seed"])

    cfg_kwargs = resolve_cfg_kwargs(args, generation_inputs.get("meta"))
    if cfg_kwargs:
        ct = cfg_kwargs.get("cfg_type")
        cw = cfg_kwargs.get("cfg_weight")
        if cw is not None:
            print(f"Using CFG: cfg_type={ct!r}, cfg_weight={cw!r}")
        else:
            print(f"Using CFG: cfg_type={ct!r}")

    # G1: postprocessing is disabled (does not work well for this model).
    use_postprocess = False if "g1" in resolved_model else (not args.no_postprocess)
    output = model(
        texts,
        num_frames,
        constraint_lst=constraint_lst,
        num_denoising_steps=generation_inputs["diffusion_steps"],
        num_samples=generation_inputs["num_samples"],
        multi_prompt=True,
        num_transition_frames=args.num_transition_frames,
        post_processing=use_postprocess,
        return_numpy=True,
        **cfg_kwargs,
    )

    n_samples = int(output["posed_joints"].shape[0])
    # Parse the output stem once; all formats (NPZ, AMASS NPZ, CSV, BVH) use this base name.
    output_base = args.output

    # Save the NPZ output
    if n_samples == 1:
        npz_path = _single_file_path(output_base, ".npz")
        print(f"Saving the npz output to {npz_path}")
        single = {
            k: (v[0] if hasattr(v, "shape") and len(v.shape) > 0 and v.shape[0] == n_samples else v)
            for k, v in output.items()
        }
        save_kimodo_npz(npz_path, single)
    else:
        out_dir, _, base_name = _output_dir_and_path(output_base, "motion", ".npz")
        print(f"Saving the npz output to {out_dir}/ ({base_name}_00.npz ...)")
        for i in range(n_samples):
            single = {
                k: (v[i] if hasattr(v, "shape") and len(v.shape) > 0 and v.shape[0] == n_samples else v)
                for k, v in output.items()
            }
            save_kimodo_npz(os.path.join(out_dir, f"{base_name}_{i:02d}.npz"), single)

    # Save the AMASS NPZ output
    if resolved_model == "kimodo-smplx-rp":
        from kimodo.exports.smplx import AMASSConverter

        converter = AMASSConverter(skeleton=model.skeleton, fps=model.fps)
        if n_samples == 1:
            # Use distinct name so AMASS NPZ does not overwrite the main NPZ
            amass_single_path = _single_file_path(output_base + "_amass", ".npz")
            print(f"Saving the amass output to {amass_single_path}")
            converter.convert_save_npz(output, amass_single_path)
        else:
            out_dir, _, base_name = _output_dir_and_path(output_base, "amass", ".npz")
            print(f"Saving the amass output to {out_dir}/ (amass_00.npz ...)")
            converter.convert_save_npz(output, os.path.join(out_dir, "amass.npz"))

    # Save the CSV output
    if resolved_model == "kimodo-g1-rp":
        from kimodo.exports.mujoco import MujocoQposConverter

        converter = MujocoQposConverter(model.skeleton)
        qpos = converter.dict_to_qpos(output, device)
        if n_samples == 1:
            csv_path = _single_file_path(output_base, ".csv")
            print(f"Saving the csv output to {csv_path}")
            converter.save_csv(qpos, csv_path)
        else:
            out_dir, _, base_name = _output_dir_and_path(output_base, "qpos", ".csv")
            print(f"Saving the csv output to {out_dir}/ ({base_name}_00.csv ...)")
            converter.save_csv(qpos, os.path.join(out_dir, base_name + ".csv"))

    # Save the BVH output
    if args.bvh:
        skeleton = model.skeleton
        if "somaskel" not in skeleton.name:
            print("BVH export is only supported for SOMA skeletons. Skipping --bvh.")
        else:
            from kimodo.exports.bvh import save_motion_bvh
            from kimodo.skeleton import SOMASkeleton30, global_rots_to_local_rots

            if isinstance(skeleton, SOMASkeleton30):
                # Motion has already been converted to somaskel77 within the model for output
                skeleton = skeleton.somaskel77.to(device)

            if n_samples == 1:
                bvh_path = _single_file_path(output_base, ".bvh")
                print(f"Saving the BVH output to {bvh_path}")
                joints_pos = torch.from_numpy(output["posed_joints"][0]).to(device)
                joints_rot = torch.from_numpy(output["global_rot_mats"][0]).to(device)
                local_rot_mats = global_rots_to_local_rots(joints_rot, skeleton)
                root_positions = joints_pos[:, skeleton.root_idx, :]
                save_motion_bvh(
                    bvh_path,
                    local_rot_mats,
                    root_positions,
                    skeleton=skeleton,
                    fps=model.fps,
                    standard_tpose=args.bvh_standard_tpose,
                )
            else:
                out_dir, _, base_name = _output_dir_and_path(output_base, "motion", ".bvh")
                print(f"Saving the BVH output to {out_dir}/ ({base_name}_00.bvh ...)")
                for i in range(n_samples):
                    joints_pos = torch.from_numpy(output["posed_joints"][i]).to(device)
                    joints_rot = torch.from_numpy(output["global_rot_mats"][i]).to(device)
                    local_rot_mats = global_rots_to_local_rots(joints_rot, skeleton)
                    root_positions = joints_pos[:, skeleton.root_idx, :]
                    save_motion_bvh(
                        os.path.join(out_dir, f"{base_name}_{i:02d}.bvh"),
                        local_rot_mats,
                        root_positions,
                        skeleton=skeleton,
                        fps=model.fps,
                        standard_tpose=args.bvh_standard_tpose,
                    )

    # Save the example directory
    if args.save_example_dir:
        output_stem = os.path.splitext(output_base)[0].rstrip(os.sep)
        base_name = os.path.basename(output_stem)

        if n_samples == 1:
            parent_dir = None
            example_dirs = [output_stem + "_example"]
        else:
            parent_dir = output_stem + "_examples"
            if os.path.exists(parent_dir):
                raise FileExistsError(f"Example directory already exists: {parent_dir}")
            os.makedirs(parent_dir)
            example_dirs = [
                os.path.join(parent_dir, f"{base_name}_example_{i:02d}") for i in range(n_samples)
            ]

        durations_sec = [nf / model.fps for nf in num_frames]
        if len(texts) == 1:
            meta_info: dict = {"text": texts[0], "duration": durations_sec[0]}
        else:
            meta_info = {"texts": texts, "durations": durations_sec}
        meta_info["num_samples"] = generation_inputs["num_samples"]
        if generation_inputs["seed"] is not None:
            meta_info["seed"] = generation_inputs["seed"]
        meta_info["diffusion_steps"] = generation_inputs["diffusion_steps"]
        if cfg_kwargs:
            cfg_type = cfg_kwargs.get("cfg_type", "nocfg")
            cfg_weight = cfg_kwargs.get("cfg_weight")
            if cfg_type == "nocfg":
                meta_info["cfg"] = {"enabled": False}
            elif cfg_type == "separated" and isinstance(cfg_weight, list) and len(cfg_weight) == 2:
                meta_info["cfg"] = {
                    "enabled": True,
                    "text_weight": cfg_weight[0],
                    "constraint_weight": cfg_weight[1],
                }
            elif cfg_type == "regular" and cfg_weight is not None:
                meta_info["cfg"] = {
                    "enabled": True,
                    "text_weight": float(cfg_weight),
                    "constraint_weight": float(cfg_weight),
                }

        for i, example_dir in enumerate(example_dirs):
            if os.path.exists(example_dir):
                raise FileExistsError(f"Example directory already exists: {example_dir}")
            os.makedirs(example_dir)
            sample = {
                k: (v[i] if hasattr(v, "shape") and len(v.shape) > 0 and v.shape[0] == n_samples else v)
                for k, v in output.items()
            }
            save_kimodo_npz(os.path.join(example_dir, "motion.npz"), sample)
            if constraints_path:
                shutil.copy2(constraints_path, os.path.join(example_dir, "constraints.json"))
            save_json(os.path.join(example_dir, "meta.json"), meta_info)

        if parent_dir is None:
            print(f"Saved demo example to {example_dirs[0]}")
        else:
            print(f"Saved {n_samples} demo examples to {parent_dir}/")


if __name__ == "__main__":
    main()
