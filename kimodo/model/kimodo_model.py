# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kimodo model: denoiser, text encoder, diffusion sampling, and post-processing."""

import logging
from typing import Dict, List, Optional, Tuple, Union

import torch
from torch import nn
from tqdm.auto import tqdm

from kimodo.constraints import EndEffectorConstraintSet, FullBodyConstraintSet
from kimodo.motion_rep.feature_utils import compute_heading_angle, length_to_mask
from kimodo.postprocess import post_process_motion
from kimodo.sanitize import sanitize_texts
from kimodo.skeleton import SOMASkeleton30
from kimodo.tools import to_numpy

from .cfg import ClassifierFreeGuidedModel
from .diffusion import DDIMSampler, Diffusion

log = logging.getLogger(__name__)


def _normalize_mps_floating_tensors(module: nn.Module) -> None:
    """MPS cannot materialize float64 tensors; coerce stray fp64 buffers/params."""

    for child in module.modules():
        for name, buffer in list(child._buffers.items()):
            if isinstance(buffer, torch.Tensor) and buffer.is_floating_point() and buffer.dtype == torch.float64:
                child._buffers[name] = buffer.to(dtype=torch.float32)
        for name, parameter in list(child._parameters.items()):
            if isinstance(parameter, nn.Parameter) and parameter.is_floating_point() and parameter.dtype == torch.float64:
                child._parameters[name] = nn.Parameter(
                    parameter.to(dtype=torch.float32),
                    requires_grad=parameter.requires_grad,
                )


def _move_module_to_mps(module: nn.Module, device: Union[str, torch.device]) -> None:
    """Move a module to MPS while preserving existing dtypes except fp64 -> fp32."""

    target_device = torch.device(device)

    def _convert(tensor: torch.Tensor) -> torch.Tensor:
        if not isinstance(tensor, torch.Tensor):
            return tensor
        if tensor.is_floating_point():
            target_dtype = torch.float32 if tensor.dtype == torch.float64 else tensor.dtype
            return tensor.to(device=target_device, dtype=target_dtype)
        return tensor.to(device=target_device)

    module._apply(_convert)


class Kimodo(nn.Module):
    """Helper class for test time."""

    def __init__(
        self,
        denoiser: nn.Module,
        text_encoder: nn.Module,
        num_base_steps: int,
        device: Optional[Union[str, torch.device]] = None,
        cfg_type: Optional[str] = "separated",
    ):
        super().__init__()

        self.denoiser = denoiser.eval()

        if cfg_type is None:
            cfg_type = "nocfg"

        # Add Classifier-free guidance to the model if needed
        self.denoiser = ClassifierFreeGuidedModel(self.denoiser, cfg_type=cfg_type)

        self.motion_rep = denoiser.motion_rep
        self.skeleton = self.motion_rep.skeleton

        self.fps = denoiser.motion_rep.fps

        self.diffusion = Diffusion(num_base_steps=num_base_steps)
        self.sampler = DDIMSampler(self.diffusion)
        self.text_encoder = text_encoder

        self.device = device
        # for classifier-free guidance

        if str(device).lower().startswith("mps"):
            _normalize_mps_floating_tensors(self)
            _move_module_to_mps(self, device)
        else:
            self.to(device)

    @property
    def output_skeleton(self):
        """Skeleton used for model output (somaskel77 for SOMA, else unchanged)."""
        if isinstance(self.skeleton, SOMASkeleton30):
            return self.skeleton.somaskel77
        return self.skeleton

    def train(self, mode: bool):
        self.denoiser.train(mode)
        return self

    def eval(self):
        self.denoiser.eval()
        return self

    def denoising_step(
        self,
        motion: torch.Tensor,
        pad_mask: torch.Tensor,
        text_feat: torch.Tensor,
        text_pad_mask: torch.Tensor,
        t: torch.Tensor,
        first_heading_angle: Optional[torch.Tensor],
        motion_mask: torch.Tensor,
        observed_motion: torch.Tensor,
        num_denoising_steps: torch.Tensor,
        cfg_weight: Union[float, Tuple[float, float]],
        guide_masks: Optional[Dict] = None,
        cfg_type: Optional[str] = None,
    ) -> torch.Tensor:
        """Single denoising step.

        Returns:
            torch.Tensor: [B, T, D] noisy motion input to t-1
        """
        # subsample timesteps
        #   NOTE: do this at every step due to ONNX export, i.e. num_samp_stepsmay change dynamically when
        #       running onnx version so need to account for that.
        num_denoising_steps = num_denoising_steps[0]
        use_timesteps, map_tensor = self.diffusion.space_timesteps(num_denoising_steps)
        self.diffusion.calc_diffusion_vars(use_timesteps)

        # first compute initial clean prediction from denoiser
        t_map = map_tensor[t]

        with torch.inference_mode():
            pred_clean = self.denoiser(
                cfg_weight,
                motion,
                pad_mask,
                text_feat,
                text_pad_mask,
                t_map,
                first_heading_angle,
                motion_mask,
                observed_motion,
                cfg_type=cfg_type,
            )

        # sampler computes next step noisy motion
        x_tm1 = self.sampler(use_timesteps, motion, pred_clean, t)
        return x_tm1

    def _multiprompt(
        self,
        prompts: list[str],
        num_frames: int | list[int],
        num_denoising_steps: int,
        constraint_lst: Optional[list] = [],
        cfg_weight: Optional[float] = [2.0, 2.0],
        num_samples: Optional[int] = None,
        cfg_type: Optional[str] = None,
        return_numpy: bool = False,
        first_heading_angle: Optional[torch.Tensor] = None,
        # for transitioning
        num_transition_frames: int = 5,
        # for postprocess
        post_processing: bool = False,
        root_margin: float = 0.04,
        # progress bar
        progress_bar=tqdm,
    ) -> torch.Tensor:
        device = self.device

        bs = num_samples
        texts = sanitize_texts(prompts)

        if isinstance(num_frames, int):
            # same duration for all the segments
            num_frames = [num_frames for _ in range(num_samples)]

        tosqueeze = False
        if num_samples is None:
            num_samples = 1
            tosqueeze = True

        if constraint_lst is None:
            constraint_lst = []

        # Generate one chunck at a time
        current_frame = 0
        generated_motions = []

        for idx, (text, num_frame) in enumerate(zip(texts, num_frames)):
            texts_bs = [text for _ in range(num_samples)]

            lengths = torch.tensor(
                [num_frame for _ in range(num_samples)],
                device=device,
            )

            is_first_motion = not generated_motions

            observed_motion, motion_mask = None, None

            # filter the constraint_lst to only keep the relevent ones
            constraint_lst_base = [
                constraint.crop_move(current_frame, current_frame + num_frame) for constraint in constraint_lst
            ]  # this move temporally but not spatially

            observed_motion, motion_mask = self.motion_rep.create_conditions_from_constraints_batched(
                constraint_lst_base,
                lengths,
                to_normalize=False,  # don't normalize yet, it needs to be moved around
                device=device,
            )

            if not is_first_motion:
                nb_transition_frames = num_transition_frames

                if nb_transition_frames < 1:
                    raise ValueError(f"num_transition_frames must be at least 1, got {nb_transition_frames}")

                latest_motions = generated_motions.pop()
                # remove the transition part of A (will be put back afterward)
                generated_motions.append(latest_motions[:, :-nb_transition_frames])
                latest_frames = latest_motions[:, -nb_transition_frames:]

                last_output = self.motion_rep.inverse(
                    latest_frames,
                    is_normalized=False,
                    return_numpy=False,
                )
                smooth_root_2d = last_output["smooth_root_pos"][..., [0, 2]]

                # add constraints at the begining to allow natural transitions
                constraint_lst_transition = []
                for batch_id in range(bs):
                    new_constraint = FullBodyConstraintSet(
                        self.skeleton,
                        torch.arange(num_transition_frames),
                        last_output["posed_joints"][batch_id, :num_transition_frames],
                        last_output["global_rot_mats"][batch_id, :num_transition_frames],
                        smooth_root_2d[batch_id, :num_transition_frames],
                    )
                    # separate end-effector constraint to capture hand/feet rotations
                    new_ee_constraint = EndEffectorConstraintSet(
                        self.skeleton,
                        torch.arange(num_transition_frames),
                        last_output["posed_joints"][batch_id, :num_transition_frames],
                        last_output["global_rot_mats"][batch_id, :num_transition_frames],
                        smooth_root_2d[batch_id, :num_transition_frames],
                        joint_names=["LeftHand", "RightHand", "LeftFoot", "RightFoot"],
                    )

                    constraint_lst_transition.append([new_constraint, new_ee_constraint])

                transition_lengths = torch.tensor(
                    [nb_transition_frames for _ in range(num_samples)],
                    device=device,
                )

                observed_motion_transition, motion_mask_transition = (
                    self.motion_rep.create_conditions_from_constraints_batched(
                        constraint_lst_transition,
                        transition_lengths,
                        to_normalize=False,  # don't normalize yet
                        device=device,
                    )
                )

                # concatenate the obversed motion / motion mask
                observed_motion = torch.cat([observed_motion_transition, observed_motion], axis=1)
                motion_mask = torch.cat([motion_mask_transition, motion_mask], axis=1)

                # we need to move each observed motion in the batch to the new starting points
                last_smooth_root_2d = smooth_root_2d[:, 0]
                observed_motion = self.motion_rep.translate_2d(
                    observed_motion, -last_smooth_root_2d
                )  # equivalent to:  self.motion_rep.translate_2d_to_zero(observed_motion)

                # remove dummy values after moving
                observed_motion = observed_motion * motion_mask

                lengths = lengths + transition_lengths
                first_heading_angle = compute_heading_angle(last_output["posed_joints"], self.skeleton)[:, 0]
            else:
                if first_heading_angle is None:
                    # Start at 0 angle, but this will change afterward
                    first_heading_angle = torch.tensor([0.0] * bs, device=device)
                else:
                    first_heading_angle = torch.as_tensor(first_heading_angle, device=device)
                    if first_heading_angle.numel() == 1:
                        first_heading_angle = first_heading_angle.repeat(bs)

            observed_motion = self.motion_rep.normalize(observed_motion)

            max_frames = max(lengths)
            motion_pad_mask = length_to_mask(lengths)

            motion = self._generate(
                texts_bs,
                max_frames,
                num_denoising_steps=num_denoising_steps,
                pad_mask=motion_pad_mask,
                first_heading_angle=first_heading_angle,
                motion_mask=motion_mask,
                observed_motion=observed_motion,
                cfg_weight=cfg_weight,
                cfg_type=cfg_type,
            )

            motion = self.motion_rep.unnormalize(motion)

            if not is_first_motion:
                motion_with_transition = self.motion_rep.translate_2d(
                    motion,
                    last_smooth_root_2d,
                )

                if post_processing:
                    # Per-segment postprocessing: inverse, postprocess, re-encode.
                    # The full transition+segment is postprocessed together so the
                    # transition constraints keep the junction smooth.
                    seg_output = self.motion_rep.inverse(
                        motion_with_transition, is_normalized=False, return_numpy=False,
                    )
                    seg_constraints = [list(cl) for cl in constraint_lst_transition]
                    for bi in range(bs):
                        seg_constraints[bi].extend(
                            [c.crop_move(current_frame - nb_transition_frames,
                                         current_frame - nb_transition_frames + num_frame + nb_transition_frames)
                             for c in constraint_lst]
                        )
                    corrected = post_process_motion(
                        seg_output["local_rot_mats"],
                        seg_output["root_positions"],
                        seg_output["foot_contacts"],
                        self.skeleton,
                        seg_constraints,
                        root_margin=root_margin,
                    )
                    seg_output.update(corrected)
                    motion = self.motion_rep(
                        seg_output["local_rot_mats"],
                        seg_output["root_positions"],
                        to_normalize=False,
                        lengths=lengths,
                    )
                else:
                    motion = motion_with_transition[:, num_transition_frames:]
                    transition_frames = motion_with_transition[:, :num_transition_frames]

                    # linearly combine the previously generated transitions with the newly generated ones
                    alpha = torch.linspace(1, 0, num_transition_frames, device=device)[:, None]
                    new_transition_frames = (
                        latest_frames[:, :num_transition_frames] * alpha + (1 - alpha) * transition_frames
                    )

                    # add new transitions frames for A (merging with B prediction of the history)
                    generated_motions.append(new_transition_frames)

            elif post_processing:
                # First segment: postprocess immediately
                seg_output = self.motion_rep.inverse(
                    motion, is_normalized=False, return_numpy=False,
                )
                seg_constraints = constraint_lst_base if constraint_lst_base else []
                corrected = post_process_motion(
                    seg_output["local_rot_mats"],
                    seg_output["root_positions"],
                    seg_output["foot_contacts"],
                    self.skeleton,
                    seg_constraints,
                    root_margin=root_margin,
                )
                seg_output.update(corrected)
                motion = self.motion_rep(
                    seg_output["local_rot_mats"],
                    seg_output["root_positions"],
                    to_normalize=False,
                    lengths=lengths,
                )

            generated_motions.append(motion)
            current_frame += num_frame

        generated_motions = torch.cat(generated_motions, axis=1)  # temporal axis (b, t, d)

        if tosqueeze:
            generated_motions = generated_motions[0]

        output = self.motion_rep.inverse(
            generated_motions,
            is_normalized=False,
            return_numpy=False,
        )

        # Post-processing: already applied per-segment inside the loop above,
        # so no additional post-processing pass is needed here.

        # Convert SOMA output to somaskel77 for external API
        if isinstance(self.skeleton, SOMASkeleton30):
            output = self.skeleton.output_to_SOMASkeleton77(output)

        # Convert to numpy if requested
        if return_numpy:
            output = to_numpy(output)
        return output

    def __call__(
        self,
        prompts: str | list[str],
        num_frames: int | list[int],
        num_denoising_steps: int,
        multi_prompt: bool = False,
        constraint_lst: Optional[list] = [],
        cfg_weight: Optional[float] = [2.0, 2.0],
        num_samples: Optional[int] = None,
        cfg_type: Optional[str] = None,
        return_numpy: bool = False,
        first_heading_angle: Optional[torch.Tensor] = None,
        # for transitioning
        num_transition_frames: int = 5,
        # for postprocess
        post_processing: bool = False,
        root_margin: float = 0.04,
        # progress bar
        progress_bar=tqdm,
    ) -> dict:
        """Generate motion from text prompts and optional kinematic constraints.

        When a single prompt/num_frames pair is given, one motion is generated.
        Passing lists of prompts and/or num_frames produces a batch of
        independent motions. With ``multi_prompt=True``, the prompts are
        treated as sequential segments that are generated and stitched together
        with smooth transitions.

        Args:
            prompts: One or more text descriptions of the desired motion.
                A single string generates one sample; a list generates a batch
                (or sequential segments when ``multi_prompt=True``).
            num_frames: Duration of the generated motion in frames.  Can be a
                single int applied to every prompt or a per-prompt list.
            num_denoising_steps: Number of DDIM denoising steps.  More steps
                generally improve quality at the cost of speed.
            multi_prompt: If ``True``, treat ``prompts`` as an ordered sequence
                of segments and concatenate them with transitions.
            constraint_lst: Per-sample list of kinematic constraints (e.g.
                keyframe poses, end-effector targets, 2-D paths).  Pass an
                empty list for unconstrained generation.
            cfg_weight: Classifier-free guidance scale(s).  A two-element list
                ``[text_cfg, constraint_cfg]`` controls text and constraint
                guidance independently.
            num_samples: Number of samples to generate.
            cfg_type: Override the default CFG strategy set at init
                (e.g. ``"separated"``).
            return_numpy: If ``True``, convert all output tensors to numpy
                arrays.
            first_heading_angle: Initial body heading in radians.  Shape
                ``(B,)`` or scalar.  Defaults to ``0`` (facing +Z).
            num_transition_frames: Number of overlapping frames used to blend
                consecutive segments in multi-prompt mode.
            post_processing: If ``True``, apply post-processing
                (foot-skate cleanup and constraint enforcement).
            root_margin: Horizontal margin (in meters) used by the post-processor
                to determine when to correct root motion. When root deviates more than
                margin from the constraint, the post-processor will correct it.
            progress_bar: Callable wrapping an iterable to display progress
                (default: ``tqdm``).  Pass a no-op to silence output.

        Returns:
            dict: A dictionary of motion tensors (or numpy arrays if
            ``return_numpy=True``) with the following keys:

            - ``local_rot_mats`` – Local joint rotations as rotation matrices.
            - ``global_rot_mats`` – Global joint rotations as rotation matrices.
            - ``posed_joints`` – Joint positions in world space.
            - ``root_positions`` – Root joint positions.
            - ``smooth_root_pos`` – Smoothed root trajectory.
            - ``foot_contacts`` – Boolean foot-contact labels [left heel, left toe, right heel, right toe].
            - ``global_root_heading`` – Root heading angle over time.
        """
        device = self.device

        if multi_prompt:
            # multi prompt generation
            return self._multiprompt(
                prompts,
                num_frames,
                num_denoising_steps,
                constraint_lst,
                cfg_weight,
                num_samples,
                cfg_type,
                return_numpy,
                first_heading_angle,
                num_transition_frames,
                post_processing,
                root_margin,
                progress_bar,
            )

        # Input checking
        tosqueeze = False
        if isinstance(prompts, list) and isinstance(num_frames, list):
            assert len(prompts) == len(num_frames), "The number of prompts should match the number of num_frames."
            num_samples = len(prompts)
        elif isinstance(prompts, list):
            num_samples = len(prompts)
            num_frames = [num_frames for _ in range(num_samples)]
        elif isinstance(num_frames, list):
            num_samples = len(num_frames)
            prompts = [prompts for _ in range(num_samples)]
        else:
            if num_samples is None:
                tosqueeze = True
                num_samples = 1
            prompts = [prompts for _ in range(num_samples)]
            num_frames = [num_frames for _ in range(num_samples)]

        bs = num_samples
        texts = sanitize_texts(prompts)

        lengths = torch.tensor(
            num_frames,
            device=device,
        )
        max_frames = max(lengths)
        motion_pad_mask = length_to_mask(lengths)

        if first_heading_angle is None:
            # Start at 0 angle
            first_heading_angle = torch.tensor([0.0] * bs, device=device)
        else:
            first_heading_angle = torch.as_tensor(first_heading_angle, device=device)
            if first_heading_angle.numel() == 1:
                first_heading_angle = first_heading_angle.repeat(bs)

        observed_motion, motion_mask = None, None
        if constraint_lst:
            observed_motion, motion_mask = self.motion_rep.create_conditions_from_constraints_batched(
                constraint_lst,
                lengths,
                to_normalize=True,
                device=device,
            )

        motion = self._generate(
            texts,
            max_frames,
            num_denoising_steps=num_denoising_steps,
            pad_mask=motion_pad_mask,
            first_heading_angle=first_heading_angle,
            motion_mask=motion_mask,
            observed_motion=observed_motion,
            cfg_weight=cfg_weight,
            cfg_type=cfg_type,
            progress_bar=progress_bar,
        )

        if tosqueeze:
            motion = motion[0]

        output = self.motion_rep.inverse(
            motion,
            is_normalized=True,
            return_numpy=False,  # Keep as tensor for potential post-processing
        )

        # Apply post-processing if requested
        if post_processing:
            corrected = post_process_motion(
                output["local_rot_mats"],
                output["root_positions"],
                output["foot_contacts"],
                self.skeleton,
                constraint_lst,
                root_margin=root_margin,
            )
            # key frame outputs / foot contacts are not changed
            output.update(corrected)

        # Convert SOMA output to somaskel77 for external API
        if isinstance(self.skeleton, SOMASkeleton30):
            output = self.skeleton.output_to_SOMASkeleton77(output)

        # Convert to numpy if requested
        if return_numpy:
            output = to_numpy(output)
        return output

    def _generate(
        self,
        texts: List[str],
        max_frames: int,
        num_denoising_steps: int,
        pad_mask: torch.Tensor,
        first_heading_angle: Optional[torch.Tensor],
        motion_mask: torch.Tensor,
        observed_motion: torch.Tensor,
        cfg_weight: Optional[float] = 2.0,
        text_feat: Optional[torch.Tensor] = None,
        text_pad_mask: Optional[torch.Tensor] = None,
        guide_masks: Optional[Dict] = None,
        cfg_type: Optional[str] = None,
        progress_bar=tqdm,
    ) -> torch.Tensor:
        """Sample full denoising loop.

        Args:
            texts (List[str]): batch of text prompts to use for sampling (if text_feat is not passed in)
        """

        device = self.device
        model_dtype = next(self.denoiser.parameters()).dtype
        if text_feat is None:
            assert text_pad_mask is None
            log.info("Encoding text...")
            text_feat, text_length = self.text_encoder(texts)
            text_feat = text_feat.to(device=device, dtype=model_dtype)

            # handle empty string (set to zero)
            empty_text_mask = [len(text.strip()) == 0 for text in texts]
            text_feat[empty_text_mask] = 0

            # Create the pad mask for the text
            batch_size, maxlen = text_feat.shape[:2]
            tensor_text_length = torch.tensor(text_length, device=device)
            tensor_text_length[empty_text_mask] = 0
            text_pad_mask = torch.arange(maxlen, device=device).expand(batch_size, maxlen) < tensor_text_length[:, None]
        else:
            text_feat = text_feat.to(device=device, dtype=model_dtype)
            if text_pad_mask is not None:
                text_pad_mask = text_pad_mask.to(device=device)

        if motion_mask is not None:
            if motion_mask.dtype == torch.bool:
                motion_mask = 1 * motion_mask

        batch_size = text_feat.shape[0]

        # sample loop
        indices = list(range(num_denoising_steps))[::-1]
        shape = (batch_size, max_frames, self.motion_rep.motion_rep_dim)
        cur_mot = torch.randn(shape, device=self.device)
        num_denoising_steps = torch.tensor(
            [num_denoising_steps], device=self.device
        )  # this and t need to be tensor for onnx export
        # init diffusion with correct num steps before looping
        use_timesteps = self.diffusion.space_timesteps(num_denoising_steps[0])[0]
        self.diffusion.calc_diffusion_vars(use_timesteps)
        for i in progress_bar(indices):
            t = torch.tensor([i] * cur_mot.size(0), device=self.device)
            with torch.inference_mode():
                cur_mot = self.denoising_step(
                    cur_mot,
                    pad_mask,
                    text_feat,
                    text_pad_mask,
                    t,
                    first_heading_angle,
                    motion_mask,
                    observed_motion,
                    num_denoising_steps,
                    cfg_weight,
                    guide_masks=guide_masks,
                    cfg_type=cfg_type,
                )
        return cur_mot
