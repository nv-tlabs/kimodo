# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Classifier-free guidance wrapper for the denoiser at sampling time."""

from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn

CFG_TYPES = ["nocfg", "regular", "separated"]


def _no_active_constraint(motion_mask: Optional[torch.Tensor]) -> bool:
    """True when no constraint is set (mask is None or all zero).

    Returns False while tracing or compiling so exported graphs keep the general path.
    """
    if torch.jit.is_tracing() or torch.compiler.is_compiling():
        return False
    return motion_mask is None or not motion_mask.any()


class ClassifierFreeGuidedModel(nn.Module):
    """Wrapper around denoiser to use classifier-free guidance at sampling time."""

    def __init__(self, model: nn.Module, cfg_type: Optional[str] = "separated"):
        """Wrap the denoiser for classifier-free guidance; cfg_type in CFG_TYPES (e.g. 'regular',
        'nocfg')."""
        super().__init__()
        self.model = model
        assert cfg_type in CFG_TYPES, f"Invalid cfg_type: {cfg_type}"
        self.cfg_type_default = cfg_type

    def forward(
        self,
        cfg_weight: Union[float, Tuple[float, float]],
        x: torch.Tensor,
        x_pad_mask: torch.Tensor,
        text_feat: torch.Tensor,
        text_feat_pad_mask: torch.Tensor,
        timesteps: torch.Tensor,
        first_heading_angle: Optional[torch.Tensor] = None,
        motion_mask: Optional[torch.Tensor] = None,
        observed_motion: Optional[torch.Tensor] = None,
        cfg_type: Optional[str] = None,
    ) -> torch.Tensor:
        """
        Args:
            cfg_weight (float): guidance weight float or tuple of floats with (text, constraint) weights if using separated cfg
            x (torch.Tensor): [B, T, dim_motion] current noisy motion
            x_pad_mask (torch.Tensor): [B, T] attention mask, positions with True are allowed to attend, False are not
            text_feat (torch.Tensor): [B, max_text_len, llm_dim] embedded text prompts
            text_feat_pad_mask (torch.Tensor): [B, max_text_len] attention mask, positions with True are allowed to attend, False are not
            timesteps (torch.Tensor): [B,] current denoising step
            motion_mask
            observed_motion
            neutral_joints (torch.Tensor): [B, nbjoints] The neutral joints of the motions

        Returns:
            torch.Tensor: same size as input x
        """

        if cfg_type is None:
            cfg_type = self.cfg_type_default

        assert cfg_type in CFG_TYPES, f"Invalid cfg_type: {cfg_type}"

        # batched conditional and uncond pass together
        if cfg_type == "nocfg":
            return self.model(
                x,
                x_pad_mask,
                text_feat,
                text_feat_pad_mask,
                timesteps,
                first_heading_angle=first_heading_angle,
                motion_mask=motion_mask,
                observed_motion=observed_motion,
            )
        elif cfg_type == "regular":
            assert isinstance(cfg_weight, (float, int)), "cfg_weight must be a single float for regular CFG"
            # out_uncond + w * (out_text_and_constraint - out_uncond)
            text_feat = torch.concatenate([text_feat, 0 * text_feat], dim=0)
            if motion_mask is not None:
                motion_mask = torch.concatenate([motion_mask, 0 * motion_mask], dim=0)
            if observed_motion is not None:
                observed_motion = torch.concatenate([observed_motion, observed_motion], dim=0)
            if first_heading_angle is not None:
                first_heading_angle = torch.concatenate([first_heading_angle, first_heading_angle], dim=0)

            out_cond_uncond = self.model(
                torch.concatenate([x, x], dim=0),
                torch.concatenate([x_pad_mask, x_pad_mask], dim=0),
                text_feat,
                torch.concatenate([text_feat_pad_mask, False * text_feat_pad_mask], dim=0),
                torch.concatenate([timesteps, timesteps], dim=0),
                first_heading_angle=first_heading_angle,
                motion_mask=motion_mask,
                observed_motion=observed_motion,
            )

            out, out_uncond = torch.chunk(out_cond_uncond, 2)
            out_new = out_uncond + (cfg_weight * (out - out_uncond))
        elif cfg_type == "separated" and _no_active_constraint(motion_mask):
            assert len(cfg_weight) == 2, "cfg_weight must be a tuple of two floats for separated CFG"
            # Without an active constraint, the constraint pass gets exactly the inputs of the uncond
            # pass, so w_constraint * (out_constraint - out_uncond) is zero. Skip that pass:
            # out_uncond + w_text * (out_text - out_uncond)
            text_feat = torch.concatenate([text_feat, 0 * text_feat], dim=0)
            if motion_mask is not None:
                motion_mask = torch.concatenate([0 * motion_mask, 0 * motion_mask], dim=0)
            if observed_motion is not None:
                observed_motion = torch.concatenate([observed_motion, observed_motion], dim=0)
            if first_heading_angle is not None:
                first_heading_angle = torch.concatenate([first_heading_angle, first_heading_angle], dim=0)

            out_text_uncond = self.model(
                torch.concatenate([x, x], dim=0),
                torch.concatenate([x_pad_mask, x_pad_mask], dim=0),
                text_feat,
                torch.concatenate([text_feat_pad_mask, False * text_feat_pad_mask], dim=0),
                torch.concatenate([timesteps, timesteps], dim=0),
                first_heading_angle=first_heading_angle,
                motion_mask=motion_mask,
                observed_motion=observed_motion,
            )

            out_text, out_uncond = torch.chunk(out_text_uncond, 2)
            out_new = out_uncond + (cfg_weight[0] * (out_text - out_uncond))
        elif cfg_type == "separated":
            assert len(cfg_weight) == 2, "cfg_weight must be a tuple of two floats for separated CFG"
            # out_uncond + w_text * (out_text - out_uncond) + w_constraint * (out_constraint - out_uncond)
            text_feat = torch.concatenate([text_feat, 0 * text_feat, 0 * text_feat], dim=0)
            if motion_mask is not None:
                motion_mask = torch.concatenate([0 * motion_mask, motion_mask, 0 * motion_mask], dim=0)
            if observed_motion is not None:
                observed_motion = torch.concatenate([observed_motion, observed_motion, observed_motion], dim=0)
            if first_heading_angle is not None:
                first_heading_angle = torch.concatenate(
                    [first_heading_angle, first_heading_angle, first_heading_angle],
                    dim=0,
                )

            out_cond_uncond = self.model(
                torch.concatenate([x, x, x], dim=0),
                torch.concatenate([x_pad_mask, x_pad_mask, x_pad_mask], dim=0),
                text_feat,
                torch.concatenate(
                    [
                        text_feat_pad_mask,
                        False * text_feat_pad_mask,
                        False * text_feat_pad_mask,
                    ],
                    dim=0,
                ),
                torch.concatenate([timesteps, timesteps, timesteps], dim=0),
                first_heading_angle=first_heading_angle,
                motion_mask=motion_mask,
                observed_motion=observed_motion,
            )

            out_text, out_constraint, out_uncond = torch.chunk(out_cond_uncond, 3)
            out_new = (
                out_uncond + (cfg_weight[0] * (out_text - out_uncond)) + (cfg_weight[1] * (out_constraint - out_uncond))
            )
        else:
            raise ValueError(f"Invalid cfg_type: {cfg_type}")

        return out_new
