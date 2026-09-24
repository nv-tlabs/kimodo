#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Generate deterministic MotionCorrection output for cross-architecture comparison."""

import argparse

import numpy as np
import torch
from motion_correction.motion_postprocess import correct_motion


class Joint:
    def __init__(self, name, parent, translation, rotation, tag=""):
        self.name = name
        self.parent = parent
        self.t_pose_translation = translation
        self.t_pose_rotation = rotation
        self.retarget_tag = tag


def create_rig():
    identity = [0.0, 0.0, 0.0, 1.0]
    return [
        Joint("Hips", None, [0.0, 1.0, 0.0], identity, "Root"),
        Joint("Spine", "Hips", [0.0, 0.1, 0.0], identity),
        Joint("LeftUpLeg", "Hips", [-0.1, -0.05, 0.0], identity),
        Joint("LeftLeg", "LeftUpLeg", [0.0, -0.4, 0.0], identity),
        Joint("LeftFoot", "LeftLeg", [0.0, -0.4, 0.0], identity, "LeftFoot"),
        Joint("RightUpLeg", "Hips", [0.1, -0.05, 0.0], identity),
        Joint("RightLeg", "RightUpLeg", [0.0, -0.4, 0.0], identity),
        Joint("RightFoot", "RightLeg", [0.0, -0.4, 0.0], identity, "RightFoot"),
        Joint("LeftArm", "Spine", [-0.3, 0.3, 0.0], identity),
        Joint("LeftHand", "LeftArm", [-0.3, 0.0, 0.0], identity, "LeftHand"),
        Joint("RightArm", "Spine", [0.3, 0.3, 0.0], identity),
        Joint("RightHand", "RightArm", [0.3, 0.0, 0.0], identity, "RightHand"),
    ]


def generate_fixture() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(20260716)
    frames, joints = 48, 12
    hips_np = rng.normal(0.0, 0.15, size=(1, frames, 3)).astype(np.float32)
    rotations_np = rng.normal(size=(1, frames, joints, 4)).astype(np.float32)
    rotations_np /= np.linalg.norm(rotations_np, axis=-1, keepdims=True)
    contacts_np = rng.uniform(size=(1, frames, 4)).astype(np.float32)

    hips = torch.from_numpy(hips_np.copy())
    rotations = torch.from_numpy(rotations_np.copy())
    contacts = torch.from_numpy(contacts_np)
    hips_target = torch.from_numpy((hips_np[0] + np.array([0.1, 0.0, -0.05], dtype=np.float32)))
    rotations_target = torch.from_numpy(rotations_np[0].copy())

    masks = {
        name: torch.zeros(frames) for name in ("Root", "FullBody", "LeftHand", "RightHand", "LeftFoot", "RightFoot")
    }
    masks["Root"][[4, 17, 31]] = 1.0
    masks["FullBody"][[12, 36]] = 1.0
    masks["LeftHand"][[8, 27]] = 1.0
    masks["RightFoot"][[20, 41]] = 1.0

    correct_motion(
        hipTranslations=hips,
        jointRotations=rotations,
        contacts=contacts,
        hipTranslationsInput=hips_target,
        rotationsInput=rotations_target,
        constraint_masks=masks,
        contact_threshold=0.5,
        root_margin=0.01,
        working_rig=create_rig(),
    )

    return hips.numpy(), rotations.numpy()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output")
    args = parser.parse_args()

    hips, rotations = generate_fixture()
    np.savez(args.output, hips=hips, rotations=rotations)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
