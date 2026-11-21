# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Constraint conditioning: build index and data dicts from constraint sets for the denoiser."""

from collections import defaultdict

import torch


def build_condition_dicts(constraints_lst: list):
    """Bridge a constraint set to the denoiser's conditioning inputs.

    Each constraint (see ``kimodo.constraints``) appends its target indices (frame or
    (frame, joint) pairs) and values into per-feature-name lists via ``update_constraints``.
    The reps' ``create_conditions`` then consume the two dicts (through
    :class:`ConditionWriters`) to fill ``observed_motion`` / ``motion_mask`` for the denoiser.

    Args:
        constraints_lst: Constraint objects exposing ``update_constraints(data_dict, index_dict)``.

    Returns:
        ``(index_dict, data_dict)``: feature name -> list of index tensors / value tensors,
        in matching order.
    """
    index_dict = defaultdict(list)
    data_dict = defaultdict(list)
    for constraint in constraints_lst:
        constraint.update_constraints(data_dict, index_dict)
    return index_dict, data_dict


class ConditionWriters:
    """Gather/write helpers shared by the reps' ``create_conditions`` implementations.

    Fills ``observed_motion`` / ``motion_mask`` in place. The block routing (which feature names
    exist and where constraint values land) stays in each rep; only the mechanics live here.

    Args:
        joints_per_block: number of joints laid out in a per-joint feature block, per feature
            name (the write stride is ``block_width / joints_per_block(fname)``).
    """

    def __init__(self, observed_motion, motion_mask, slice_dict, index_dict, data_dict, device, joints_per_block):
        self.observed_motion = observed_motion
        self.motion_mask = motion_mask
        self.slice_dict = slice_dict
        self.index_dict = index_dict
        self.data_dict = data_dict
        self.device = device
        self.joints_per_block = joints_per_block

    def gather(self, fname: str):
        """Concatenate, uniquify, and dtype-match the constraint entries of one feature."""
        if fname not in self.index_dict or not self.index_dict[fname]:
            return None
        indices = torch.cat([x if isinstance(x, torch.Tensor) else torch.tensor(x) for x in self.index_dict[fname]])
        indices = indices.to(device=self.device, dtype=torch.long)
        indices, data = get_unique_index_and_data(indices, torch.cat(self.data_dict[fname]))
        return indices, data.to(device=self.device, dtype=self.observed_motion.dtype)

    def write_frames(self, fname: str, frame_idx, values, components=None) -> None:
        """Write per-frame values into (a subset of the components of) a feature block."""
        s = self.slice_dict[fname]
        cols = components if components is not None else range(s.stop - s.start)
        cols = torch.tensor([s.start + c for c in cols], device=self.device)
        self.observed_motion[frame_idx[:, None], cols] = values.reshape(len(frame_idx), len(cols))
        self.motion_mask[frame_idx[:, None], cols] = True

    def write_joints(self, fname: str, pair_indices, values) -> None:
        """Write per-(frame, joint) values into a per-joint feature block."""
        s = self.slice_dict[fname]
        stride = (s.stop - s.start) // self.joints_per_block(fname)
        rows = pair_indices[:, 0, None].expand(-1, stride)
        cols = s.start + pair_indices[:, 1, None] * stride + torch.arange(stride, device=self.device)
        self.observed_motion[rows, cols] = values.reshape(-1, stride)
        self.motion_mask[rows, cols] = True


def get_unique_index_and_data(indices_lst, data):
    """Deduplicate constraint indices, keeping one data row per unique index.

    Multiple constraints may target the same frame (or (frame, joint) pair); writing them all
    would let list order decide silently. This keeps exactly one ``data`` row per unique index
    (among duplicates, the surviving occurrence is the ``scatter_`` winner, which torch leaves
    unspecified) and returns indices sorted as ``torch.unique`` sorts them.

    Args:
        indices_lst: ``[N]`` or ``[N, 2]`` integer tensor of target indices.
        data: ``[N, ...]`` values aligned with ``indices_lst``.

    Returns:
        ``(unique_indices, data)`` with matching first dimension.
    """
    # unique + sort them by t
    indices_unique, inverse = torch.unique(indices_lst, dim=0, return_inverse=True)
    # pick first value for each unique (t, j)
    first_idx = torch.zeros(indices_unique.size(0), dtype=torch.long, device=inverse.device)
    first_idx.scatter_(0, inverse, torch.arange(len(inverse), device=inverse.device))
    assert (indices_lst[first_idx] == indices_unique).all()
    # get the data
    indices_lst = indices_lst[first_idx]
    data = data[first_idx]
    return indices_lst, data
