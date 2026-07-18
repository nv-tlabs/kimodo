# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared utilities: validation decorator, batching, JSON I/O, seeding, tensor conversion."""

import inspect
import json
import random
from collections.abc import Mapping
from functools import wraps
from math import prod
from pathlib import Path
from typing import Any, Callable, ParamSpec, TypeVar, Union

import numpy as np
import torch


def validate(validator, save_args: bool = False, super_init: bool = False):
    """Create a decorator function for validating user inputs.

    Args:
        validator: the function to validate (pydantic dataclass)
        save (bool): save all the attributes to the obj [args[0]]
        super_init (bool): init parent with no arguments (useful for using save on a nn.Module)

    Returns:
        decorator: the decorator function
    """

    def decorator(func):
        @wraps(func)
        def validated_func(*args, **kwargs):
            conf = validator(**kwargs)

            if save_args:
                assert len(args) != 0
                obj = args[0]

                if super_init:
                    # init the parent module
                    super(type(obj), obj).__init__()

                for key, val in conf.__dict__.items():
                    setattr(obj, key, val)
            return func(*args, conf)

        return validated_func

    return decorator


# Type alias for clarity
Tensor = Any

P = ParamSpec("P")
R = TypeVar("R")


def ensure_batched(**spec: int) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Decorator to flatten complex batch dimensions.

    Fixes included:
    1. Handles 1D tensors (tail_ndim=0) correctly without slicing errors.
    2. Skips .reshape() if the input is already purely flat (Optimization).
    """
    if not spec:
        raise ValueError("At least one argument spec must be provided.")

    def decorator(fn: Callable[P, R]) -> Callable[P, R]:
        sig = inspect.signature(fn)

        @wraps(fn)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            bound = sig.bind(*args, **kwargs)
            bound.apply_defaults()

            def _sequence_shape(name: str, value: Any) -> tuple[int, ...]:
                if not isinstance(value, (list, tuple)):
                    return ()
                if len(value) == 0:
                    return (0,)
                first_shape = _sequence_shape(name, value[0])
                for item in value[1:]:
                    item_shape = _sequence_shape(name, item)
                    if item_shape != first_shape:
                        raise ValueError(f"'{name}' must be a rectangular nested sequence, got ragged shape.")
                return (len(value), *first_shape)

            def _shape_and_ndim(name: str, value: Any) -> tuple[tuple[int, ...], int]:
                if hasattr(value, "shape") and hasattr(value, "ndim"):
                    shape = tuple(value.shape)
                    return shape, int(value.ndim)
                if isinstance(value, (list, tuple)):
                    shape = _sequence_shape(name, value)
                    return shape, len(shape)
                raise TypeError(f"'{name}' must be tensor-like or a nested list/tuple, got {type(value)}.")

            def _reshape_like(value: Any, shape: tuple[int, ...], name: str) -> Any:
                if hasattr(value, "reshape"):
                    return value.reshape(*shape)

                if not isinstance(value, (list, tuple)):
                    raise TypeError(f"Cannot reshape '{name}' of type {type(value)}.")

                flat: list[Any] = []

                def _flatten(x: Any) -> None:
                    if isinstance(x, (list, tuple)):
                        for item in x:
                            _flatten(item)
                    else:
                        flat.append(x)

                _flatten(value)
                expected_size = prod(shape) if shape else 1
                if len(flat) != expected_size:
                    raise ValueError(f"Cannot reshape '{name}' with {len(flat)} elements into shape {shape}.")

                def _build(index: int, dims: tuple[int, ...]) -> tuple[Any, int]:
                    if not dims:
                        return flat[index], index + 1
                    items = []
                    for _ in range(dims[0]):
                        item, index = _build(index, dims[1:])
                        items.append(item)
                    return items, index

                rebuilt, used = _build(0, shape)
                if used != len(flat):
                    raise ValueError(f"Internal reshape error for '{name}': used {used}/{len(flat)} elements.")
                if isinstance(value, tuple) and isinstance(rebuilt, list):
                    return tuple(rebuilt)
                return rebuilt

            # --- 1. CANONICAL ARGUMENT ---
            spec_items = list(spec.items())
            canonical_name = None
            canonical_ndim = None
            x0 = None
            for name, ndim in spec_items:
                candidate = bound.arguments.get(name, None)
                if candidate is not None:
                    canonical_name = name
                    canonical_ndim = ndim
                    x0 = candidate
                    break
            if canonical_name is None:
                raise ValueError(
                    "All canonical candidates are None: " + ", ".join(f"'{name}'" for name, _ in spec_items)
                )

            # Calculate split between Batch dims and Feature dims
            expected_tail_dims = canonical_ndim - 1  # e.g. 3 - 1 = 2 (Sequence, Feat)
            x0_shape, x0_ndim = _shape_and_ndim(canonical_name, x0)

            # Validation
            if x0_ndim < expected_tail_dims:
                raise ValueError(f"'{canonical_name}' ndim={x0_ndim} < expected {expected_tail_dims} tail dims.")

            # --- LOGIC FIX 1: Handle 0 tail dims correctly ---
            if expected_tail_dims == 0:
                orig_batch_shape = x0_shape
                tail_shape = ()
            else:
                orig_batch_shape = x0_shape[:-expected_tail_dims]
                tail_shape = x0_shape[-expected_tail_dims:]

            # Calculate flattened batch size
            # If orig_batch_shape is () (scalar input), size is 1.
            B_flat = prod(orig_batch_shape) if orig_batch_shape else 1

            # Determine if we added a fake batch dim (unbatched input)
            is_unbatched_input = len(orig_batch_shape) == 0

            # --- LOGIC FIX 2: Skip reshape if already flat (Optimization) ---
            # If batch shape is already 1D (e.g. [2]), we don't need to reshape [2, 140, 5] -> [2, 140, 5]
            is_already_flat = len(orig_batch_shape) == 1

            if is_unbatched_input:
                # (H, W) -> (1, H, W)
                x0_batched = _reshape_like(x0, (1, *tail_shape), canonical_name)
            elif is_already_flat:
                # (B, H, W) -> Keep as is
                x0_batched = x0
            else:
                # (B1, B2, H, W) -> (B1*B2, H, W)
                x0_batched = _reshape_like(x0, (B_flat, *tail_shape), canonical_name)

            bound.arguments[canonical_name] = x0_batched

            # --- 2. OTHER ARGUMENTS ---
            for name, target_ndim in spec_items:
                if name == canonical_name:
                    continue
                val = bound.arguments.get(name, None)
                if val is None:
                    continue

                arg_tail_dims = target_ndim - 1  # e.g. for lengths=1, tail=0
                val_shape, val_ndim = _shape_and_ndim(name, val)

                # Validate
                if val_ndim < arg_tail_dims:
                    raise ValueError(f"'{name}' ndim={val_ndim} too small.")

                # --- Get Batch Shape (With 0-tail fix) ---
                if arg_tail_dims == 0:
                    val_batch_shape = val_shape
                    val_tail_shape = ()
                else:
                    val_batch_shape = val_shape[:-arg_tail_dims]
                    val_tail_shape = val_shape[-arg_tail_dims:]

                # --- Check Mismatch ---
                # Unbatched inputs must match unbatched canonical
                if len(val_batch_shape) == 0:
                    if not is_unbatched_input:
                        raise ValueError(f"'{name}' is unbatched but canonical is batched.")
                    val_batched = _reshape_like(val, (1, *val_tail_shape), name)
                else:
                    # Batched inputs must match canonical batch shape EXACTLY
                    if val_batch_shape != orig_batch_shape:
                        raise ValueError(
                            f"Batch dimensions mismatch! '{canonical_name}' has {orig_batch_shape}, "
                            f"but '{name}' has {val_batch_shape}."
                        )

                    # Optimization: Don't reshape if already flat
                    if is_already_flat:
                        val_batched = val
                    else:
                        val_batched = _reshape_like(val, (B_flat, *val_tail_shape), name)

                bound.arguments[name] = val_batched

            # --- 3. EXECUTION ---
            out = fn(**bound.arguments)

            # --- 4. RESTORE ---
            def restore(obj):
                if isinstance(obj, Mapping):
                    return {k: restore(v) for k, v in obj.items()}
                if isinstance(obj, (list, tuple)):
                    return type(obj)(restore(x) for x in obj)

                if hasattr(obj, "shape"):
                    if obj.ndim == 0:
                        return obj

                    # Verify batch dimension exists and wasn't reduced
                    if obj.shape[0] != B_flat:
                        return obj

                    # If input was simple (B, ...), return simple (B, ...)
                    if is_already_flat:
                        return obj

                    rest = obj.shape[1:]

                    if is_unbatched_input:
                        assert obj.shape[0] == 1, "The batch size should be 1 for unbatched."
                        return obj[0]

                    return obj.reshape(*orig_batch_shape, *rest)
                return obj

            return restore(out)

        return wrapper

    return decorator


def to_numpy(obj):
    """Recursively convert tensors in dicts/lists/tuples to numpy arrays; leave other types
    unchanged."""
    if isinstance(obj, Mapping):
        return {k: to_numpy(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(to_numpy(x) for x in obj)
    if isinstance(obj, torch.Tensor):
        return obj.cpu().numpy()
    return obj


def to_torch(obj, device=None, dtype=None):
    """Recursively convert numpy arrays in dicts/lists/tuples to torch tensors; optionally move to
    device/dtype."""
    if isinstance(obj, Mapping):
        return {k: to_torch(v, device, dtype) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(to_torch(x, device, dtype) for x in obj)
    if isinstance(obj, np.ndarray):
        obj = torch.from_numpy(obj)
    if isinstance(obj, torch.Tensor):
        if dtype is not None:
            obj = obj.to(dtype=dtype)
        if device is None:
            return obj
        return obj.to(device)
    return obj


def seed_everything(seed: int, deterministic: bool = False) -> None:
    """Seed all random number generators."""
    random.seed(seed)  # for Python random module.
    np.random.seed(seed)  # for NumPy.
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic and torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True  # for deterministic behavior.
        torch.backends.cudnn.benchmark = False  # if you want to make the behavior deterministic.


def load_json(path: Union[str, Path]) -> Any:
    """Load a JSON file and return its contents.

    Args:
        path (str | Path): Path to the JSON file.

    Returns:
        Any: Parsed JSON content (dict, list, etc.).

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file is not valid JSON.
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"JSON file not found: {path}")

    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in file {path}: {e}") from e


def save_json(path: Union[str, Path], data: Any) -> None:
    """Save data to a JSON file.

    Args:
        path (str | Path): Path to the JSON file.
        data (Any): Data to save (must be JSON serializable).

    Raises:
        ValueError: If the data is not JSON serializable.
    """
    path = Path(path)

    # Create parent directories if they don't exist
    path.parent.mkdir(parents=True, exist_ok=True)

    try:
        with path.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    except (TypeError, ValueError) as e:
        raise ValueError(f"Data is not JSON serializable: {e}") from e
