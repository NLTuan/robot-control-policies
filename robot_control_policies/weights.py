from dataclasses import dataclass
from pathlib import Path
import re

import torch
from torch import nn


@dataclass(frozen=True)
class WeightLoadReport:
    loaded: list[str]
    missing: list[str]
    unexpected: list[str]
    shape_mismatched: list[tuple[str, tuple[int, ...], tuple[int, ...]]]

    def summary(self):
        return (
            f"loaded={len(self.loaded)} "
            f"missing={len(self.missing)} "
            f"unexpected={len(self.unexpected)} "
            f"shape_mismatched={len(self.shape_mismatched)}"
        )


def checkpoint_to_state_dict(checkpoint):
    """Return the tensor dictionary inside a common PyTorch checkpoint format."""
    if not isinstance(checkpoint, dict):
        raise TypeError(f"checkpoint must be a dict, got {type(checkpoint)!r}")

    for key in ("model", "state_dict", "params"):
        value = checkpoint.get(key)
        if _looks_like_state_dict(value):
            return value

    if _looks_like_state_dict(checkpoint):
        return checkpoint

    keys = ", ".join(str(key) for key in checkpoint.keys())
    raise ValueError(f"could not find a state dict in checkpoint keys: {keys}")


def load_checkpoint_state_dict(path, map_location="cpu"):
    checkpoint = torch.load(Path(path), map_location=map_location)
    return checkpoint_to_state_dict(checkpoint)


def merge_pretrained_state_dict(
    model,
    pretrained_state_dict,
    *,
    allow_missing_regex=".*",
):
    """Overlay compatible pretrained tensors onto an initialized model.

    This mirrors the pattern used by OpenPI-style loaders: first initialize the
    complete target model, then replace matching tensors from the pretrained
    checkpoint, and keep allowed missing tensors from the initialized model.
    """
    target = model.state_dict()
    allow_missing = re.compile(allow_missing_regex)

    merged = {}
    loaded = []
    unexpected = []
    shape_mismatched = []

    for key, value in pretrained_state_dict.items():
        if key not in target:
            unexpected.append(key)
            continue

        if target[key].shape != value.shape:
            shape_mismatched.append((key, tuple(value.shape), tuple(target[key].shape)))
            continue

        merged[key] = value.detach().to(dtype=target[key].dtype, device=target[key].device)
        loaded.append(key)

    missing = []
    disallowed_missing = []
    for key, value in target.items():
        if key in merged:
            continue

        missing.append(key)
        if not allow_missing.fullmatch(key):
            disallowed_missing.append(key)
        merged[key] = value

    if disallowed_missing:
        joined = "\n  ".join(disallowed_missing)
        raise ValueError(f"checkpoint is missing required weights:\n  {joined}")

    return merged, WeightLoadReport(
        loaded=loaded,
        missing=missing,
        unexpected=unexpected,
        shape_mismatched=shape_mismatched,
    )


def load_pretrained_weights(
    model,
    path,
    *,
    map_location="cpu",
    allow_missing_regex=".*",
):
    state_dict = load_checkpoint_state_dict(path, map_location=map_location)
    merged, report = merge_pretrained_state_dict(
        model,
        state_dict,
        allow_missing_regex=allow_missing_regex,
    )
    model.load_state_dict(merged, strict=True)
    return report


def _looks_like_state_dict(value):
    if not isinstance(value, dict) or not value:
        return False
    return all(isinstance(key, str) for key in value) and all(
        torch.is_tensor(tensor) for tensor in value.values()
    )
