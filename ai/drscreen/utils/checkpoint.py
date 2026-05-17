from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn


def load_state_from_checkpoint(
    model: nn.Module,
    checkpoint: dict[str, Any],
    *,
    strict: bool = True,
) -> tuple[list[str], list[str]]:
    state = checkpoint.get("model_state_dict", checkpoint)
    if not strict:
        model_state = model.state_dict()
        shape_mismatched = [
            k for k, v in state.items()
            if k in model_state and model_state[k].shape != v.shape
        ]
        if shape_mismatched:
            state = {k: v for k, v in state.items() if k not in shape_mismatched}
        missing, unexpected = model.load_state_dict(state, strict=False)
        return list(missing) + shape_mismatched, unexpected
    missing, unexpected = model.load_state_dict(state, strict=strict)
    return missing, unexpected


def read_checkpoint_auroc(path: Path) -> float:
    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        return float(ckpt.get("val_metrics", {}).get("auroc") or 0.0)
    except Exception:
        return 0.0
