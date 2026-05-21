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
    if isinstance(checkpoint, dict) and "fusion_version" in checkpoint:
        if not hasattr(model, "classifier") or not hasattr(model, "segmenter"):
            raise ValueError(
                "Fusion checkpoint requires a model with classifier and segmenter modules."
            )
        missing_classifier, unexpected_classifier = model.classifier.load_state_dict(
            checkpoint["classifier_state_dict"],
            strict=strict,
        )
        missing_segmenter, unexpected_segmenter = model.segmenter.load_state_dict(
            checkpoint["segmenter_state_dict"],
            strict=strict,
        )
        if hasattr(model, "meta_params"):
            model.meta_params = model._normalize_meta_params(checkpoint.get("meta_classifier"))
        if hasattr(model, "feature_schema"):
            model.feature_schema = list(checkpoint.get("feature_schema") or [])
        if hasattr(model, "feature_extraction"):
            model.feature_extraction = dict(checkpoint.get("feature_extraction") or {})
        return (
            [f"classifier.{key}" for key in missing_classifier]
            + [f"segmenter.{key}" for key in missing_segmenter],
            [f"classifier.{key}" for key in unexpected_classifier]
            + [f"segmenter.{key}" for key in unexpected_segmenter],
        )

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
