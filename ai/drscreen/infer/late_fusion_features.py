from __future__ import annotations

from collections.abc import Sequence

import torch

LESION_CHANNELS = ("MA", "HE", "EX", "SE")
FUSION_AREA_THRESHOLDS = (0.05, 0.1, 0.2, 0.3, 0.5)
FUSION_TOPK_FRACS = (0.001, 0.01, 0.05)


def extract_late_fusion_features(
    *,
    v31_probability: float,
    v31_logit: float,
    seg_prob: torch.Tensor,
    schema: Sequence[str],
    area_thresholds: Sequence[float] = FUSION_AREA_THRESHOLDS,
    topk_fracs: Sequence[float] = FUSION_TOPK_FRACS,
) -> list[float]:
    """Return the v31+v8b late-fusion feature vector in training order."""

    if seg_prob.ndim != 3:
        raise ValueError(f"Expected seg_prob [C,H,W], got {tuple(seg_prob.shape)}")
    if seg_prob.shape[0] != len(LESION_CHANNELS):
        raise ValueError(
            f"Expected {len(LESION_CHANNELS)} lesion channels, got {seg_prob.shape[0]}"
        )

    seg_prob = seg_prob.float()
    union = seg_prob.amax(dim=0, keepdim=True)
    maps = torch.cat([seg_prob, union], dim=0)
    labels = (*LESION_CHANNELS, "union")
    flat = maps.flatten(1)

    features: dict[str, float] = {
        "v31_probability": float(v31_probability),
        "v31_logit": float(v31_logit),
    }

    means = flat.mean(dim=1)
    maxes = flat.amax(dim=1)
    stds = flat.std(dim=1)
    for idx, label in enumerate(labels):
        features[f"{label}_mean"] = float(means[idx].item())
    for idx, label in enumerate(labels):
        features[f"{label}_max"] = float(maxes[idx].item())
    for idx, label in enumerate(labels):
        features[f"{label}_std"] = float(stds[idx].item())

    for threshold in area_thresholds:
        threshold_text = f"{float(threshold):g}"
        areas = (flat >= float(threshold)).float().mean(dim=1)
        for idx, label in enumerate(labels):
            features[f"{label}_area_ge_{threshold_text}"] = float(areas[idx].item())

    n_pixels = int(flat.shape[-1])
    for frac in topk_fracs:
        frac_float = float(frac)
        frac_text = f"{frac_float:g}"
        k = max(1, int(round(n_pixels * frac_float)))
        top_means = torch.topk(flat, k=k, dim=1).values.mean(dim=1)
        for idx, label in enumerate(labels):
            features[f"{label}_top_{frac_text}_mean"] = float(top_means[idx].item())

    missing = [name for name in schema if name not in features]
    if missing:
        raise ValueError(f"Late-fusion feature extractor missing schema keys: {missing}")
    return [features[name] for name in schema]
