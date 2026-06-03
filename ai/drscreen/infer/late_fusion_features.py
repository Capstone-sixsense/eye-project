from __future__ import annotations

from collections.abc import Sequence

import cv2
import numpy as np
import torch

LESION_CHANNELS = ("MA", "HE", "EX", "SE")
FUSION_AREA_THRESHOLDS = (0.05, 0.1, 0.2, 0.3, 0.5)
FUSION_TOPK_FRACS = (0.001, 0.01, 0.05)
FUSION_CC_THRESHOLDS = (0.1, 0.3, 0.5)
FUSION_EXTENDED_ACTIVE_THRESHOLD = 0.3


def base_lesion_feature_names(
    *,
    area_thresholds: Sequence[float] = FUSION_AREA_THRESHOLDS,
    topk_fracs: Sequence[float] = FUSION_TOPK_FRACS,
) -> list[str]:
    labels = [*LESION_CHANNELS, "union"]
    names = [f"{label}_mean" for label in labels]
    names.extend(f"{label}_max" for label in labels)
    names.extend(f"{label}_std" for label in labels)
    for threshold in area_thresholds:
        names.extend(f"{label}_area_ge_{float(threshold):g}" for label in labels)
    for frac in topk_fracs:
        names.extend(f"{label}_top_{float(frac):g}_mean" for label in labels)
    return names


def extended_lesion_feature_names() -> list[str]:
    labels = [*LESION_CHANNELS, "union"]
    names: list[str] = []
    for threshold in FUSION_CC_THRESHOLDS:
        names.extend(f"{label}_cc_count_ge_{float(threshold):g}" for label in labels)
    names.extend(
        f"{label}_cc_mean_area_ge_{FUSION_EXTENDED_ACTIVE_THRESHOLD:g}"
        for label in labels
    )
    names.extend(
        f"{label}_center_outer_area_ratio_ge_{FUSION_EXTENDED_ACTIVE_THRESHOLD:g}"
        for label in labels
    )
    names.extend(f"{label}_entropy_norm" for label in labels)
    names.extend(
        [
            f"HE_EX_overlap_ge_{FUSION_EXTENDED_ACTIVE_THRESHOLD:g}",
            f"MA_HE_overlap_ge_{FUSION_EXTENDED_ACTIVE_THRESHOLD:g}",
        ]
    )
    return names


def extract_late_fusion_features(
    *,
    v31_probability: float,
    v31_logit: float,
    seg_prob: torch.Tensor,
    schema: Sequence[str],
    area_thresholds: Sequence[float] = FUSION_AREA_THRESHOLDS,
    topk_fracs: Sequence[float] = FUSION_TOPK_FRACS,
    od_xy: tuple[float, float] | None = None,
    od_diameter: float | None = None,
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
        features.update(extract_extended_lesion_feature_dict(seg_prob))
        missing = [name for name in schema if name not in features]
    if missing and od_xy is not None and od_diameter is not None:
        features.update(extract_od_anchored_feature_dict(seg_prob, od_xy, od_diameter))
        missing = [name for name in schema if name not in features]
    if missing:
        raise ValueError(f"Late-fusion feature extractor missing schema keys: {missing}")
    return [features[name] for name in schema]


def extract_extended_lesion_feature_dict(seg_prob: torch.Tensor) -> dict[str, float]:
    if seg_prob.ndim != 3:
        raise ValueError(f"Expected seg_prob [C,H,W], got {tuple(seg_prob.shape)}")
    if seg_prob.shape[0] != len(LESION_CHANNELS):
        raise ValueError(
            f"Expected {len(LESION_CHANNELS)} lesion channels, got {seg_prob.shape[0]}"
        )

    seg_prob = seg_prob.detach().float().cpu()
    union = seg_prob.amax(dim=0, keepdim=True)
    maps = torch.cat([seg_prob, union], dim=0)
    labels = (*LESION_CHANNELS, "union")
    h, w = int(maps.shape[1]), int(maps.shape[2])
    n_pixels = max(1, h * w)
    features: dict[str, float] = {}

    for threshold in FUSION_CC_THRESHOLDS:
        threshold_text = f"{float(threshold):g}"
        for idx, label in enumerate(labels):
            count, _ = _connected_component_stats(maps[idx], float(threshold), n_pixels)
            features[f"{label}_cc_count_ge_{threshold_text}"] = float(count)

    threshold = float(FUSION_EXTENDED_ACTIVE_THRESHOLD)
    threshold_text = f"{threshold:g}"
    center_mask = _center_radius_mask(h, w)
    outer_mask = ~center_mask
    for idx, label in enumerate(labels):
        _, mean_area = _connected_component_stats(maps[idx], threshold, n_pixels)
        features[f"{label}_cc_mean_area_ge_{threshold_text}"] = float(mean_area)
        active = (maps[idx].numpy() >= threshold)
        center_rate = active[center_mask].mean() if center_mask.any() else 0.0
        outer_rate = active[outer_mask].mean() if outer_mask.any() else 0.0
        features[f"{label}_center_outer_area_ratio_ge_{threshold_text}"] = float(
            center_rate / (outer_rate + 1e-8)
        )
        flat = maps[idx].flatten().numpy().astype(np.float64)
        total = float(flat.sum())
        if total <= 1e-12:
            features[f"{label}_entropy_norm"] = 0.0
        else:
            p = flat / total
            entropy = -float(np.sum(p * np.log(p + 1e-12)))
            features[f"{label}_entropy_norm"] = float(entropy / np.log(n_pixels))

    channel_maps = {label: maps[idx].numpy() for idx, label in enumerate(LESION_CHANNELS)}
    features[f"HE_EX_overlap_ge_{threshold_text}"] = float(
        np.logical_and(
            channel_maps["HE"] >= threshold,
            channel_maps["EX"] >= threshold,
        ).mean()
    )
    features[f"MA_HE_overlap_ge_{threshold_text}"] = float(
        np.logical_and(
            channel_maps["MA"] >= threshold,
            channel_maps["HE"] >= threshold,
        ).mean()
    )
    return features


def extract_extended_lesion_feature_values(
    seg_prob: torch.Tensor,
    schema: Sequence[str],
) -> list[float]:
    features = extract_extended_lesion_feature_dict(seg_prob)
    missing = [name for name in schema if name not in features]
    if missing:
        raise ValueError(f"Extended lesion feature extractor missing schema keys: {missing}")
    return [features[name] for name in schema]


OD_PERIPAPILLARY_RADIUS_FACTOR = 2.0


def od_anchored_feature_names() -> list[str]:
    """OD-anchored (optic-disc-relative) lesion feature names (Problem 3, option A).

    Anatomy-grounded spatial features replacing the fixed geometric center: how
    lesion activity is distributed relative to the optic disc.
    """
    labels = [*LESION_CHANNELS, "union"]
    t = f"{FUSION_EXTENDED_ACTIVE_THRESHOLD:g}"
    names: list[str] = []
    names.extend(f"{label}_peripapillary_ratio_ge_{t}" for label in labels)
    names.extend(f"{label}_od_dist_mean_ge_{t}" for label in labels)
    names.extend(f"{label}_od_dist_p90_ge_{t}" for label in labels)
    return names


def extract_od_anchored_feature_dict(
    seg_prob: torch.Tensor,
    od_xy: tuple[float, float],
    od_diameter: float,
) -> dict[str, float]:
    """OD-anchored features in seg_prob pixel space.

    od_xy = (x, y) optic-disc centre and od_diameter are expressed in the
    seg_prob (H, W) coordinate system. Distances are normalized to OD-diameter
    units so the features are resolution- and scale-invariant.
    """
    if seg_prob.ndim != 3 or seg_prob.shape[0] != len(LESION_CHANNELS):
        raise ValueError(
            f"Expected seg_prob [{len(LESION_CHANNELS)},H,W], got {tuple(seg_prob.shape)}"
        )
    seg = seg_prob.detach().float().cpu().numpy()
    union = seg.max(axis=0, keepdims=True)
    maps = np.concatenate([seg, union], axis=0)
    labels = (*LESION_CHANNELS, "union")
    h, w = int(maps.shape[1]), int(maps.shape[2])
    od_x, od_y = float(od_xy[0]), float(od_xy[1])
    diam = max(1e-6, float(od_diameter))
    yy, xx = np.mgrid[0:h, 0:w]
    dist = np.sqrt((xx - od_x) ** 2 + (yy - od_y) ** 2) / diam
    thr = float(FUSION_EXTENDED_ACTIVE_THRESHOLD)
    t = f"{thr:g}"
    features: dict[str, float] = {}
    for idx, label in enumerate(labels):
        active = maps[idx] >= thr
        if not active.any():
            features[f"{label}_peripapillary_ratio_ge_{t}"] = 0.0
            features[f"{label}_od_dist_mean_ge_{t}"] = 0.0
            features[f"{label}_od_dist_p90_ge_{t}"] = 0.0
            continue
        d = dist[active]
        features[f"{label}_peripapillary_ratio_ge_{t}"] = float(
            (d <= OD_PERIPAPILLARY_RADIUS_FACTOR).mean()
        )
        features[f"{label}_od_dist_mean_ge_{t}"] = float(d.mean())
        features[f"{label}_od_dist_p90_ge_{t}"] = float(np.percentile(d, 90))
    return features


def _connected_component_stats(
    prob_map: torch.Tensor,
    threshold: float,
    n_pixels: int,
) -> tuple[int, float]:
    binary = (prob_map.numpy() >= threshold).astype(np.uint8)
    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    count = max(0, int(num_labels) - 1)
    if count == 0:
        return 0, 0.0
    areas = stats[1:, cv2.CC_STAT_AREA].astype(np.float64)
    return count, float(areas.mean() / max(1, n_pixels))


def _center_radius_mask(h: int, w: int) -> np.ndarray:
    y, x = np.ogrid[:h, :w]
    cy = (h - 1) / 2.0
    cx = (w - 1) / 2.0
    radius = 0.25 * min(h, w)
    return (y - cy) ** 2 + (x - cx) ** 2 <= radius ** 2
