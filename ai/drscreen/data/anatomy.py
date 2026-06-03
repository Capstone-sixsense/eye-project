"""Lightweight optic-disc (OD) and fovea localization for fundus images.

First-cut, training-free detector for Problem 3 (anatomy-aware lesion evidence):
- OD: brightest blurred region on the red channel within the retina mask. The OD
  is the brightest large structure; bright lesions (hard exudates) are the main
  failure mode, hence confidence is reported for a geometric-center fallback.
- Fovea: darkest blurred region ~2.5 OD-diameters temporal to the OD (toward the
  retina centre), searched on the green channel.

Coordinates are returned in the input image's pixel space. Detection runs on a
downscaled copy (max dim `work_max_dim`) for speed and is scaled back.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(slots=True)
class AnatomyLandmarks:
    od_xy: tuple[float, float]
    od_confidence: float
    od_diameter: float
    fovea_xy: tuple[float, float]
    fovea_confidence: float
    image_size: tuple[int, int]  # (width, height)


def _retina_mask(rgb: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    _, thresholded = cv2.threshold(gray, 8, 255, cv2.THRESH_BINARY)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(thresholded)
    if num_labels <= 1:
        return np.ones(gray.shape, dtype=bool)
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return labels == largest


def _as_rgb(image: object) -> np.ndarray:
    if hasattr(image, "convert"):
        return np.asarray(image.convert("RGB"))
    arr = np.asarray(image)
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    return arr[..., :3]


def locate_od_fovea(image: object, *, work_max_dim: int = 1024) -> AnatomyLandmarks:
    rgb_full = _as_rgb(image)
    h0, w0 = rgb_full.shape[:2]
    scale = min(1.0, float(work_max_dim) / float(max(h0, w0)))
    if scale < 1.0:
        rgb = cv2.resize(rgb_full, (max(1, int(w0 * scale)), max(1, int(h0 * scale))), interpolation=cv2.INTER_AREA)
    else:
        rgb = rgb_full
    h, w = rgb.shape[:2]
    mask = _retina_mask(rgb)

    cols = np.where(mask.any(axis=0))[0]
    retina_w = float(cols.max() - cols.min() + 1) if cols.size else float(w)
    od_diameter = max(8.0, 0.10 * retina_w)
    sigma = od_diameter / 3.0

    # OD: brightest red-channel region inside the retina.
    red = cv2.GaussianBlur(rgb[..., 0].astype(np.float32), (0, 0), sigma)
    red_masked = np.where(mask, red, -1.0)
    od_y, od_x = np.unravel_index(int(np.argmax(red_masked)), red_masked.shape)
    ref = red[mask]
    od_confidence = float((red[od_y, od_x] - ref.mean()) / (ref.std() + 1e-6))

    # Fovea: darkest green-channel region ~2.5 OD-diameters temporal to OD.
    centre_x = w / 2.0
    temporal_sign = -1.0 if od_x > centre_x else 1.0  # OD is nasal; fovea toward centre.
    fx0 = od_x + temporal_sign * 2.5 * od_diameter
    fy0 = float(od_y)
    green = cv2.GaussianBlur(rgb[..., 1].astype(np.float32), (0, 0), sigma)
    yy, xx = np.ogrid[:h, :w]
    window = 2.0 * od_diameter
    search = mask & (np.abs(xx - fx0) <= window) & (np.abs(yy - fy0) <= window)
    if search.any():
        cand = np.where(search, green, np.inf)
        fov_y, fov_x = np.unravel_index(int(np.argmin(cand)), cand.shape)
        ref_g = green[search]
        fovea_confidence = float((ref_g.mean() - green[fov_y, fov_x]) / (ref_g.std() + 1e-6))
    else:
        fov_x, fov_y, fovea_confidence = fx0, fy0, 0.0

    inv = 1.0 / scale
    return AnatomyLandmarks(
        od_xy=(float(od_x) * inv, float(od_y) * inv),
        od_confidence=od_confidence,
        od_diameter=od_diameter * inv,
        fovea_xy=(float(fov_x) * inv, float(fov_y) * inv),
        fovea_confidence=fovea_confidence,
        image_size=(w0, h0),
    )
