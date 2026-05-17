from __future__ import annotations

import cv2
import numpy as np

LESION_CODES = ("MA", "HE", "EX", "SE")

_LESION_DIR = {
    "MA": "1. Microaneurysms",
    "HE": "2. Haemorrhages",
    "EX": "3. Hard Exudates",
    "SE": "4. Soft Exudates",
}

_MAPLES_LESION_DIR = {
    "MA": "Microaneurysms",
    "HE": "Hemorrhages",
    "EX": "Exudates",
    "SE": "CottonWoolSpots",
}


def load_lesion_masks(
    mask_base_dir,
    image_stem: str,
    target_size: tuple[int, int] | None = None,
) -> dict[str, np.ndarray]:
    """Load IDRiD lesion masks for one image.

    Args:
        mask_base_dir: Path to the split directory
            (e.g. ".../2. All Segmentation Groundtruths/a. Training Set")
        image_stem: Filename stem, e.g. "IDRiD_01"
        target_size: (width, height) to resize masks. None keeps original size.

    Returns:
        dict of lesion_code -> uint8 binary mask (0/1).
        Only present codes are included.
    """
    from pathlib import Path

    masks: dict[str, np.ndarray] = {}
    for code, subdir in _LESION_DIR.items():
        path = Path(mask_base_dir) / subdir / f"{image_stem}_{code}.tif"
        if not path.exists():
            continue
        arr = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if arr is None:
            continue
        binary = (arr > 0).astype(np.uint8)
        if target_size is not None:
            binary = cv2.resize(binary, target_size, interpolation=cv2.INTER_NEAREST)
        masks[code] = binary
    return masks


def load_maples_masks(
    annotations_dir,
    image_stem: str,
    target_size: tuple[int, int] | None = None,
) -> dict[str, np.ndarray]:
    """Load MAPLES-DR lesion masks for one MESSIDOR image.

    Args:
        annotations_dir: Path to MAPLES-DR annotations dir
            (e.g. ".../MAPLES-DR/AdditionalData/annotations")
        image_stem: Filename stem, e.g. "20051019_38557_0100_PP"
        target_size: (width, height) to resize masks. None keeps original size.

    Returns:
        dict of lesion_code -> uint8 binary mask (0/1).
        Only present codes are included.
    """
    from pathlib import Path

    masks: dict[str, np.ndarray] = {}
    for code, subdir in _MAPLES_LESION_DIR.items():
        path = Path(annotations_dir) / subdir / f"{image_stem}.png"
        if not path.exists():
            continue
        arr = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if arr is None:
            continue
        binary = (arr > 0).astype(np.uint8)
        if target_size is not None:
            binary = cv2.resize(binary, target_size, interpolation=cv2.INTER_NEAREST)
        masks[code] = binary
    return masks


def union_mask(masks: dict[str, np.ndarray]) -> np.ndarray | None:
    """Element-wise OR of all masks. Returns None for empty dict."""
    result: np.ndarray | None = None
    for m in masks.values():
        result = m if result is None else np.logical_or(result, m).astype(np.uint8)
    return result


def normalize_cam_fov(cam: np.ndarray, retina_mask: np.ndarray) -> np.ndarray:
    """Min-max normalize CAM within the retina FOV, zero outside.

    Removes score calibration bias so different XAI methods are comparable
    on the same threshold scale (Choe et al., CVPR 2020).
    """
    out = np.zeros_like(cam, dtype=np.float32)
    active = cam[retina_mask > 0]
    if active.size == 0:
        return out
    lo, hi = float(active.min()), float(active.max())
    if hi - lo < 1e-8:
        return out
    out[retina_mask > 0] = (cam[retina_mask > 0] - lo) / (hi - lo)
    return out


def binarize_cam(
    cam: np.ndarray,
    retina_mask: np.ndarray | None = None,
    top_percent: float = 0.20,
) -> np.ndarray:
    """Threshold CAM at the (1 - top_percent) percentile of retina pixels.

    Args:
        cam: 2-D float array [H, W], values in [0, 1].
        retina_mask: Binary mask of retina area (1 = retina). If None, uses
            the whole image.
        top_percent: Fraction of retina pixels to select (default 0.20).

    Returns:
        uint8 binary mask (1 = high activation).
    """
    if retina_mask is None:
        retina_mask = np.ones_like(cam, dtype=np.uint8)

    active = cam[retina_mask > 0]
    if active.size == 0:
        return np.zeros_like(cam, dtype=np.uint8)

    threshold = float(np.percentile(active, (1.0 - top_percent) * 100.0))
    return ((cam >= threshold) & (retina_mask > 0)).astype(np.uint8)


def compute_iou(pred: np.ndarray, gt: np.ndarray) -> float | None:
    """IoU between two binary masks. Returns None when union is 0."""
    intersection = float(np.logical_and(pred, gt).sum())
    union_area = float(np.logical_or(pred, gt).sum())
    if union_area == 0:
        return None
    return intersection / union_area


def compute_auprc(
    cam: np.ndarray,
    gt_mask: np.ndarray,
    retina_mask: np.ndarray | None = None,
) -> float | None:
    """Pixel-level AUPRC: CAM as continuous ranking score vs GT lesion mask.

    Treats each retina pixel as an instance. A perfect localizer scores 1.0;
    a random baseline scores approximately lesion_area / retina_area.

    Returns None when gt_mask has no positive pixels.
    """
    from sklearn.metrics import average_precision_score

    if retina_mask is not None:
        scores = cam[retina_mask > 0].ravel().astype(np.float64)
        labels = gt_mask[retina_mask > 0].ravel().astype(np.int32)
    else:
        scores = cam.ravel().astype(np.float64)
        labels = gt_mask.ravel().astype(np.int32)

    if labels.sum() == 0:
        return None
    return float(average_precision_score(labels, scores))


def compute_auc_iou(
    cam: np.ndarray,
    retina_mask: np.ndarray,
    gt_mask: np.ndarray,
    n_thresholds: int = 50,
) -> float | None:
    """Mean IoU across a uniform threshold sweep (AUC-IoU proxy).

    Removes threshold choice bias: instead of a single top-k% cutoff, sweeps
    the full range of CAM values and averages IoU (Choe et al., CVPR 2020).

    Returns None when gt_mask has no positive pixels.
    """
    if gt_mask.sum() == 0:
        return None
    active = cam[retina_mask > 0]
    if active.size == 0:
        return None
    lo, hi = float(active.min()), float(active.max())
    if hi - lo < 1e-8:
        return None
    thresholds = np.linspace(lo, hi, n_thresholds)
    ious = []
    for t in thresholds:
        binary = ((cam >= t) & (retina_mask > 0)).astype(np.uint8)
        iou = compute_iou(binary, gt_mask)
        ious.append(iou if iou is not None else 0.0)
    return float(np.mean(ious))


def pointing_game(cam: np.ndarray, gt: np.ndarray) -> bool | None:
    """True if the CAM argmax pixel falls inside the GT mask.

    Returns None when gt has no positive pixels.
    """
    if gt.sum() == 0:
        return None
    peak_idx = int(np.argmax(cam))
    h, w = cam.shape
    peak_y, peak_x = divmod(peak_idx, w)
    return bool(gt[peak_y, peak_x] > 0)


# ---------------------------------------------------------------------------
# Baseline CAM generators (for contextualization)
# ---------------------------------------------------------------------------

def make_random_cam(
    retina_mask: np.ndarray,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Uniform random heatmap within retina FOV."""
    if rng is None:
        rng = np.random.default_rng(0)
    cam = np.zeros(retina_mask.shape, dtype=np.float32)
    cam[retina_mask > 0] = rng.random(int((retina_mask > 0).sum())).astype(np.float32)
    return cam


def make_center_gaussian_cam(retina_mask: np.ndarray) -> np.ndarray:
    """2-D Gaussian centered at image center, σ = min(H,W)/4, masked to FOV."""
    h, w = retina_mask.shape
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    sigma = float(min(h, w)) / 4.0
    cam = np.exp(-((ys - h / 2) ** 2 + (xs - w / 2) ** 2) / (2 * sigma ** 2))
    cam = cam.astype(np.float32)
    cam[retina_mask == 0] = 0.0
    return cam


def make_retina_uniform_cam(retina_mask: np.ndarray) -> np.ndarray:
    """Constant 1.0 within retina FOV — lower bound for any threshold-based metric."""
    cam = np.zeros(retina_mask.shape, dtype=np.float32)
    cam[retina_mask > 0] = 1.0
    return cam
