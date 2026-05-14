"""Heatmap vs GT lesion mask visualization for a single IDRiD image.

Usage
-----
    python visualize_heatmap_gt.py --config configs/base.yaml --image IDRiD_01
    python visualize_heatmap_gt.py --config configs/base.yaml --image IDRiD_05 --split test
    python visualize_heatmap_gt.py --config configs/base.yaml --image IDRiD_01 --top-percent 0.15

Output
------
    artifacts/heatmaps/heatmap_vs_gt_{image_id}.png
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))

from drscreen.infer.service import InferenceSession
from drscreen.xai.gradcam import generate_gradcam
from drscreen.xai.iou import (
    LESION_CODES,
    binarize_cam,
    load_lesion_masks,
    normalize_cam_fov,
    union_mask,
)

_SPLIT_IMAGE_SUBDIR = {
    "train": "a. Training Set",
    "test": "b. Testing Set",
}

_LESION_COLORS = {
    "MA": (1.0, 0.2, 0.2),
    "HE": (0.2, 0.2, 1.0),
    "EX": (1.0, 0.9, 0.1),
    "SE": (0.2, 0.9, 0.2),
}


def _build_retina_mask(image: Image.Image) -> np.ndarray:
    rgb = np.asarray(image.convert("RGB"))
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    _, thresh = cv2.threshold(gray, 8, 255, cv2.THRESH_BINARY)
    min_dim = min(gray.shape[:2])
    ks = max(3, min(11, (min_dim // 50) * 2 + 1))
    kernel = np.ones((ks, ks), dtype=np.uint8)
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(cleaned)
    if n <= 1:
        return np.ones(gray.shape, dtype=np.uint8)
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return (labels == largest).astype(np.uint8)


def _overlay_heatmap(base_rgb: np.ndarray, cam_norm: np.ndarray) -> np.ndarray:
    heat = plt.cm.jet(cam_norm)[..., :3]
    heat_uint8 = (heat * 255).astype(np.uint8)
    blended = cv2.addWeighted(base_rgb, 0.5, heat_uint8, 0.5, 0)
    return blended


def _overlay_masks(base_rgb: np.ndarray, masks: dict[str, np.ndarray]) -> np.ndarray:
    out = base_rgb.astype(np.float32).copy()
    for code, mask in masks.items():
        if mask is None or not mask.any():
            continue
        color = _LESION_COLORS.get(code, (1.0, 1.0, 1.0))
        for c, v in enumerate(color):
            out[..., c] = np.where(mask > 0, out[..., c] * 0.35 + v * 255 * 0.65, out[..., c])
    return np.clip(out, 0, 255).astype(np.uint8)


def _overlay_binary_cam(base_rgb: np.ndarray, binary: np.ndarray) -> np.ndarray:
    out = base_rgb.astype(np.float32).copy()
    color = (0.0, 1.0, 1.0)
    for c, v in enumerate(color):
        out[..., c] = np.where(binary > 0, out[..., c] * 0.35 + v * 255 * 0.65, out[..., c])
    return np.clip(out, 0, 255).astype(np.uint8)


def visualize(
    config_path: str,
    image_id: str = "IDRiD_01",
    split: str = "train",
    idrid_root: str | None = None,
    top_percent: float = 0.20,
    output_path: str | None = None,
) -> None:
    project_root = Path(config_path).resolve().parents[1]
    idrid_root_path = Path(idrid_root) if idrid_root else project_root / "data" / "raw" / "IDRiD"

    split_subdir = _SPLIT_IMAGE_SUBDIR[split]
    image_dir = idrid_root_path / "A. Segmentation" / "1. Original Images" / split_subdir
    mask_base_dir = idrid_root_path / "A. Segmentation" / "2. All Segmentation Groundtruths" / split_subdir

    image_path = image_dir / f"{image_id}.jpg"
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    session = InferenceSession.from_config_path(config_path)
    session.preprocessor = None

    version = session.config.get("project", {}).get("version", "unknown")
    gradcam_method = session.config.get("infer", {}).get("gradcam_method", "layercam")
    target_block = session.config.get("infer", {}).get("gradcam_target_block", None)

    target_layer = None
    if target_block is not None:
        blocks = getattr(session.model, "blocks", getattr(session.model, "features", None))
        if blocks is not None:
            target_layer = blocks[int(target_block)]

    pil_image = Image.open(image_path).convert("RGB")
    image_tensor = session.eval_transform(pil_image).to(session.device)
    h, w = image_tensor.shape[-2], image_tensor.shape[-1]

    gradcam = generate_gradcam(
        session.model,
        image_tensor.unsqueeze(0),
        method=gradcam_method,
        target_layer=target_layer,
    )
    cam_raw = gradcam.heatmap[0].detach().cpu().numpy()

    pil_resized = pil_image.resize((w, h), Image.BILINEAR)
    base_rgb = np.asarray(pil_resized.convert("RGB"))
    retina_mask = _build_retina_mask(pil_resized)
    cam_norm = normalize_cam_fov(cam_raw, retina_mask)
    binary_cam = binarize_cam(cam_norm, retina_mask, top_percent=top_percent)

    gt_masks = load_lesion_masks(mask_base_dir, image_id, target_size=(w, h))
    gt_union = union_mask(gt_masks)

    present = [c for c in LESION_CODES if c in gt_masks]
    print(f"Image    : {image_id}  ({split})")
    print(f"Version  : {version}  method={gradcam_method}")
    print(f"GT masks : {present}")
    print(f"Top-%    : {int(top_percent*100)}%")

    # ── Compute quick IoU for title ──────────────────────────────────────────
    if gt_union is not None and binary_cam.any():
        inter = float((binary_cam & gt_union).sum())
        union = float((binary_cam | gt_union).sum())
        iou = inter / union if union > 0 else 0.0
    else:
        iou = None

    # ── Build figure (5 panels) ──────────────────────────────────────────────
    fig, axes = plt.subplots(1, 5, figsize=(25, 5))
    fig.suptitle(
        f"{image_id} | {version} | {gradcam_method} | top-{int(top_percent*100)}%"
        + (f" | IoU={iou:.4f}" if iou is not None else ""),
        fontsize=12, fontweight="bold",
    )

    panels = [
        ("Original", base_rgb),
        (f"LayerCAM overlay\n({gradcam_method})", _overlay_heatmap(base_rgb, cam_norm)),
        (f"Binary CAM top-{int(top_percent*100)}%\n(cyan = activated)", _overlay_binary_cam(base_rgb, binary_cam)),
        ("GT lesion masks\n(MA=red HE=blue EX=yellow SE=green)", _overlay_masks(base_rgb, gt_masks)),
        ("CAM binary vs GT union\n(cyan=cam, colored=GT)", _overlay_binary_cam(
            _overlay_masks(base_rgb, gt_masks), binary_cam
        )),
    ]

    for ax, (title, img) in zip(axes, panels):
        ax.imshow(img)
        ax.set_title(title, fontsize=9)
        ax.axis("off")

    legend_handles = [
        mpatches.Patch(color=_LESION_COLORS[c], label=c)
        for c in LESION_CODES if c in gt_masks
    ]
    legend_handles.append(mpatches.Patch(color=(0.0, 1.0, 1.0), label=f"CAM top-{int(top_percent*100)}%"))
    axes[-1].legend(handles=legend_handles, loc="lower right", fontsize=7, framealpha=0.7)

    plt.tight_layout()

    if output_path is None:
        out_dir = project_root / "artifacts" / "heatmaps"
        out_dir.mkdir(parents=True, exist_ok=True)
        output_path = str(out_dir / f"heatmap_vs_gt_{image_id}_{version}.png")

    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved: {output_path}")
    plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(description="Heatmap vs GT lesion mask visualization")
    parser.add_argument("--config", required=True, help="Path to model config YAML")
    parser.add_argument("--image", default="IDRiD_01", help="Image stem (default: IDRiD_01)")
    parser.add_argument("--split", default="train", choices=["train", "test"])
    parser.add_argument("--idrid-root", help="Override IDRiD root directory")
    parser.add_argument("--top-percent", type=float, default=0.20,
                        help="Top-N%% CAM binarization threshold (default: 0.20)")
    parser.add_argument("--output", help="Output PNG path")
    args = parser.parse_args()

    visualize(
        config_path=args.config,
        image_id=args.image,
        split=args.split,
        idrid_root=args.idrid_root,
        top_percent=args.top_percent,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
