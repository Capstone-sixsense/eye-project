"""Lesion evidence prediction vs GT visualization for one IDRiD image.

This script keeps the deployed inference path intact while making the
presentation background easier to read:

1. Backend QuickQual crop/pad/resize is applied.
2. AI inference still uses FundusPreprocess, including Ben Graham normalization.
3. The displayed fundus background uses the same AI crop/resize geometry but
   skips Ben Graham normalization.
4. Prediction, GT, and comparison masks are drawn in that shared coordinate
   frame.

Usage
-----
    python visualize_lesion_pred_gt.py --config configs/base.yaml --image IDRiD_25
    python visualize_lesion_pred_gt.py --config configs/base.yaml --image IDRiD_25 --split train
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageOps

sys.path.insert(0, str(Path(__file__).resolve().parent))
for _candidate in (
    Path(__file__).resolve().parents[1] / "backend",
    Path("/app"),
):
    if _candidate.exists():
        sys.path.insert(0, str(_candidate))

from drscreen.infer.service import InferenceSession, _build_retina_mask
from drscreen.xai.iou import LESION_CODES

try:
    from models.quickqual_wrapper import preprocess_fundus_image as _backend_quickqual_preprocess
except Exception:  # pragma: no cover - fallback for non-backend environments.
    _backend_quickqual_preprocess = None


_SPLIT_IMAGE_SUBDIR = {
    "train": "a. Training Set",
    "test": "b. Testing Set",
}

_LESION_DIR = {
    "MA": "1. Microaneurysms",
    "HE": "2. Haemorrhages",
    "EX": "3. Hard Exudates",
    "SE": "4. Soft Exudates",
}

_LESION_COLORS = {
    "MA": np.array([255.0, 32.0, 20.0], dtype=np.float32),
    "HE": np.array([33.0, 67.0, 235.0], dtype=np.float32),
    "EX": np.array([255.0, 224.0, 0.0], dtype=np.float32),
    "SE": np.array([35.0, 205.0, 60.0], dtype=np.float32),
}

_PRED_ONLY = np.array([0.0, 220.0, 235.0], dtype=np.float32)
_GT_ONLY = np.array([240.0, 0.0, 205.0], dtype=np.float32)
_OVERLAP = np.array([255.0, 255.0, 255.0], dtype=np.float32)


def _quickqual_crop_padding(
    image: Image.Image,
    *,
    threshold: int = 15,
) -> tuple[tuple[int, int, int, int], tuple[int, int, int, int]] | None:
    arr = np.asarray(image.convert("RGB"))
    mean = arr.mean(-1)
    rows = np.where(mean > threshold)[0]
    cols = np.where(mean > threshold)[1]
    if rows.size == 0 or cols.size == 0:
        return None

    top, bottom = int(rows.min()), int(rows.max())
    left, right = int(cols.min()), int(cols.max())

    buffer = 20
    left = max(0, left - buffer)
    right = min(arr.shape[1], right + buffer)
    top = max(0, top - buffer)
    bottom = min(arr.shape[0], bottom + buffer)

    crop_w = right - left
    crop_h = bottom - top
    if crop_w > crop_h:
        pad = crop_w - crop_h
        padding = (0, pad // 2, 0, pad - pad // 2)
    else:
        pad = crop_h - crop_w
        padding = (pad // 2, 0, pad - pad // 2, 0)
    return (left, top, right, bottom), padding


def _quickqual_preprocess(image: Image.Image) -> Image.Image:
    if _backend_quickqual_preprocess is not None:
        return _backend_quickqual_preprocess(image)

    geometry = _quickqual_crop_padding(image)
    if geometry is None:
        return image.convert("RGB").resize((1024, 1024), Image.Resampling.LANCZOS)

    crop_box, padding = geometry
    cropped = image.convert("RGB").crop(crop_box)
    padded = ImageOps.expand(cropped, border=padding, fill=0)
    return padded.resize((1024, 1024), Image.Resampling.LANCZOS)


def _apply_quickqual_mask_geometry(mask_hwc: np.ndarray, reference: Image.Image) -> np.ndarray:
    geometry = _quickqual_crop_padding(reference)
    channels: list[np.ndarray] = []
    for idx in range(mask_hwc.shape[-1]):
        mask = Image.fromarray((mask_hwc[..., idx].astype(np.uint8) * 255), mode="L")
        if geometry is None:
            transformed = mask.resize((1024, 1024), Image.Resampling.NEAREST)
        else:
            crop_box, padding = geometry
            transformed = ImageOps.expand(mask.crop(crop_box), border=padding, fill=0)
            transformed = transformed.resize((1024, 1024), Image.Resampling.NEAREST)
        channels.append((np.asarray(transformed) > 0).astype(np.uint8))
    return np.stack(channels, axis=-1)


def _apply_ai_image_geometry_without_ben_graham(
    image: Image.Image,
    preprocessor: object | None,
    *,
    output_size: int,
) -> Image.Image:
    if preprocessor is None:
        return image.convert("RGB").resize((output_size, output_size), Image.Resampling.BICUBIC)

    arr = np.asarray(image.convert("RGB")).copy()
    if bool(getattr(preprocessor, "_align", False)):
        matrix = preprocessor._alignment_matrix(arr)
        if matrix is not None:
            h, w = arr.shape[:2]
            arr = cv2.warpAffine(
                arr,
                matrix,
                (w, h),
                flags=cv2.INTER_LANCZOS4,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(0, 0, 0),
            )

    geometry = preprocessor._circular_crop_geometry(arr)
    if geometry is not None:
        x1, y1, x2, y2, pad_top, pad_bottom, pad_left, pad_right = geometry
        arr = arr[y1:y2, x1:x2]
        arr = cv2.copyMakeBorder(
            arr,
            pad_top,
            pad_bottom,
            pad_left,
            pad_right,
            cv2.BORDER_CONSTANT,
            value=(0, 0, 0),
        )

    result = Image.fromarray(arr)
    if result.size != (output_size, output_size):
        result = result.resize((output_size, output_size), Image.Resampling.BICUBIC)
    return result


def _resize_probability_channels(prob_chw: np.ndarray, shape_hw: tuple[int, int]) -> np.ndarray:
    h, w = shape_hw
    if prob_chw.shape[-2:] == (h, w):
        return prob_chw.astype(np.float32)
    return np.stack(
        [
            cv2.resize(channel.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
            for channel in prob_chw
        ],
        axis=0,
    )


def _blend(base_rgb: np.ndarray, mask: np.ndarray, color: np.ndarray, alpha: float = 0.72) -> np.ndarray:
    out = base_rgb.astype(np.float32).copy()
    active = mask.astype(bool)
    if active.any():
        out[active] = out[active] * (1.0 - alpha) + color * alpha
    return np.clip(out, 0.0, 255.0).astype(np.uint8)


def _overlay_multiclass(base_rgb: np.ndarray, masks_chw: np.ndarray, *, alpha: float = 0.72) -> np.ndarray:
    out = base_rgb.copy()
    for idx, code in enumerate(LESION_CODES):
        out = _blend(out, masks_chw[idx], _LESION_COLORS[code], alpha=alpha)
    return out


def _overlay_probability(base_rgb: np.ndarray, union_prob: np.ndarray, fundus_mask: np.ndarray) -> np.ndarray:
    cmap = plt.get_cmap("magma")
    heat_rgb = (cmap(np.clip(union_prob, 0.0, 1.0))[..., :3] * 255.0).astype(np.float32)

    active_values = union_prob[fundus_mask]
    if active_values.size == 0 or float(active_values.max()) <= 0.0:
        return base_rgb.copy()

    # Keep low-probability regions mostly transparent so the fundus stays readable.
    floor = min(0.15, max(0.02, float(np.quantile(active_values, 0.75))))
    alpha = np.clip((union_prob - floor) / max(1.0 - floor, 1e-6), 0.0, 1.0)
    alpha = np.power(alpha, 0.55) * 0.78
    alpha = np.where(fundus_mask, alpha, 0.0)[..., None]

    out = base_rgb.astype(np.float32) * (1.0 - alpha) + heat_rgb * alpha
    return np.clip(out, 0.0, 255.0).astype(np.uint8)


def _binary_metrics(pred: np.ndarray, gt: np.ndarray) -> dict[str, float | int | None]:
    pred_bool = pred.astype(bool)
    gt_bool = gt.astype(bool)
    intersection = int(np.logical_and(pred_bool, gt_bool).sum())
    union = int(np.logical_or(pred_bool, gt_bool).sum())
    pred_pixels = int(pred_bool.sum())
    gt_pixels = int(gt_bool.sum())
    return {
        "iou": float(intersection / union) if union else None,
        "dice": float((2 * intersection) / (pred_pixels + gt_pixels)) if pred_pixels + gt_pixels else None,
        "precision": float(intersection / pred_pixels) if pred_pixels else None,
        "recall": float(intersection / gt_pixels) if gt_pixels else None,
        "intersection_pixels": intersection,
        "union_pixels": union,
        "pred_pixels": pred_pixels,
        "gt_pixels": gt_pixels,
    }


def _load_idrid_masks(mask_base_dir: Path, image_id: str) -> np.ndarray:
    channels: list[np.ndarray] = []
    for code in LESION_CODES:
        mask_path = mask_base_dir / _LESION_DIR[code] / f"{image_id}_{code}.tif"
        if not mask_path.exists():
            raise FileNotFoundError(f"Missing IDRiD lesion mask: {mask_path}")
        arr = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if arr is None:
            raise ValueError(f"Could not read lesion mask: {mask_path}")
        channels.append((arr > 0).astype(np.uint8))
    return np.stack(channels, axis=-1)


def visualize(
    *,
    config_path: str | Path,
    image_id: str,
    split: str = "train",
    idrid_root: str | Path | None = None,
    output_path: str | Path | None = None,
    json_output_path: str | Path | None = None,
) -> tuple[Path, Path]:
    config_path = Path(config_path).resolve()
    project_root = config_path.parents[1]
    idrid_root_path = Path(idrid_root).resolve() if idrid_root else project_root / "data" / "raw" / "IDRiD"
    split_subdir = _SPLIT_IMAGE_SUBDIR[split]

    image_path = idrid_root_path / "A. Segmentation" / "1. Original Images" / split_subdir / f"{image_id}.jpg"
    mask_base_dir = (
        idrid_root_path
        / "A. Segmentation"
        / "2. All Segmentation Groundtruths"
        / split_subdir
    )
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    session = InferenceSession.from_config_path(config_path)
    version = str(session.config.get("project", {}).get("version", "unknown"))
    infer_cfg = session.config.get("infer", {})
    lesion_threshold = float(infer_cfg.get("lesion_threshold", 0.5))
    decision_threshold = float(session.decision_threshold)
    output_size = int(session.config.get("data", {}).get("image_size", 512))

    raw_image = Image.open(image_path).convert("RGB")
    quickqual_image = _quickqual_preprocess(raw_image)
    model_input_image = (
        session.preprocessor(quickqual_image)
        if session.preprocessor is not None
        else quickqual_image.resize((output_size, output_size), Image.Resampling.BICUBIC)
    )
    display_image = _apply_ai_image_geometry_without_ben_graham(
        quickqual_image,
        session.preprocessor,
        output_size=output_size,
    )
    display_rgb = np.asarray(display_image.convert("RGB"))
    fundus_mask = _build_retina_mask(display_image).astype(bool)

    raw_masks_hwc = _load_idrid_masks(mask_base_dir, image_id)
    quickqual_masks_hwc = _apply_quickqual_mask_geometry(raw_masks_hwc, raw_image)
    if session.preprocessor is not None:
        ai_masks_hwc = session.preprocessor.apply_mask_geometry(
            quickqual_masks_hwc,
            quickqual_image,
            output_size=output_size,
        )
    else:
        ai_masks_hwc = np.stack(
            [
                cv2.resize(
                    quickqual_masks_hwc[..., idx],
                    (output_size, output_size),
                    interpolation=cv2.INTER_NEAREST,
                )
                for idx in range(quickqual_masks_hwc.shape[-1])
            ],
            axis=-1,
        )
    gt_chw_unclipped = np.moveaxis((ai_masks_hwc > 0), -1, 0)[: len(LESION_CODES)]
    gt_chw = gt_chw_unclipped & fundus_mask[None, ...]

    image_tensor = session.eval_transform(model_input_image).to(session.device)
    with torch.inference_mode():
        if bool(infer_cfg.get("use_meta_classifier", False)) and hasattr(session.model, "predict_fusion_score"):
            fusion_output = session.model.predict_fusion_score(image_tensor.unsqueeze(0))
            seg_prob = fusion_output["seg_prob"][0].detach().cpu().float().numpy()
            abnormal_probability = float(fusion_output["meta_probability"])
        else:
            seg_prob = session.model.predict_seg(image_tensor.unsqueeze(0))[0].detach().cpu().float().numpy()
            abnormal_probability = None

    prob_chw = _resize_probability_channels(np.clip(seg_prob, 0.0, 1.0), display_rgb.shape[:2])
    prob_chw = prob_chw[: len(LESION_CODES)] * fundus_mask[None, ...]
    union_prob = prob_chw.max(axis=0)
    pred_chw = (prob_chw >= lesion_threshold) & fundus_mask[None, ...]
    pred_union = pred_chw.any(axis=0)
    gt_union = gt_chw.any(axis=0)

    union_metrics = _binary_metrics(pred_union, gt_union)
    per_class_metrics = {
        code: _binary_metrics(pred_chw[idx], gt_chw[idx])
        for idx, code in enumerate(LESION_CODES)
    }
    dice_values = [m["dice"] for m in per_class_metrics.values() if m["dice"] is not None]
    iou_values = [m["iou"] for m in per_class_metrics.values() if m["iou"] is not None]
    mdice = float(np.mean(dice_values)) if dice_values else None
    miou = float(np.mean(iou_values)) if iou_values else None

    probability_panel = _overlay_probability(display_rgb, union_prob, fundus_mask)
    pred_panel = _overlay_multiclass(display_rgb, pred_chw, alpha=0.78)
    gt_panel = _overlay_multiclass(display_rgb, gt_chw, alpha=0.78)
    comparison_panel = display_rgb.copy()
    comparison_panel = _blend(comparison_panel, gt_union & ~pred_union, _GT_ONLY, alpha=0.78)
    comparison_panel = _blend(comparison_panel, pred_union & ~gt_union, _PRED_ONLY, alpha=0.78)
    comparison_panel = _blend(comparison_panel, pred_union & gt_union, _OVERLAP, alpha=0.88)

    fig, axes = plt.subplots(1, 5, figsize=(24, 5.05), dpi=160)
    fig.suptitle(
        f"{image_id} | {version} | non-Ben-Graham display | decision={decision_threshold:.2f}"
        f" | lesion thr={lesion_threshold:.2f} | clipped IoU={(union_metrics['iou'] or 0):.4f}"
        f" | Dice={(union_metrics['dice'] or 0):.4f}",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )

    panels = [
        (
            "Display fundus\n(QuickQual + AI geometry, no Ben Graham)",
            display_rgb,
        ),
        (
            "Lesion probability overlay\n(magma; low scores transparent)",
            probability_panel,
        ),
        (
            f"Binary prediction\n(config lesion_threshold={lesion_threshold:.2f}; pixels={union_metrics['pred_pixels']})",
            pred_panel,
        ),
        (
            "GT lesion masks\n(fundus clipped)",
            gt_panel,
        ),
        (
            "Prediction union vs GT union\n(cyan=pred only, magenta=GT only, white=overlap)",
            comparison_panel,
        ),
    ]
    for ax, (title, panel) in zip(axes, panels):
        ax.imshow(panel)
        ax.set_title(title, fontsize=10)
        ax.axis("off")

    if union_prob[fundus_mask].size and float(union_prob[fundus_mask].max()) > 0.0:
        contour_mask = (union_prob >= lesion_threshold) & fundus_mask
        if contour_mask.any():
            axes[1].contour(contour_mask, levels=[0.5], colors="white", linewidths=0.8)

    colorbar_image = axes[1].imshow(
        np.ma.masked_where(~fundus_mask, union_prob),
        cmap="magma",
        vmin=0.0,
        vmax=1.0,
        alpha=0.0,
    )
    colorbar = fig.colorbar(colorbar_image, ax=axes[1], fraction=0.046, pad=0.02)
    colorbar.set_label("lesion probability", fontsize=8)
    colorbar.ax.tick_params(labelsize=7)

    legend_handles = [
        mpatches.Patch(facecolor=_LESION_COLORS["MA"] / 255.0, label="MA"),
        mpatches.Patch(facecolor=_LESION_COLORS["HE"] / 255.0, label="HE"),
        mpatches.Patch(facecolor=_LESION_COLORS["EX"] / 255.0, label="EX"),
        mpatches.Patch(facecolor=_LESION_COLORS["SE"] / 255.0, label="SE"),
        mpatches.Patch(facecolor=_PRED_ONLY / 255.0, label="Pred only"),
        mpatches.Patch(facecolor=_GT_ONLY / 255.0, label="GT only"),
        mpatches.Patch(facecolor=_OVERLAP / 255.0, label="Overlap"),
    ]
    axes[-1].legend(handles=legend_handles, loc="lower right", fontsize=7, framealpha=0.88)

    fig.tight_layout(rect=[0, 0, 1, 0.93])

    if output_path is None:
        out_dir = project_root / "artifacts" / "heatmaps"
        out_dir.mkdir(parents=True, exist_ok=True)
        output_path = out_dir / (
            f"lesion_pred_vs_gt_{image_id}_{version}_non_bengraham_overlay_compare.png"
        )
    else:
        output_path = Path(output_path).resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)

    if json_output_path is None:
        json_output_path = Path(output_path).with_suffix(".json")
    else:
        json_output_path = Path(json_output_path).resolve()
        json_output_path.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)

    active_values = union_prob[fundus_mask]
    payload = {
        "image_id": image_id,
        "split": split,
        "version": version,
        "model": type(session.model).__name__,
        "active_config_path": str(config_path),
        "checkpoint_path": str(session.checkpoint_path),
        "visualization_background": "QuickQual + active AI geometry, Ben Graham skipped",
        "model_input": "QuickQual + active AI FundusPreprocess including Ben Graham",
        "mask_pipeline": "IDRiD GT -> QuickQual geometry -> active AI geometry -> fundus clip",
        "thresholds": {
            "classification_decision_threshold": decision_threshold,
            "lesion_segmentation_threshold": lesion_threshold,
            "lesion_threshold_source": "infer.lesion_threshold",
        },
        "classification": {
            "abnormal_probability": abnormal_probability,
        },
        "size_check": {
            "raw_image_size_wh": list(raw_image.size),
            "quickqual_image_size_wh": list(quickqual_image.size),
            "display_image_shape_hwc": list(display_rgb.shape),
            "model_input_size_wh": list(model_input_image.size),
            "probability_shape_chw": list(prob_chw.shape),
            "prediction_shape_chw": list(pred_chw.shape),
            "gt_shape_chw": list(gt_chw.shape),
            "same_spatial_size": bool(
                pred_chw.shape[-2:] == gt_chw.shape[-2:] == display_rgb.shape[:2]
            ),
            "gt_union_outside_fundus_pixels_before_clip": int(
                gt_chw_unclipped.any(axis=0).sum() - gt_union.sum()
            ),
        },
        "prob_stats": {
            "max": float(active_values.max()) if active_values.size else 0.0,
            "mean": float(active_values.mean()) if active_values.size else 0.0,
            "p99": float(np.quantile(active_values, 0.99)) if active_values.size else 0.0,
            "p999": float(np.quantile(active_values, 0.999)) if active_values.size else 0.0,
        },
        "foreground_clipped_metrics": {
            "union_iou": union_metrics["iou"],
            "union_dice": union_metrics["dice"],
            "mdice": mdice,
            "miou": miou,
            "union_precision": union_metrics["precision"],
            "union_recall": union_metrics["recall"],
            "intersection_pixels": union_metrics["intersection_pixels"],
            "union_pixels": union_metrics["union_pixels"],
            "pred_union_pixels": union_metrics["pred_pixels"],
            "gt_union_pixels": union_metrics["gt_pixels"],
            "per_class": per_class_metrics,
        },
        "outputs": {
            "png": str(output_path),
            "json": str(json_output_path),
        },
    }
    json_output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(json.dumps(payload, indent=2))
    return Path(output_path), Path(json_output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize deployed lesion evidence vs IDRiD GT")
    parser.add_argument("--config", required=True, help="Path to model config YAML")
    parser.add_argument("--image", default="IDRiD_25", help="Image stem, e.g. IDRiD_25")
    parser.add_argument("--split", default="train", choices=sorted(_SPLIT_IMAGE_SUBDIR))
    parser.add_argument("--idrid-root", help="Override IDRiD root directory")
    parser.add_argument("--output", help="Output PNG path")
    parser.add_argument("--json-output", help="Output JSON path")
    args = parser.parse_args()

    visualize(
        config_path=args.config,
        image_id=args.image,
        split=args.split,
        idrid_root=args.idrid_root,
        output_path=args.output,
        json_output_path=args.json_output,
    )


if __name__ == "__main__":
    main()
