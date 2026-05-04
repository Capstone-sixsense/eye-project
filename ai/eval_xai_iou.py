"""XAI quantitative validation: Layer-CAM vs IDRiD lesion masks.

Generates Layer-CAM (or Grad-CAM) for each IDRiD segmentation image,
binarizes the heatmap at the top-N% threshold, and computes IoU against
ground truth lesion masks (MA / HE / EX / SE) and a pointing-game score.

FundusPreprocess is intentionally skipped: v21 was trained on unprocessed
images (data.use_preprocessing=false), so raw resize keeps the spatial
layout consistent between the CAM and GT masks.

Usage
-----
    python eval_xai_iou.py --config configs/v21_512_layercam.yaml

    # custom threshold sweep
    python eval_xai_iou.py --config configs/v21_512_layercam.yaml \\
        --top-percents 0.10 0.20 0.30

    # evaluate test split instead of training
    python eval_xai_iou.py --config configs/v21_512_layercam.yaml \\
        --split test

Output
------
    artifacts/evaluations/xai_iou_{version}_{split}.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))

from drscreen.infer.service import InferenceSession
from drscreen.xai.gradcam import generate_gradcam
from drscreen.xai.iou import (
    LESION_CODES,
    binarize_cam,
    compute_iou,
    load_lesion_masks,
    pointing_game,
    union_mask,
)


_SPLIT_IMAGE_SUBDIR = {
    "train": "a. Training Set",
    "test": "b. Testing Set",
}


def _build_retina_mask(image: Image.Image) -> np.ndarray:
    """Binary retina foreground mask (1 = retina pixel)."""
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


def _process_image(
    session: InferenceSession,
    image_path: Path,
    mask_base_dir: Path,
    gradcam_method: str,
    top_percents: list[float],
    target_layer=None,
    use_seg_head: bool = False,
) -> dict:
    image_stem = image_path.stem  # e.g. "IDRiD_01"

    # Load image — skip FundusPreprocess for spatial alignment with GT masks
    pil_image = Image.open(image_path).convert("RGB")
    image_tensor = session.eval_transform(pil_image).to(session.device)

    cam_h, cam_w = image_tensor.shape[-2], image_tensor.shape[-1]

    if use_seg_head:
        # Use seg_head sigmoid output directly as heatmap
        seg_prob = session.model.predict_seg(image_tensor.unsqueeze(0))  # [1,1,H,W]
        cam_raw = seg_prob[0, 0].detach().cpu().numpy()  # [H, W], already in [0,1]
        # Resize to model input resolution if needed
        if cam_raw.shape != (cam_h, cam_w):
            cam_raw = cv2.resize(cam_raw, (cam_w, cam_h), interpolation=cv2.INTER_LINEAR)
        cam = cam_raw
    else:
        # Generate CAM (gradients required — no torch.no_grad)
        gradcam = generate_gradcam(
            session.model,
            image_tensor.unsqueeze(0),
            method=gradcam_method,
            target_layer=target_layer,
        )
        cam = gradcam.heatmap[0].detach().cpu().numpy()  # [H, W], float32 in [0, 1]

    # Retina mask at model input resolution
    pil_resized = pil_image.resize((cam_w, cam_h), Image.BILINEAR)
    retina_mask = _build_retina_mask(pil_resized)

    # Load GT masks resized to model input resolution
    gt_masks = load_lesion_masks(mask_base_dir, image_stem, target_size=(cam_w, cam_h))
    gt_union = union_mask(gt_masks)

    result: dict = {
        "image_id": image_stem,
        "masks_present": list(gt_masks.keys()),
        "pointing_game": None,
        "thresholds": {},
    }

    # Pointing game (threshold-independent)
    if gt_union is not None:
        result["pointing_game"] = pointing_game(cam, gt_union)

    for top_pct in top_percents:
        binary_cam = binarize_cam(cam, retina_mask, top_percent=top_pct)
        key = f"top{int(top_pct * 100):02d}"

        per_code: dict[str, float | None] = {}
        for code in LESION_CODES:
            if code in gt_masks:
                per_code[code] = compute_iou(binary_cam, gt_masks[code])
            else:
                per_code[code] = None

        iou_union = compute_iou(binary_cam, gt_union) if gt_union is not None else None

        result["thresholds"][key] = {
            "iou_union": iou_union,
            "iou_per_lesion": per_code,
        }

    return result


def _aggregate(per_image: list[dict], top_percents: list[float]) -> dict:
    agg: dict = {"pointing_game": None, "thresholds": {}}

    pg_scores = [r["pointing_game"] for r in per_image if r["pointing_game"] is not None]
    if pg_scores:
        agg["pointing_game"] = {
            "mean": float(np.mean(pg_scores)),
            "n": len(pg_scores),
        }

    for top_pct in top_percents:
        key = f"top{int(top_pct * 100):02d}"
        union_ious = [
            r["thresholds"][key]["iou_union"]
            for r in per_image
            if r["thresholds"][key]["iou_union"] is not None
        ]

        per_code_ious: dict[str, list[float]] = {c: [] for c in LESION_CODES}
        for r in per_image:
            for code in LESION_CODES:
                v = r["thresholds"][key]["iou_per_lesion"].get(code)
                if v is not None:
                    per_code_ious[code].append(v)

        agg["thresholds"][key] = {
            "mean_iou_union": float(np.mean(union_ious)) if union_ious else None,
            "n_images_with_gt": len(union_ious),
            "per_lesion": {
                code: {
                    "mean_iou": float(np.mean(vals)) if vals else None,
                    "n": len(vals),
                }
                for code, vals in per_code_ious.items()
            },
        }

    return agg


def evaluate(
    config_path: str,
    split: str = "train",
    idrid_root: str | None = None,
    top_percents: list[float] | None = None,
    target_block: int | None = None,
    output_path: str | None = None,
    use_seg_head: bool = False,
) -> dict:
    if top_percents is None:
        top_percents = [0.10, 0.20, 0.30]

    project_root = Path(config_path).resolve().parents[1]

    if idrid_root is None:
        idrid_root = project_root / "data" / "raw" / "IDRiD"
    else:
        idrid_root = Path(idrid_root)

    split_subdir = _SPLIT_IMAGE_SUBDIR.get(split)
    if split_subdir is None:
        raise ValueError(f"split must be 'train' or 'test', got {split!r}")

    image_dir = idrid_root / "A. Segmentation" / "1. Original Images" / split_subdir
    mask_base_dir = (
        idrid_root / "A. Segmentation" / "2. All Segmentation Groundtruths" / split_subdir
    )

    if not image_dir.exists():
        raise FileNotFoundError(f"IDRiD image directory not found: {image_dir}")
    if not mask_base_dir.exists():
        raise FileNotFoundError(f"IDRiD mask directory not found: {mask_base_dir}")

    session = InferenceSession.from_config_path(config_path)
    # Skip FundusPreprocess so GT masks stay spatially aligned with the CAM.
    # v21 was trained without preprocessing, so this matches the training distribution.
    session.preprocessor = None

    gradcam_method = session.config.get("infer", {}).get("gradcam_method", "gradcam")
    version = session.config.get("project", {}).get("version", "unknown")

    # Resolve target layer from block index
    target_layer = None
    block_label = "seg_head" if use_seg_head else "default"
    if not use_seg_head and target_block is not None:
        blocks = getattr(session.model, "blocks", getattr(session.model, "features", None))
        if blocks is None:
            raise ValueError("Model has neither .blocks nor .features attribute")
        target_layer = blocks[target_block]
        block_label = f"block{target_block}"

    image_paths = sorted(image_dir.glob("*.jpg")) + sorted(image_dir.glob("*.png"))
    if not image_paths:
        raise FileNotFoundError(f"No images found in {image_dir}")

    print(f"Config  : {config_path}")
    print(f"Version : {version}")
    print(f"Method  : {gradcam_method}")
    print(f"Layer   : {block_label}")
    print(f"Split   : {split}  ({len(image_paths)} images)")
    print(f"Thresholds: top {[int(p*100) for p in top_percents]}%")
    print()

    per_image: list[dict] = []
    for i, image_path in enumerate(image_paths, 1):
        rec = _process_image(
            session, image_path, mask_base_dir, gradcam_method, top_percents, target_layer,
            use_seg_head=use_seg_head,
        )

        union_iou_20 = rec["thresholds"].get("top20", {}).get("iou_union")
        iou_str = f"{union_iou_20:.4f}" if union_iou_20 is not None else "N/A"
        pg_str = str(rec["pointing_game"]) if rec["pointing_game"] is not None else "N/A"
        print(
            f"  [{i:02d}/{len(image_paths)}] {rec['image_id']}"
            f"  union_iou={iou_str}"
            f"  pg={pg_str}"
            f"  masks={rec['masks_present']}"
        )
        per_image.append(rec)

    aggregate = _aggregate(per_image, top_percents)

    print()
    print("=== Aggregate ===")
    if aggregate["pointing_game"]:
        pg = aggregate["pointing_game"]
        print(f"Pointing game : {pg['mean']:.4f}  (n={pg['n']})")
    for top_pct in top_percents:
        key = f"top{int(top_pct * 100):02d}"
        agg_t = aggregate["thresholds"][key]
        miou = agg_t["mean_iou_union"]
        miou_str = f"{miou:.4f}" if miou is not None else "N/A"
        print(f"IoU top{int(top_pct*100):02d}%    : {miou_str}  (n={agg_t['n_images_with_gt']})")

    output = {
        "version": version,
        "checkpoint_path": str(session.checkpoint_path),
        "gradcam_method": gradcam_method,
        "target_block": block_label,
        "split": split,
        "n_images": len(per_image),
        "top_percents": top_percents,
        "aggregate": aggregate,
        "per_image": per_image,
    }

    if output_path is None:
        eval_dir = project_root / "artifacts" / "evaluations"
        eval_dir.mkdir(parents=True, exist_ok=True)
        output_path = str(eval_dir / f"xai_iou_{version}_{block_label}_{split}.json")

    Path(output_path).write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"\nSaved: {output_path}")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="XAI IoU evaluation against IDRiD lesion masks")
    parser.add_argument("--config", required=True, help="Path to model config YAML")
    parser.add_argument(
        "--split", default="train", choices=["train", "test"],
        help="IDRiD segmentation split (default: train)"
    )
    parser.add_argument("--idrid-root", help="Override IDRiD root directory")
    parser.add_argument(
        "--top-percents", nargs="+", type=float, default=[0.10, 0.20, 0.30],
        help="Top-N%% thresholds for CAM binarization (default: 0.10 0.20 0.30)"
    )
    parser.add_argument(
        "--target-block", type=int, default=None,
        help="Index into model.blocks for CAM target layer. Default: last block (blocks[-1]). "
             "Try 2 (64x64), 3 (32x32), 4 (32x32) for higher spatial resolution."
    )
    parser.add_argument("--output", help="Output JSON path")
    parser.add_argument(
        "--use-seg-head", action="store_true",
        help="Use auxiliary seg_head sigmoid output as heatmap instead of Layer-CAM"
    )
    args = parser.parse_args()

    evaluate(
        config_path=args.config,
        split=args.split,
        idrid_root=args.idrid_root,
        top_percents=args.top_percents,
        target_block=args.target_block,
        output_path=args.output,
        use_seg_head=args.use_seg_head,
    )


if __name__ == "__main__":
    main()
