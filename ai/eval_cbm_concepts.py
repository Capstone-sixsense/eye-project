"""Evaluate CBM concept maps against IDRiD/MAPLES lesion masks."""
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
from drscreen.settings import get_run_evaluation_dir
from drscreen.xai.iou import LESION_CODES, load_lesion_masks, load_maples_masks


def _mask_dict_to_tensor(masks: dict[str, np.ndarray], size: tuple[int, int]) -> torch.Tensor:
    w, h = size
    channels: list[torch.Tensor] = []
    for code in LESION_CODES:
        arr = masks.get(code)
        if arr is None:
            arr = np.zeros((h, w), dtype=np.uint8)
        elif arr.shape != (h, w):
            arr = cv2.resize(arr, (w, h), interpolation=cv2.INTER_NEAREST)
        channels.append(torch.from_numpy((arr > 0).astype(np.float32)))
    return torch.stack(channels, dim=0)


def _dice_iou_arrays(pred: np.ndarray, gt: np.ndarray) -> dict:
    per_class: dict[str, dict[str, float | None]] = {}
    dice_values: list[float] = []
    iou_values: list[float] = []
    for idx, code in enumerate(LESION_CODES):
        p = pred[idx] > 0
        g = gt[idx] > 0
        if not g.any():
            per_class[code] = {"dice": None, "iou": None}
            continue
        inter = float(np.logical_and(p, g).sum())
        p_sum = float(p.sum())
        g_sum = float(g.sum())
        union = p_sum + g_sum - inter
        dice = (2.0 * inter + 1e-6) / (p_sum + g_sum + 1e-6)
        iou = (inter + 1e-6) / (union + 1e-6)
        per_class[code] = {"dice": dice, "iou": iou}
        dice_values.append(dice)
        iou_values.append(iou)

    pred_union = pred.max(axis=0) > 0
    gt_union = gt.max(axis=0) > 0
    union_dice = None
    union_iou = None
    if gt_union.any():
        inter = float(np.logical_and(pred_union, gt_union).sum())
        p_sum = float(pred_union.sum())
        g_sum = float(gt_union.sum())
        union = p_sum + g_sum - inter
        union_dice = (2.0 * inter + 1e-6) / (p_sum + g_sum + 1e-6)
        union_iou = (inter + 1e-6) / (union + 1e-6)

    return {
        "mdice": float(np.mean(dice_values)) if dice_values else None,
        "miou": float(np.mean(iou_values)) if iou_values else None,
        "union_dice": union_dice,
        "union_iou": union_iou,
        "per_class": per_class,
    }


def _image_candidates(base: Path, stem: str) -> list[Path]:
    return [base / f"{stem}{ext}" for ext in (".jpg", ".jpeg", ".png", ".tif", ".tiff")]


def _idrid_items(project_root: Path, split: str) -> list[tuple[Path, dict[str, np.ndarray]]]:
    split_dir = {"train": "a. Training Set", "test": "b. Testing Set"}[split]
    image_dir = project_root / "data/raw/IDRiD/A. Segmentation/1. Original Images" / split_dir
    mask_dir = project_root / "data/raw/IDRiD/A. Segmentation/2. All Segmentation Groundtruths" / split_dir
    items: list[tuple[Path, dict[str, np.ndarray]]] = []
    for image_path in sorted(image_dir.glob("*.jpg")) + sorted(image_dir.glob("*.png")):
        masks = load_lesion_masks(mask_dir, image_path.stem)
        if masks:
            items.append((image_path, masks))
    return items


def _maples_items(project_root: Path, split: str) -> list[tuple[Path, dict[str, np.ndarray]]]:
    import yaml

    maples_root = project_root / "data/raw/MAPLES-DR/AdditionalData"
    annotations_dir = maples_root / "annotations"
    messidor_dir = project_root / "data/raw/messidor/images"
    with (maples_root / "dataset_record.yaml").open("r", encoding="utf-8") as handle:
        record = yaml.safe_load(handle)
    items: list[tuple[Path, dict[str, np.ndarray]]] = []
    for stem in record[split]:
        image_path = next((p for p in _image_candidates(messidor_dir, stem) if p.exists()), None)
        if image_path is None:
            continue
        masks = load_maples_masks(annotations_dir, stem)
        if masks:
            items.append((image_path, masks))
    return items


def evaluate(
    config_path: str,
    *,
    mask_provider: str = "idrid",
    split: str = "test",
    output: str | None = None,
    lesion_threshold: float = 0.5,
) -> dict:
    session = InferenceSession.from_config_path(config_path)
    project_root = session.project_root
    version = str(session.config.get("project", {}).get("version", "cbm"))

    if mask_provider == "idrid":
        items = _idrid_items(project_root, split)
        dataset_name = "idrid"
    elif mask_provider == "maples":
        items = _maples_items(project_root, split)
        dataset_name = "maples"
    else:
        raise ValueError("mask_provider must be 'idrid' or 'maples'")
    if not items:
        raise FileNotFoundError(f"No evaluation images found for {dataset_name}:{split}")

    per_image = []
    for image_path, masks in items:
        image = Image.open(image_path).convert("RGB")
        tensor = session.eval_transform(image).unsqueeze(0).to(session.device)
        with torch.no_grad():
            probs = session.model.predict_seg(tensor)[0].detach().cpu().numpy()
        _, h, w = probs.shape
        gt = _mask_dict_to_tensor(masks, (w, h)).numpy()
        pred = (probs >= lesion_threshold).astype(np.uint8)
        metrics = _dice_iou_arrays(pred, gt)
        per_image.append({"image_id": image_path.stem, **metrics})

    def _mean(key: str) -> float | None:
        vals = [r[key] for r in per_image if r.get(key) is not None]
        return float(np.mean(vals)) if vals else None

    aggregate = {
        "mdice": _mean("mdice"),
        "miou": _mean("miou"),
        "union_dice": _mean("union_dice"),
        "union_iou": _mean("union_iou"),
        "n_images": len(per_image),
    }
    result = {
        "version": version,
        "checkpoint_path": str(session.checkpoint_path),
        "dataset": dataset_name,
        "split": split,
        "lesion_threshold": lesion_threshold,
        "aggregate": aggregate,
        "per_image": per_image,
    }

    if output is None:
        eval_dir = get_run_evaluation_dir(project_root, version)
        eval_dir.mkdir(parents=True, exist_ok=True)
        output = str(eval_dir / f"cbm_concept_eval_{dataset_name}_{split}.json")
    Path(output).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Saved: {output}")
    print(json.dumps(aggregate, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate CBM concept maps.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--mask-provider", default="idrid", choices=["idrid", "maples"])
    parser.add_argument("--split", default="test", choices=["train", "test"])
    parser.add_argument("--output")
    parser.add_argument("--lesion-threshold", type=float, default=0.5)
    args = parser.parse_args()
    evaluate(
        args.config,
        mask_provider=args.mask_provider,
        split=args.split,
        output=args.output,
        lesion_threshold=args.lesion_threshold,
    )


if __name__ == "__main__":
    main()
