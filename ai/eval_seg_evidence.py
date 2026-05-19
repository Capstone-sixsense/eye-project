"""Evaluate standalone lesion segmentation evidence models."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))

from drscreen.data.transforms import FundusPreprocess, build_eval_transform
from drscreen.models.profiles import get_model_profile
from drscreen.models.seg_evidence import LesionSegEvidence
from drscreen.settings import get_run_evaluation_dir, load_app_config, resolve_project_path
from drscreen.xai.iou import LESION_CODES, load_lesion_masks, load_maples_masks


def _load_model(config: dict, project_root: Path, checkpoint_path: Path) -> torch.nn.Module:
    device = torch.device(str(config["train"].get("device", "cuda")))
    if device.type == "cuda" and not torch.cuda.is_available():
        device = torch.device("cpu")
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    ckpt_cfg = payload.get("config", config)
    model_cfg = ckpt_cfg.get("model", config.get("model", {}))
    model = LesionSegEvidence(
        encoder=str(model_cfg.get("encoder", "resnet50")),
        out_channels=int(model_cfg.get("out_channels", 4)),
        pretrained=False,
        decoder_channels=tuple(model_cfg.get("decoder_channels", [256, 128, 64, 32])),
    )
    model.load_state_dict(payload["model_state_dict"])
    model.to(device)
    model.eval()
    return model


def _manifest_uses_offline_preprocessing(config: dict, project_root: Path) -> bool:
    data_cfg = config["data"]
    manifest_value = data_cfg.get("manifest_path")
    if not manifest_value:
        return False
    manifest_path = resolve_project_path(project_root, manifest_value)
    if not manifest_path.exists():
        return False
    with manifest_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        first = next(reader, None)
    if not first:
        return False
    return str(first.get("image_path", "")).startswith("processed/images/")


def _eval_preprocessing_enabled(config: dict, project_root: Path) -> bool:
    data_cfg = config["data"]
    use_preprocessing = bool(data_cfg.get("use_preprocessing", False))
    if data_cfg.get("eval_use_preprocessing") is not None:
        return bool(data_cfg["eval_use_preprocessing"])
    if _manifest_uses_offline_preprocessing(config, project_root):
        return True
    return use_preprocessing


def _eval_transform(config: dict, project_root: Path):
    model_cfg = config.get("model", {})
    data_cfg = config["data"]
    profile = get_model_profile(str(model_cfg.get("encoder", "resnet50")))
    image_size = int(data_cfg.get("image_size", profile.crop_size))
    resize_size = int(data_cfg.get("resize_size", image_size))
    use_preprocessing = _eval_preprocessing_enabled(config, project_root)
    return build_eval_transform(
        crop_size=image_size,
        resize_size=resize_size,
        interpolation=profile.interpolation,
        mean=profile.mean,
        std=profile.std,
        use_preprocessing=use_preprocessing,
    )


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


def _mask_dict_to_eval_tensor(
    masks: dict[str, np.ndarray],
    *,
    image: Image.Image,
    size: tuple[int, int],
    preprocessor: FundusPreprocess | None,
) -> torch.Tensor:
    w, h = size
    base_shape = next((arr.shape for arr in masks.values() if arr is not None), (h, w))
    raw_channels: list[np.ndarray] = []
    for code in LESION_CODES:
        arr = masks.get(code)
        if arr is None:
            arr = np.zeros(base_shape, dtype=np.uint8)
        raw_channels.append((arr > 0).astype(np.uint8))
    stacked = np.stack(raw_channels, axis=-1)
    if preprocessor is not None:
        stacked = preprocessor.apply_mask_geometry(stacked, image, output_size=w)
    elif stacked.shape[:2] != (h, w):
        stacked = cv2.resize(stacked, (w, h), interpolation=cv2.INTER_NEAREST)
    if stacked.ndim == 2:
        stacked = stacked[..., None]
    return torch.from_numpy((stacked > 0).astype(np.float32)).permute(2, 0, 1)


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


def _load_tjdr_masks(annotation_path: Path) -> dict[str, np.ndarray]:
    # TJDR labels: 1=EX, 2=HE, 3=MA, 4=SE. Project order is MA/HE/EX/SE.
    arr = np.array(Image.open(annotation_path), dtype=np.uint8)
    if arr.ndim == 3:
        arr = arr[..., 0]
    return {
        "MA": (arr == 3).astype(np.uint8),
        "HE": (arr == 2).astype(np.uint8),
        "EX": (arr == 1).astype(np.uint8),
        "SE": (arr == 4).astype(np.uint8),
    }


def _tjdr_items(project_root: Path, split: str) -> list[tuple[Path, dict[str, np.ndarray]]]:
    root = project_root / "data/raw/TJDR" / split
    image_dir = root / "image"
    ann_dir = root / "annotation"
    items: list[tuple[Path, dict[str, np.ndarray]]] = []
    for image_path in sorted(image_dir.glob("*.png")):
        ann_path = ann_dir / image_path.name
        if not ann_path.exists():
            continue
        items.append((image_path, _load_tjdr_masks(ann_path)))
    return items


def evaluate(
    config_path: str,
    *,
    checkpoint: str | None = None,
    mask_provider: str = "idrid",
    split: str = "test",
    output: str | None = None,
    lesion_threshold: float | None = None,
) -> dict:
    config_path_obj = Path(config_path).resolve()
    project_root = config_path_obj.parents[1]
    base_path = config_path_obj.parent / "base.yaml"
    config = load_app_config(config_path_obj, base_path=base_path if base_path.exists() else None)
    version = str(config["project"].get("version", "seg_evidence"))
    checkpoint_path = (
        resolve_project_path(project_root, checkpoint)
        if checkpoint
        else project_root / "artifacts/runs/09_evidence_segmentation" / version / "checkpoints/best.pt"
    )
    model = _load_model(config, project_root, checkpoint_path)
    device = next(model.parameters()).device
    transform = _eval_transform(config, project_root)
    data_cfg = config["data"]
    mask_preprocessor = (
        FundusPreprocess(output_size=int(data_cfg.get("image_size", 512)))
        if _eval_preprocessing_enabled(config, project_root)
        else None
    )
    threshold = (
        float(lesion_threshold)
        if lesion_threshold is not None
        else float(config.get("infer", {}).get("lesion_threshold", 0.5))
    )

    if mask_provider == "idrid":
        items = _idrid_items(project_root, split)
        dataset_name = "idrid"
    elif mask_provider == "maples":
        items = _maples_items(project_root, split)
        dataset_name = "maples"
    elif mask_provider == "tjdr":
        items = _tjdr_items(project_root, split)
        dataset_name = "tjdr"
    else:
        raise ValueError("mask_provider must be 'idrid', 'maples', or 'tjdr'")
    if not items:
        raise FileNotFoundError(f"No evaluation images found for {dataset_name}:{split}")

    per_image = []
    for image_path, masks in items:
        image = Image.open(image_path).convert("RGB")
        tensor = transform(image).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(tensor)
            probs = torch.sigmoid(logits)[0].detach().cpu().numpy()
        _, h, w = probs.shape
        gt = _mask_dict_to_eval_tensor(
            masks,
            image=image,
            size=(w, h),
            preprocessor=mask_preprocessor,
        ).numpy()
        pred = (probs >= threshold).astype(np.uint8)
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
        "checkpoint_path": str(checkpoint_path),
        "dataset": dataset_name,
        "split": split,
        "lesion_threshold": threshold,
        "aggregate": aggregate,
        "per_image": per_image,
    }

    if output is None:
        eval_dir = get_run_evaluation_dir(project_root, version)
        eval_dir.mkdir(parents=True, exist_ok=True)
        output = str(eval_dir / f"seg_eval_{dataset_name}_{split}.json")
    Path(output).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Saved: {output}")
    print(json.dumps(aggregate, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate lesion segmentation evidence model.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint")
    parser.add_argument("--mask-provider", default="idrid", choices=["idrid", "maples", "tjdr"])
    parser.add_argument("--split", default="test", choices=["train", "test"])
    parser.add_argument("--output")
    parser.add_argument("--lesion-threshold", type=float)
    args = parser.parse_args()
    evaluate(
        args.config,
        checkpoint=args.checkpoint,
        mask_provider=args.mask_provider,
        split=args.split,
        output=args.output,
        lesion_threshold=args.lesion_threshold,
    )


if __name__ == "__main__":
    main()
