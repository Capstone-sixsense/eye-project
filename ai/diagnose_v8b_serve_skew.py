"""M2 check: does the fixed v8b segmenter's lesion overlay (mDice) degrade when fed
the actual SERVE-geometry input (backend QuickQual square -> AI circular) vs the
TRAIN-geometry input (AI circular on raw, = the AI_HANDOFF baseline)?

(한글 요약) v8b 병변 오버레이(mDice)가 학습 geometry 입력 대비 실제 서빙 geometry 입력에서
저하되는지 측정한다. 분류기 AUROC가 아니라 '병변 공간 정합'에 대한 학습-서빙 skew 점검.

reeval (.omc/research/preprocessing_color/resize_path_reeval_v1.json) already showed the
fusion CLASSIFIER AUROC is robust to the skew. It did NOT measure the v8b spatial overlay.
This script measures exactly that, reusing eval_seg_evidence helpers so train-path
reproduces the published baseline and serve-path only prepends QuickQual to BOTH image
and mask (identical bbox), keeping mask/image geometry consistent.

Run (training interpreter has torch/cv2/sklearn):
    py -3.14 diagnose_v8b_serve_skew.py
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision.transforms import functional as TF

from eval_seg_evidence import (
    _dice_iou_arrays,
    _eval_preprocessing_enabled,
    _eval_transform,
    _idrid_items,
    _load_model,
    _maples_items,
    _mask_dict_to_eval_tensor,
)
from drscreen.data.transforms import FundusPreprocess, preprocess_kwargs_from_config
from drscreen.settings import load_app_config

PROJECT_ROOT = Path(__file__).resolve().parent
CONFIG = "configs/seg_evidence_v8b_ddrseg_tjdr_maplesfix.yaml"
REPORT = PROJECT_ROOT / ".omc/research/preprocessing_color/v8b_serve_skew_v1.json"
THRESHOLD = 0.5  # deployment infer.lesion_threshold; same threshold both paths so the delta is unbiased


def _qq_bbox(ref_rgb: np.ndarray, threshold: int = 15, buffer: int = 20):
    mean = ref_rgb.mean(-1)
    rows = np.where(mean > threshold)[0]
    cols = np.where(mean > threshold)[1]
    if rows.size == 0 or cols.size == 0:
        return None
    top, bottom = int(rows.min()), int(rows.max())
    left, right = int(cols.min()), int(cols.max())
    left = max(0, left - buffer)
    right = min(ref_rgb.shape[1], right + buffer)
    top = max(0, top - buffer)
    bottom = min(ref_rgb.shape[0], bottom + buffer)
    return left, top, right, bottom


def _square_pad(width: int, height: int):
    """Return TF.pad padding [left, top, right, bottom] matching backend QuickQual."""
    if width > height:
        pad = width - height
        return [0, pad // 2, 0, pad - pad // 2]
    pad = height - width
    return [pad // 2, 0, pad - pad // 2, 0]


def quickqual_image(pil: Image.Image) -> Image.Image:
    img = pil.convert("RGB")
    arr = np.asarray(img)
    bbox = _qq_bbox(arr)
    if bbox is None:
        return img.resize((1024, 1024), Image.LANCZOS)
    left, top, right, bottom = bbox
    img = img.crop((left, top, right, bottom))
    pad = _square_pad(img.size[0], img.size[1])
    img = TF.pad(img, pad)
    return img.resize((1024, 1024), Image.LANCZOS)


def quickqual_mask(mask_arr: np.ndarray, ref_pil: Image.Image) -> np.ndarray:
    """Apply the IDENTICAL QuickQual crop/pad/resize (NEAREST) the image gets, computed
    from the same reference image, so mask stays geometrically aligned to the image."""
    arr = np.asarray(ref_pil.convert("RGB"))
    bbox = _qq_bbox(arr)
    if bbox is None:
        return cv2.resize(mask_arr, (1024, 1024), interpolation=cv2.INTER_NEAREST)
    left, top, right, bottom = bbox
    cropped = mask_arr[top:bottom, left:right]
    pl, pt, pr, pb = _square_pad(cropped.shape[1], cropped.shape[0])
    padded = np.pad(cropped, ((pt, pb), (pl, pr)), mode="constant", constant_values=0)
    return cv2.resize(padded, (1024, 1024), interpolation=cv2.INTER_NEAREST)


def _eval_path(model, transform, mask_preprocessor, items, *, serve: bool) -> dict:
    device = next(model.parameters()).device
    rows = []
    for image_path, masks in items:
        raw = Image.open(image_path).convert("RGB")
        if serve:
            image = quickqual_image(raw)
            masks_use = {k: quickqual_mask(v, raw) for k, v in masks.items() if v is not None}
        else:
            image = raw
            masks_use = masks
        tensor = transform(image).unsqueeze(0).to(device)
        with torch.no_grad():
            probs = torch.sigmoid(model(tensor))[0].detach().cpu().numpy()
        _, h, w = probs.shape
        gt = _mask_dict_to_eval_tensor(masks_use, image=image, size=(w, h), preprocessor=mask_preprocessor).numpy()
        pred = (probs >= THRESHOLD).astype(np.uint8)
        rows.append(_dice_iou_arrays(pred, gt))

    def _mean(key):
        vals = [r[key] for r in rows if r.get(key) is not None]
        return float(np.mean(vals)) if vals else None

    return {"n": len(rows), "mdice": _mean("mdice"), "union_dice": _mean("union_dice"), "union_iou": _mean("union_iou")}


def main() -> None:
    config_path = (PROJECT_ROOT / CONFIG).resolve()
    config = load_app_config(config_path, base_path=config_path.parent / "base.yaml")
    version = str(config["project"]["version"])
    checkpoint_path = PROJECT_ROOT / "artifacts/runs/09_evidence_segmentation" / version / "checkpoints/best.pt"
    model = _load_model(config, PROJECT_ROOT, checkpoint_path)
    transform = _eval_transform(config, PROJECT_ROOT)
    data_cfg = config["data"]
    mask_preprocessor = (
        FundusPreprocess(output_size=int(data_cfg.get("image_size", 512)), **preprocess_kwargs_from_config(data_cfg))
        if _eval_preprocessing_enabled(config, PROJECT_ROOT)
        else None
    )

    report = {"version": version, "threshold": THRESHOLD, "datasets": {}}
    for name, items in (("idrid", _idrid_items(PROJECT_ROOT, "test")), ("maples", _maples_items(PROJECT_ROOT, "test"))):
        train = _eval_path(model, transform, mask_preprocessor, items, serve=False)
        serve = _eval_path(model, transform, mask_preprocessor, items, serve=True)
        report["datasets"][name] = {
            "train_path": train,
            "serve_path": serve,
            "mdice_delta_serve_minus_train": (
                None if train["mdice"] is None or serve["mdice"] is None else serve["mdice"] - train["mdice"]
            ),
        }
        print(f"{name}: train mDice {train['mdice']:.4f} | serve mDice {serve['mdice']:.4f} | "
              f"delta {report['datasets'][name]['mdice_delta_serve_minus_train']:+.4f}")

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nSaved: {REPORT}")


if __name__ == "__main__":
    main()
