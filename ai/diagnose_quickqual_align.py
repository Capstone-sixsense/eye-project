"""P-A and P-B verification before any retrain.

(한글 요약) 재학습 전, AI의 quickqual geometry가 백엔드 전처리와 일치하는지(P-A: bbox 동일)와
두 경로의 512 입력 차이(P-B: MAE/PSNR)를 검증한다. 학습-서빙 전처리 정합 점검 도구.

P-A: For sampled raw images, the AI `quickqual` geometry bbox MUST equal the backend
     `preprocess_fundus_image` bbox (both: RGB-mean>15, +20px buffer, clamped). Semantic
     replica check, not pixel-exact — resampler/resize differences are P-B's territory.

P-B: Offline FP(quickqual, 512)(raw) vs Serve FP(none, 512)(backend_quickqual_1024(raw)).
     Reports MAE/PSNR per domain. NOT bit-exact (offline=single BICUBIC, serve=LANCZOS->BICUBIC);
     the goal is bounded skew that §9 reeval already showed is AUROC-neutral.

Run:
    py -3.14 diagnose_quickqual_align.py
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from torchvision.transforms import functional as TF

from drscreen.data.transforms import FundusPreprocess

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "data" / "raw"
REPORT = ROOT / ".omc" / "research" / "preprocessing_color" / "quickqual_align_v1.json"
MANIFEST = ROOT / "data" / "processed" / "manifest_with_maples.csv"


def _backend_bbox(arr: np.ndarray, threshold: int = 15, buffer: int = 20):
    """Bbox per backend/models/quickqual_wrapper.py:31-47."""
    mean = arr.mean(-1)
    rows = np.where(mean > threshold)[0]
    cols = np.where(mean > threshold)[1]
    if rows.size == 0 or cols.size == 0:
        return None
    top, bottom = int(rows.min()), int(rows.max())
    left, right = int(cols.min()), int(cols.max())
    left = max(0, left - buffer)
    right = min(arr.shape[1], right + buffer)
    top = max(0, top - buffer)
    bottom = min(arr.shape[0], bottom + buffer)
    return (left, top, right, bottom)


def _backend_quickqual(pil: Image.Image) -> Image.Image:
    """Inlined backend preprocess_fundus_image (LANCZOS to 1024)."""
    img = pil.convert("RGB")
    arr = np.asarray(img)
    bbox = _backend_bbox(arr)
    if bbox is None:
        return img.resize((1024, 1024), Image.LANCZOS)
    left, top, right, bottom = bbox
    img = img.crop((left, top, right, bottom))
    w, h = img.size
    pad = [0, (w - h) // 2, 0, (w - h) - (w - h) // 2] if w > h else [(h - w) // 2, 0, (h - w) - (h - w) // 2, 0]
    img = TF.pad(img, pad)
    return img.resize((1024, 1024), Image.LANCZOS)


def _pixel_metrics(a: Image.Image, b: Image.Image) -> dict:
    aa = np.asarray(a).astype(np.float32)
    bb = np.asarray(b).astype(np.float32)
    mae = float(np.abs(aa - bb).mean())
    mse = float(((aa - bb) ** 2).mean())
    psnr = float("inf") if mse == 0 else 10.0 * math.log10((255.0 ** 2) / mse)
    return {"mae": mae, "psnr": psnr}


def main() -> None:
    fp_qq = FundusPreprocess(output_size=512, preprocess_mode="quickqual")
    fp_none = FundusPreprocess(output_size=512, preprocess_mode="none")

    frame = pd.read_csv(MANIFEST)
    sample = []
    for dom, g in frame.groupby("domain"):
        sample.append(g.sample(n=min(20, len(g)), random_state=42))
    sample = pd.concat(sample).reset_index(drop=True)

    # P-A: bbox equality, no resize/resampler involved
    pa_mismatches = []
    pa_total = 0
    for _, row in sample.iterrows():
        raw_path = RAW / row["image_path"]
        if not raw_path.exists():
            continue
        with Image.open(raw_path) as img:
            arr = np.asarray(img.convert("RGB"))
        backend_box = _backend_bbox(arr)
        geom = fp_qq._quickqual_geometry(arr)
        ai_box = None if geom is None else (geom[0], geom[1], geom[2], geom[3])
        pa_total += 1
        if backend_box != ai_box:
            pa_mismatches.append({"image_id": str(row["image_id"]), "domain": str(row["domain"]),
                                  "backend": backend_box, "ai": ai_box})

    # P-B: offline (FP quickqual->512) vs serve (FP none on backend's 1024 ->512)
    per_domain: dict[str, list[dict]] = {}
    for _, row in sample.iterrows():
        raw_path = RAW / row["image_path"]
        if not raw_path.exists():
            continue
        with Image.open(raw_path) as img:
            offline = fp_qq(img.convert("RGB"))
            serve = fp_none(_backend_quickqual(img))
        m = _pixel_metrics(offline, serve)
        per_domain.setdefault(str(row["domain"]), []).append(m)

    summary = {}
    all_mae, all_psnr = [], []
    for dom, items in per_domain.items():
        maes = [x["mae"] for x in items]
        psnrs = [x["psnr"] for x in items if math.isfinite(x["psnr"])]
        summary[dom] = {
            "n": len(items),
            "mae_mean": float(np.mean(maes)),
            "mae_p90": float(np.percentile(maes, 90)),
            "psnr_mean": float(np.mean(psnrs)) if psnrs else float("inf"),
        }
        all_mae.extend(maes)
        all_psnr.extend(psnrs)

    report = {
        "P_A_bbox_equivalence": {
            "n_total": pa_total,
            "n_mismatch": len(pa_mismatches),
            "passed": len(pa_mismatches) == 0,
            "mismatches": pa_mismatches[:10],
        },
        "P_B_offline_vs_serve": {
            "per_domain": summary,
            "overall_mae": float(np.mean(all_mae)) if all_mae else 0.0,
            "overall_psnr_db": float(np.mean(all_psnr)) if all_psnr else float("inf"),
            "note": "Not bit-exact (offline BICUBIC, serve LANCZOS+BICUBIC). §9 showed bounded skew is AUROC-neutral.",
        },
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"\nSaved: {REPORT}")


if __name__ == "__main__":
    main()
