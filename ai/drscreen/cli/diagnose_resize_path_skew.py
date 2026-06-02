"""Diagnose the train/serve preprocessing skew before deciding to re-preprocess + retrain.

Two checks, both 0-training:

  Check 1 (pixeldiff): For sampled RAW images, compare the 512 model-input produced by
    - offline-style:  FundusPreprocess(raw)                       (single resize raw->512)
    - serve-style:    FundusPreprocess(QuickQual(raw))            (raw->square->1024 then ->512)
  Reports per-domain MAE / PSNR between the two 512 inputs. If the diff is negligible,
  the resize-path skew is not worth a retrain.

  Check 2 (reeval): Feed BOTH 512 inputs to the active fusion model and compare AUROC on
  the same image set. Isolates the preprocessing effect from sampling (unlike the
  0.9356 raw-live smoke vs 0.9403 offline holdout, which used different samples).

Run with the training interpreter (has torch/pandas/cv2/sklearn):

  py -3.14 -m drscreen.cli.diagnose_resize_path_skew pixeldiff \
      --config configs/base.yaml \
      --manifest data/processed/manifest_with_maples.csv \
      --n-per-domain 40

  py -3.14 -m drscreen.cli.diagnose_resize_path_skew reeval \
      --config configs/base.yaml \
      --manifest data/processed/manifest.csv \
      --domain DDR --split external_test --n 600

Neither subcommand writes images or touches active artifacts; only a JSON report is saved.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from torchvision.transforms import functional as TF

from drscreen.data.transforms import FundusPreprocess, preprocess_kwargs_from_config
from drscreen.settings import load_app_config

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_ROOT = PROJECT_ROOT / "data" / "raw"
REPORT_DIR = PROJECT_ROOT / ".omc" / "research" / "preprocessing_color"


def _quickqual_preprocess(img: Image.Image, threshold: int = 15) -> Image.Image:
    """Replicate backend QuickQual geometry (border crop -> square pad -> 1024 LANCZOS).

    Mirrors backend/models/quickqual_wrapper.py::preprocess_fundus_image so the AI repo
    does not depend on backend code. threshold=15, buffer=20 match the backend exactly.
    """
    img = img.convert("RGB")
    arr = np.asarray(img)
    mean = arr.mean(-1)
    rows = np.where(mean > threshold)[0]
    cols = np.where(mean > threshold)[1]
    if rows.size == 0 or cols.size == 0:
        return img.resize((1024, 1024), Image.LANCZOS)
    top, bottom = int(rows.min()), int(rows.max())
    left, right = int(cols.min()), int(cols.max())
    buffer = 20
    left = max(0, left - buffer)
    right = min(arr.shape[1], right + buffer)
    top = max(0, top - buffer)
    bottom = min(arr.shape[0], bottom + buffer)
    img = img.crop((left, top, right, bottom))
    width, height = img.size
    if width > height:
        pad = width - height
        padding = [0, pad // 2, 0, pad - pad // 2]
    else:
        pad = height - width
        padding = [pad // 2, 0, pad - pad // 2, 0]
    img = TF.pad(img, padding)
    return img.resize((1024, 1024), Image.LANCZOS)


def _build_active_preprocessor(config: dict) -> FundusPreprocess:
    """Build the SAME FundusPreprocess the inference service uses (service.py:621-629)."""
    data_cfg = config["data"]
    infer_cfg = config.get("infer", {})
    preprocess_size = int(data_cfg.get("preprocess_size", 0)) or None
    use_align = bool(infer_cfg.get("use_align", data_cfg.get("use_align", False)))
    options = preprocess_kwargs_from_config(data_cfg, infer_cfg)
    return FundusPreprocess(output_size=preprocess_size, align=use_align, **options)


def _load_config(config_path: Path) -> dict:
    base = config_path.parent / "base.yaml"
    return load_app_config(
        config_path,
        base_path=base if config_path.name != "base.yaml" and base.exists() else None,
    )


def _sample_rows(frame: pd.DataFrame, *, per_domain: int, seed: int) -> pd.DataFrame:
    parts = []
    for domain, group in frame.groupby("domain"):
        parts.append(group.sample(n=min(per_domain, len(group)), random_state=seed))
    return pd.concat(parts).reset_index(drop=True)


def _offline_lookup(offline_manifest: str | None) -> dict[str, Path] | None:
    """image_id -> on-disk preprocessed PNG path (ground-truth training input).

    When provided, the offline baseline uses the ACTUAL preprocessed image the model
    trained on, removing any ambiguity about which geometry mode produced it. Without
    it, the offline baseline is regenerated with the active infer-config geometry.
    """
    if not offline_manifest:
        return None
    frame = pd.read_csv(offline_manifest)
    return {str(r["image_id"]): (RAW_ROOT / r["image_path"]) for _, r in frame.iterrows()}


def _offline_input(
    row: pd.Series,
    raw_img: Image.Image,
    fp: FundusPreprocess,
    lookup: dict[str, Path] | None,
) -> Image.Image | None:
    """Offline (training-style) 512 input: from disk if a lookup is given, else regenerated."""
    if lookup is not None:
        path = lookup.get(str(row["image_id"]))
        if path is None or not path.exists():
            return None
        return Image.open(path).convert("RGB")
    return fp(raw_img.convert("RGB"))


def run_pixeldiff(args: argparse.Namespace) -> None:
    config = _load_config(Path(args.config).resolve())
    fp = _build_active_preprocessor(config)
    lookup = _offline_lookup(args.offline_manifest)
    frame = pd.read_csv(args.manifest)
    sample = _sample_rows(frame, per_domain=args.n_per_domain, seed=args.seed)

    per_domain: dict[str, list[dict]] = {}
    for i, row in sample.iterrows():
        raw_path = RAW_ROOT / row["image_path"]
        if not raw_path.exists():
            continue
        with Image.open(raw_path) as img:
            offline = _offline_input(row, img, fp, lookup)
            if offline is None:
                continue
            serve = fp(_quickqual_preprocess(img))
        a = np.asarray(offline).astype(np.float32)
        b = np.asarray(serve).astype(np.float32)
        if a.shape != b.shape:
            continue
        mae = float(np.abs(a - b).mean())
        mse = float(((a - b) ** 2).mean())
        psnr = float("inf") if mse == 0 else 10.0 * math.log10((255.0 ** 2) / mse)
        max_abs = float(np.abs(a - b).max())
        per_domain.setdefault(str(row["domain"]), []).append(
            {"mae": mae, "psnr": psnr, "max_abs": max_abs}
        )
        if (i + 1) % 50 == 0:
            print(f"  processed {i + 1}/{len(sample)}", flush=True)

    summary = {}
    all_mae, all_psnr = [], []
    for domain, items in per_domain.items():
        maes = [x["mae"] for x in items]
        psnrs = [x["psnr"] for x in items if math.isfinite(x["psnr"])]
        summary[domain] = {
            "n": len(items),
            "mae_mean": float(np.mean(maes)),
            "mae_p90": float(np.percentile(maes, 90)),
            "psnr_mean": float(np.mean(psnrs)) if psnrs else float("inf"),
            "max_abs_max": float(max(x["max_abs"] for x in items)),
        }
        all_mae.extend(maes)
        all_psnr.extend(psnrs)

    overall_mae = float(np.mean(all_mae)) if all_mae else 0.0
    overall_psnr = float(np.mean(all_psnr)) if all_psnr else float("inf")
    # Heuristic verdict: <1/255 mean MAE and >45 dB PSNR => skew is sub-pixel-class.
    negligible = overall_mae < 1.0 and overall_psnr > 45.0

    report = {
        "check": "pixeldiff",
        "manifest": str(args.manifest),
        "offline_baseline": "on_disk_preprocessed" if lookup is not None else "regenerated_active_geometry",
        "preprocess_mode": fp._preprocess_mode,
        "output_size": fp._output_size,
        "n_per_domain": args.n_per_domain,
        "per_domain": summary,
        "overall_mae": overall_mae,
        "overall_psnr_db": overall_psnr,
        "verdict_negligible_skew": negligible,
        "note": "negligible=True => resize-path skew is sub-pixel-class; retrain not justified.",
    }
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    out = REPORT_DIR / "resize_path_pixeldiff_v1.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"\nSaved: {out}")


def run_reeval(args: argparse.Namespace) -> None:
    from sklearn.metrics import roc_auc_score

    from drscreen.infer.service import InferenceSession

    config_path = Path(args.config).resolve()
    config = _load_config(config_path)
    fp = _build_active_preprocessor(config)

    session = InferenceSession.from_config_path(config_path)
    # We pre-apply FundusPreprocess ourselves for BOTH paths, so disable the
    # session's own preprocessor to avoid double application.
    session.preprocessor = None
    lookup = _offline_lookup(args.offline_manifest)

    frame = pd.read_csv(args.manifest)
    frame = frame[(frame["domain"] == args.domain) & (frame["split"] == args.split)]
    if frame.empty:
        raise SystemExit(f"No rows for domain={args.domain} split={args.split} in {args.manifest}")
    # Balance by label, then cap at --n.
    parts = [g.sample(n=min(args.n // 2, len(g)), random_state=args.seed) for _, g in frame.groupby("label")]
    sample = pd.concat(parts).reset_index(drop=True)

    labels, probs_off, probs_srv = [], [], []
    for i, row in sample.iterrows():
        raw_path = RAW_ROOT / row["image_path"]
        if not raw_path.exists():
            continue
        with Image.open(raw_path) as img:
            offline = _offline_input(row, img, fp, lookup)
            if offline is None:
                continue
            serve = fp(_quickqual_preprocess(img))
        p_off = session.predict_pil_image(offline, save_outputs=False).payload["abnormal_probability"]
        p_srv = session.predict_pil_image(serve, save_outputs=False).payload["abnormal_probability"]
        labels.append(int(row["label"]))
        probs_off.append(float(p_off))
        probs_srv.append(float(p_srv))
        if (i + 1) % 50 == 0:
            print(f"  processed {i + 1}/{len(sample)}", flush=True)

    auroc_off = float(roc_auc_score(labels, probs_off))
    auroc_srv = float(roc_auc_score(labels, probs_srv))
    prob_mae = float(np.abs(np.array(probs_off) - np.array(probs_srv)).mean())

    report = {
        "check": "reeval",
        "manifest": str(args.manifest),
        "offline_baseline": "on_disk_preprocessed" if lookup is not None else "regenerated_active_geometry",
        "domain": args.domain,
        "split": args.split,
        "n_evaluated": len(labels),
        "auroc_offline_style": auroc_off,
        "auroc_serve_style": auroc_srv,
        "auroc_delta_serve_minus_offline": auroc_srv - auroc_off,
        "abnormal_prob_mae_between_paths": prob_mae,
        "note": (
            "If |auroc_delta| and prob MAE are tiny, the preprocessing skew does not move "
            "the model; the 0.9356-vs-0.9403 gap is then mostly sampling, and retrain is "
            "not justified."
        ),
    }
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    out = REPORT_DIR / "resize_path_reeval_v1.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"\nSaved: {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p1 = sub.add_parser("pixeldiff", help="Check 1: pixel MAE/PSNR between offline- and serve-style 512 inputs.")
    p1.add_argument("--config", default="configs/base.yaml")
    p1.add_argument("--manifest", default="data/processed/manifest_with_maples.csv")
    p1.add_argument(
        "--offline-manifest",
        default=None,
        help="Preprocessed manifest (e.g. data/processed/manifest_preprocessed.csv). If set, "
        "the offline baseline uses the on-disk training image instead of regenerating it.",
    )
    p1.add_argument("--n-per-domain", type=int, default=40)
    p1.add_argument("--seed", type=int, default=42)
    p1.set_defaults(func=run_pixeldiff)

    p2 = sub.add_parser("reeval", help="Check 2: AUROC on the same images via offline- vs serve-style preprocessing.")
    p2.add_argument("--config", default="configs/base.yaml")
    p2.add_argument("--manifest", default="data/processed/manifest.csv")
    p2.add_argument(
        "--offline-manifest",
        default=None,
        help="Preprocessed manifest. If set, the offline path feeds the on-disk training "
        "image (preprocessor disabled) instead of regenerating it.",
    )
    p2.add_argument("--domain", default="DDR")
    p2.add_argument("--split", default="external_test")
    p2.add_argument("--n", type=int, default=600)
    p2.add_argument("--seed", type=int, default=42)
    p2.set_defaults(func=run_reeval)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
