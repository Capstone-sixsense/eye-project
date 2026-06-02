"""Phase 0: power / ceiling / C-premise. Replicates late_fusion_classifier setup to get
per-sample scores (the fusion metrics JSON does not persist them), then CPU analysis.

  - POWER: paired-bootstrap 95% CI of holdout AUROC and delta(fusion - v31).
  - CEILING: Q-statistic / (v31-wrong & v8b-correct)/(v31-wrong) on holdout.
  - C-PREMISE: v31 error rate on train / external_calibration / external_holdout.

Run: py -3.14 diagnose_fusion_complementarity.py --config configs/v31_v8b_late_fusion_quickqual_v1.yaml
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from drscreen.cli.late_fusion_classifier import _apply_calibration_split, _build_feature_matrix, _extract_v31_scores
from drscreen.cli.lesion_evidence_classifier import _build_transform, _extract_features, _load_segmenter, _read_items
from drscreen.infer.service import InferenceSession
from drscreen.settings import load_app_config, resolve_project_path

PROJECT_ROOT = Path(__file__).resolve().parent
REPORT = PROJECT_ROOT / ".omc/research/fusion_complementarity/phase0_power_ceiling.json"
FEATURE_SETS = ["v31_score_only", "v8b_evidence_only", "late_fusion"]
CLASSIFICATION_DOMAINS = ["APTOS", "IDRiD", "Messidor"]
HOLDOUT, CAL, TRAIN = "external_holdout", "external_calibration", "train"


def _ci(vals: list[float]) -> dict:
    arr = np.asarray(vals)
    lo, hi = float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))
    return {"mean": float(arr.mean()), "ci95_low": lo, "ci95_high": hi, "half_width": float((hi - lo) / 2)}


def _q(c1: np.ndarray, c2: np.ndarray) -> dict:
    n11 = int(np.sum(c1 & c2)); n00 = int(np.sum(~c1 & ~c2)); n01 = int(np.sum(~c1 & c2)); n10 = int(np.sum(c1 & ~c2))
    den = n11 * n00 + n01 * n10
    wrong = n01 + n00
    return {
        "Q_statistic": float((n11 * n00 - n01 * n10) / den) if den else 0.0,
        "disagreement": float((n01 + n10) / len(c1)),
        "v31_wrong_n": wrong,
        "v8b_correction_rate_of_v31_errors": float(n01 / wrong) if wrong else 0.0,
        "counts": {"both_correct": n11, "both_wrong": n00, "only_v8b_correct": n01, "only_v31_correct": n10},
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/v31_v8b_late_fusion_quickqual_v1.yaml")
    ap.add_argument("--bootstrap", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=43)
    args = ap.parse_args()

    config_path = (PROJECT_ROOT / args.config).resolve()
    config = load_app_config(config_path)
    project_root = config_path.parents[1]
    data_cfg = config["data"]
    fusion_cfg = config.get("fusion", {})
    bs = int(data_cfg.get("batch_size", 16)); nw = int(data_cfg.get("num_workers", 0))

    items = _read_items(config, project_root)
    items, _ = _apply_calibration_split(items, data_cfg.get("calibration_split", {}), seed=int(fusion_cfg.get("seed", 43)))

    classifier_session = InferenceSession.from_config_path(
        resolve_project_path(project_root, config["classifier"]["config_path"]),
        checkpoint_path=str(resolve_project_path(project_root, config["classifier"]["checkpoint_path"])),
    )
    seg_config = load_app_config(resolve_project_path(project_root, config["segmentation"]["config_path"]))
    segmenter = _load_segmenter(seg_config, project_root, resolve_project_path(project_root, config["segmentation"]["checkpoint_path"]))
    seg_transform = _build_transform(seg_config, project_root)

    v31_scores, labels, rows = _extract_v31_scores(classifier_session, items, batch_size=bs, num_workers=nw)
    v8b_features, v8b_labels, _ = _extract_features(segmenter, items, transform=seg_transform, batch_size=bs, num_workers=nw)
    if not np.array_equal(labels, v8b_labels):
        raise RuntimeError("v31/v8b label misalignment.")

    splits = np.asarray([str(r["split"]) for r in rows])
    domains = np.asarray([str(r["domain"]) for r in rows])
    y = labels.astype(int)
    train_mask = (splits == TRAIN) & np.isin(domains, CLASSIFICATION_DOMAINS)

    scores = {}
    for fs in FEATURE_SETS:
        matrix, _ = _build_feature_matrix(fs, v31_scores=v31_scores, v8b_features=v8b_features)
        model = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=5000, class_weight="balanced", solver="lbfgs", random_state=args.seed))
        model.fit(matrix[train_mask], y[train_mask])
        scores[fs] = model.predict_proba(matrix)[:, 1]

    hm = splits == HOLDOUT
    yh = y[hm]; v31h = scores["v31_score_only"][hm]; v8bh = scores["v8b_evidence_only"][hm]; fush = scores["late_fusion"][hm]

    rng = np.random.default_rng(args.seed); n = len(yh)
    b = {"v31": [], "v8b": [], "fusion": [], "delta_fus_v31": [], "delta_fus_v8b": []}
    for _ in range(args.bootstrap):
        idx = rng.integers(0, n, n); yy = yh[idx]
        if yy.min() == yy.max():
            continue
        a31, a8, af = roc_auc_score(yy, v31h[idx]), roc_auc_score(yy, v8bh[idx]), roc_auc_score(yy, fush[idx])
        b["v31"].append(a31); b["v8b"].append(a8); b["fusion"].append(af)
        b["delta_fus_v31"].append(af - a31); b["delta_fus_v8b"].append(af - a8)
    power = {
        "holdout_n": int(n),
        "auroc_point": {"v31": float(roc_auc_score(yh, v31h)), "v8b": float(roc_auc_score(yh, v8bh)), "fusion": float(roc_auc_score(yh, fush))},
        "auroc_ci": {k: _ci(b[k]) for k in ["v31", "v8b", "fusion"]},
        "delta_ci": {k: _ci(b[k]) for k in ["delta_fus_v31", "delta_fus_v8b"]},
    }
    ceiling = _q((v31h >= 0.5) == (yh == 1), (v8bh >= 0.5) == (yh == 1))
    c_premise = {}
    for sp in [TRAIN, CAL, HOLDOUT]:
        m = splits == sp
        if m.any():
            c_premise[sp] = {"n": int(m.sum()), "v31_error_rate": float(np.mean((scores["v31_score_only"][m] >= 0.5).astype(int) != y[m])), "pos_rate": float(np.mean(y[m]))}

    hw = power["delta_ci"]["delta_fus_v31"]["half_width"]
    fus = power["auroc_point"]["fusion"]
    report = {
        "config": args.config,
        "power": power, "ceiling": ceiling, "c_premise": c_premise,
        "gates": {
            "P0_G1_power": {"delta_fus_v31_halfwidth": hw, "fusion_beats_v31_significant": bool(power["delta_ci"]["delta_fus_v31"]["ci95_low"] > 0), "target_to_circular": float(0.9431 - fus), "circular_above_noise": bool((0.9431 - fus) > hw)},
            "P0_G2_ceiling": {"Q_statistic": ceiling["Q_statistic"], "v8b_correction_rate_of_v31_errors": ceiling["v8b_correction_rate_of_v31_errors"]},
            "P0_G3_c_premise": {"train_v31_error_rate": c_premise.get(TRAIN, {}).get("v31_error_rate"), "holdout_v31_error_rate": c_premise.get(HOLDOUT, {}).get("v31_error_rate")},
        },
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["gates"], indent=2))
    print(f"\nfusion {fus:.4f} | v31 {power['auroc_point']['v31']:.4f} | v8b {power['auroc_point']['v8b']:.4f}")
    print(f"delta(fus-v31) CI {power['delta_ci']['delta_fus_v31']}")
    print(f"Saved: {REPORT}")


if __name__ == "__main__":
    main()
