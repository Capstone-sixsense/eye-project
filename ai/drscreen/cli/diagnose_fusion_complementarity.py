"""Fusion complementarity diagnostics and cheap residual meta ablation.

This tool implements the Phase-C follow-up in
``.omc/plans/fusion_complementarity_plan.md`` after Phase 0 found a low
complementarity ceiling.  It does not modify active deployment aliases.

The implemented action is a calibration-fit residual weighting ablation:

- extract v31 scores and v8b scalar evidence features using an existing
  late-fusion config;
- reproduce the existing train-fit late-fusion baseline on the same holdout;
- fit the scalar meta-classifier on ``external_calibration`` instead of train;
- sweep v31-error and v31-residual sample weights;
- evaluate the unchanged ``external_holdout`` split and paired-bootstrap the
  deltas against both the train-fit baseline and the unweighted calibration-fit
  baseline.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from drscreen.cli.late_fusion_classifier import (
    _apply_calibration_split,
    _build_feature_matrix,
    _choose_threshold_by_policy,
    _extract_v31_scores,
)
from drscreen.cli.lesion_evidence_classifier import (
    _classification_report,
    _extract_features,
    _load_segmenter,
    _read_items,
)
from drscreen.infer.service import InferenceSession
from drscreen.settings import load_app_config, resolve_project_path


def _safe_auc(labels: np.ndarray, scores: np.ndarray) -> float | None:
    if len(np.unique(labels)) < 2:
        return None
    return float(roc_auc_score(labels, scores))


def _bootstrap_delta_auc(
    labels: np.ndarray,
    candidate_scores: np.ndarray,
    baseline_scores: np.ndarray,
    *,
    samples: int,
    seed: int,
) -> dict[str, float | None]:
    if len(np.unique(labels)) < 2:
        return {"mean": None, "ci95_low": None, "ci95_high": None, "half_width": None}
    rng = np.random.default_rng(seed)
    deltas: list[float] = []
    n = int(labels.size)
    for _ in range(samples):
        idx = rng.integers(0, n, size=n)
        y = labels[idx]
        if len(np.unique(y)) < 2:
            continue
        candidate_auc = roc_auc_score(y, candidate_scores[idx])
        baseline_auc = roc_auc_score(y, baseline_scores[idx])
        deltas.append(float(candidate_auc - baseline_auc))
    if not deltas:
        return {"mean": None, "ci95_low": None, "ci95_high": None, "half_width": None}
    arr = np.asarray(deltas, dtype=np.float64)
    low = float(np.percentile(arr, 2.5))
    high = float(np.percentile(arr, 97.5))
    return {
        "mean": float(np.mean(arr)),
        "ci95_low": low,
        "ci95_high": high,
        "half_width": float((high - low) / 2.0),
    }


def _q_statistic(labels: np.ndarray, a_scores: np.ndarray, b_scores: np.ndarray) -> dict[str, Any]:
    a_correct = ((a_scores >= 0.5).astype(np.int64) == labels)
    b_correct = ((b_scores >= 0.5).astype(np.int64) == labels)
    n11 = int(np.logical_and(a_correct, b_correct).sum())
    n00 = int(np.logical_and(~a_correct, ~b_correct).sum())
    n10 = int(np.logical_and(a_correct, ~b_correct).sum())
    n01 = int(np.logical_and(~a_correct, b_correct).sum())
    denom = (n11 * n00) + (n10 * n01)
    q_value = None if denom == 0 else ((n11 * n00) - (n10 * n01)) / denom
    a_wrong = int((~a_correct).sum())
    return {
        "Q_statistic": float(q_value) if q_value is not None else None,
        "disagreement": float(np.mean(a_correct != b_correct)),
        "v31_wrong_n": a_wrong,
        "v8b_correction_rate_of_v31_errors": (
            float(n01 / a_wrong) if a_wrong else None
        ),
        "counts": {
            "both_correct": n11,
            "both_wrong": n00,
            "only_v8b_correct": n01,
            "only_v31_correct": n10,
        },
    }


def _save_feature_cache(
    cache_path: Path,
    *,
    v31_scores: np.ndarray,
    v8b_features: np.ndarray,
    labels: np.ndarray,
    rows: list[dict[str, str | int]],
) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    rows_json = np.asarray(
        [json.dumps(row, sort_keys=True, ensure_ascii=True) for row in rows],
        dtype=np.str_,
    )
    np.savez_compressed(
        cache_path,
        v31_scores=v31_scores,
        v8b_features=v8b_features,
        labels=labels,
        rows_json=rows_json,
    )


def _load_feature_cache(cache_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict[str, str | int]]]:
    with np.load(cache_path, allow_pickle=False) as data:
        v31_scores = np.asarray(data["v31_scores"], dtype=np.float32)
        v8b_features = np.asarray(data["v8b_features"], dtype=np.float32)
        labels = np.asarray(data["labels"], dtype=np.int64)
        rows = [json.loads(str(row_json)) for row_json in data["rows_json"].tolist()]
    return v31_scores, v8b_features, labels, rows


def _mask_for_split_domains(
    *,
    row_splits: np.ndarray,
    row_domains: np.ndarray,
    split: str,
    domains: list[str] | None,
) -> np.ndarray:
    mask = row_splits == split
    if domains:
        mask &= np.isin(row_domains, [str(domain) for domain in domains])
    return mask


def _sample_weights(
    *,
    mode: str,
    factor: float,
    v31_probability: np.ndarray,
    labels: np.ndarray,
    train_mask: np.ndarray,
    threshold: float,
) -> np.ndarray | None:
    if mode == "none":
        return None
    weights = np.ones(labels.shape[0], dtype=np.float64)
    if mode == "v31_error":
        wrong = (v31_probability >= threshold).astype(np.int64) != labels
        weights[wrong] = float(factor)
        return weights
    if mode == "v31_margin":
        residual = np.abs(labels.astype(np.float64) - v31_probability.astype(np.float64))
        train_residual = residual[train_mask]
        max_residual = float(train_residual.max()) if train_residual.size else 0.0
        if max_residual <= 1e-8:
            return weights
        weights = 1.0 + (float(factor) - 1.0) * (residual / max_residual)
        return np.clip(weights, 1.0, float(factor))
    raise ValueError(f"Unsupported weighting mode: {mode}")


def _fit_one(
    *,
    features: np.ndarray,
    labels: np.ndarray,
    train_mask: np.ndarray,
    threshold_mask: np.ndarray,
    thresholds: np.ndarray,
    c_values: list[float],
    class_weight: str,
    seed: int,
    threshold_policy: str,
    sensitivity_guard: float | None,
    sample_weights: np.ndarray | None,
) -> tuple[Any, np.ndarray, dict, dict]:
    best_model = None
    best_scores = None
    best_selection = None
    best_rank = None
    candidates = []
    for c_value in c_values:
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=float(c_value),
                max_iter=5000,
                class_weight=class_weight,
                solver="lbfgs",
                random_state=seed,
            ),
        )
        fit_kwargs = {}
        if sample_weights is not None:
            fit_kwargs["logisticregression__sample_weight"] = sample_weights[train_mask]
        model.fit(features[train_mask], labels[train_mask], **fit_kwargs)
        scores = model.predict_proba(features)[:, 1]
        selection = _choose_threshold_by_policy(
            labels[threshold_mask],
            scores[threshold_mask],
            thresholds,
            policy=threshold_policy,
            sensitivity_guard=sensitivity_guard,
        )
        selected = selection["best"]
        val_report = _classification_report(
            labels[threshold_mask],
            scores[threshold_mask],
            float(selected["threshold"]),
        )
        if threshold_policy == "sensitivity_guard":
            guard_satisfied = selection["policy_status"] == "guard_satisfied"
            rank = (
                1.0 if guard_satisfied else 0.0,
                selected["specificity"] or 0.0,
                selected["f1"],
                val_report.get("auroc") or 0.0,
            )
        else:
            rank = (
                selected["balanced_accuracy"],
                selected["f1"],
                val_report.get("auroc") or 0.0,
            )
        candidates.append(
            {
                "c_value": float(c_value),
                "threshold_selection": selected,
                "threshold_split_metrics": val_report,
            }
        )
        if best_rank is None or rank > best_rank:
            best_rank = rank
            best_model = model
            best_scores = scores
            best_selection = selection
    assert best_model is not None
    assert best_scores is not None
    assert best_selection is not None
    return best_model, best_scores, best_selection, {"classifier_candidates": candidates}


def _variant_result(
    *,
    variant: dict[str, Any],
    features: np.ndarray,
    labels: np.ndarray,
    row_splits: np.ndarray,
    row_domains: np.ndarray,
    thresholds: np.ndarray,
    c_values: list[float],
    class_weight: str,
    seed: int,
    threshold_policy: str,
    sensitivity_guard: float | None,
    v31_scores: np.ndarray,
    v31_threshold: float,
    eval_split: str,
) -> tuple[dict[str, Any], np.ndarray]:
    train_split = str(variant["train_split"])
    threshold_split = str(variant["threshold_split"])
    train_domains = variant.get("train_domains")
    train_mask = _mask_for_split_domains(
        row_splits=row_splits,
        row_domains=row_domains,
        split=train_split,
        domains=train_domains,
    )
    threshold_mask = row_splits == threshold_split
    eval_mask = row_splits == eval_split
    if not train_mask.any():
        raise ValueError(f"No rows selected for variant {variant['name']} training.")
    if not threshold_mask.any():
        raise ValueError(f"No rows selected for variant {variant['name']} threshold calibration.")
    if not eval_mask.any():
        raise ValueError(f"No rows selected for eval split: {eval_split}")

    weights = _sample_weights(
        mode=str(variant["mode"]),
        factor=float(variant["factor"]),
        v31_probability=v31_scores[:, 0],
        labels=labels,
        train_mask=train_mask,
        threshold=v31_threshold,
    )
    model, scores, selection, extra = _fit_one(
        features=features,
        labels=labels,
        train_mask=train_mask,
        threshold_mask=threshold_mask,
        thresholds=thresholds,
        c_values=c_values,
        class_weight=class_weight,
        seed=seed,
        threshold_policy=threshold_policy,
        sensitivity_guard=sensitivity_guard,
        sample_weights=weights,
    )
    threshold = float(selection["best"]["threshold"])
    scaler = model.named_steps["standardscaler"]
    logreg = model.named_steps["logisticregression"]
    result = {
        "sample_weighting": {
            "name": variant["name"],
            "mode": variant["mode"],
            "factor": variant["factor"],
        },
        "train_split": train_split,
        "train_domains": sorted(set(row_domains[train_mask].tolist())),
        "configured_train_domains": train_domains,
        "threshold_split": threshold_split,
        "n_train_fit": int(train_mask.sum()),
        "threshold_selection": selection,
        "metrics_by_split": {
            train_split: _classification_report(labels[train_mask], scores[train_mask], threshold),
            threshold_split: _classification_report(labels[threshold_mask], scores[threshold_mask], threshold),
            eval_split: _classification_report(labels[eval_mask], scores[eval_mask], threshold),
        },
        "scaler_mean": scaler.mean_.astype(float).tolist(),
        "scaler_scale": scaler.scale_.astype(float).tolist(),
        "coef": logreg.coef_.astype(float).tolist(),
        "intercept": logreg.intercept_.astype(float).tolist(),
        "classes": logreg.classes_.astype(int).tolist(),
        **extra,
    }
    return result, scores


def run_calfit(args: argparse.Namespace) -> dict:
    config_path = Path(args.config).resolve()
    project_root = config_path.parents[1]
    config = load_app_config(config_path)
    data_cfg = config["data"]
    fusion_cfg = config.get("fusion", {})

    items = _read_items(config, project_root)
    items, calibration_info = _apply_calibration_split(
        items,
        data_cfg.get("calibration_split", {}),
        seed=int(fusion_cfg.get("seed", 43)),
    )

    classifier_session = InferenceSession.from_config_path(
        resolve_project_path(project_root, config["classifier"]["config_path"]),
        checkpoint_path=str(resolve_project_path(project_root, config["classifier"]["checkpoint_path"])),
    )
    seg_config_path = resolve_project_path(project_root, config["segmentation"]["config_path"])
    seg_config = load_app_config(seg_config_path)
    seg_checkpoint = resolve_project_path(project_root, config["segmentation"]["checkpoint_path"])
    segmenter = _load_segmenter(seg_config, project_root, seg_checkpoint)
    from drscreen.cli.lesion_evidence_classifier import _build_transform

    seg_transform = _build_transform(seg_config, project_root)
    batch_size = int(data_cfg.get("batch_size", 16))
    num_workers = int(data_cfg.get("num_workers", 0))

    cache_path = resolve_project_path(project_root, args.cache_npz) if args.cache_npz else None
    if args.reuse_cache and cache_path is not None and cache_path.exists():
        v31_scores, v8b_features, labels, rows = _load_feature_cache(cache_path)
        cache_info = {"path": str(cache_path), "loaded": True}
    else:
        v31_scores, labels, rows = _extract_v31_scores(
            classifier_session,
            items,
            batch_size=batch_size,
            num_workers=num_workers,
        )
        v8b_features, v8b_labels, v8b_rows = _extract_features(
            segmenter,
            items,
            transform=seg_transform,
            batch_size=batch_size,
            num_workers=num_workers,
        )
        if not np.array_equal(labels, v8b_labels):
            raise RuntimeError("v31 and v8b extraction labels are not aligned.")
        if [row["image_id"] for row in rows] != [row["image_id"] for row in v8b_rows]:
            raise RuntimeError("v31 and v8b extraction rows are not aligned.")
        cache_info = {"path": str(cache_path) if cache_path is not None else None, "loaded": False}
        if cache_path is not None:
            _save_feature_cache(
                cache_path,
                v31_scores=v31_scores,
                v8b_features=v8b_features,
                labels=labels,
                rows=rows,
            )
            cache_info["saved"] = True

    features, feature_names = _build_feature_matrix(
        "late_fusion",
        v31_scores=v31_scores,
        v8b_features=v8b_features,
    )
    row_splits = np.asarray([str(row["split"]) for row in rows])
    row_domains = np.asarray([str(row["domain"]) for row in rows])
    threshold_split = str(args.threshold_split)
    eval_split = str(args.eval_split)
    threshold_mask = row_splits == threshold_split
    eval_mask = row_splits == eval_split
    if not threshold_mask.any():
        raise ValueError(f"No rows for threshold split: {threshold_split}")
    if not eval_mask.any():
        raise ValueError(f"No rows for eval split: {eval_split}")

    threshold_cfg = fusion_cfg.get("thresholds", {})
    thresholds = np.linspace(
        float(threshold_cfg.get("min", 0.01)),
        float(threshold_cfg.get("max", 0.99)),
        int(threshold_cfg.get("steps", 99)),
    )
    c_values = [float(v) for v in fusion_cfg.get("c_values", [1.0])]
    seed = int(fusion_cfg.get("seed", 43))
    threshold_policy = str(fusion_cfg.get("threshold_policy", "balanced_accuracy"))
    sensitivity_guard = (
        float(fusion_cfg["sensitivity_guard"])
        if fusion_cfg.get("sensitivity_guard") is not None
        else None
    )

    train_variants = fusion_cfg.get("train_variants") or {}
    configured_train_domains = None
    trainfit_variant = train_variants.get(args.trainfit_variant)
    if isinstance(trainfit_variant, dict) and trainfit_variant.get("train_domains") is not None:
        configured_train_domains = [str(domain) for domain in trainfit_variant["train_domains"]]

    trainfit_name = f"trainfit_{args.trainfit_variant}"
    variants: list[dict[str, Any]] = [
        {
            "name": trainfit_name,
            "mode": "none",
            "factor": 1.0,
            "train_split": str(data_cfg.get("train_split", "train")),
            "train_domains": configured_train_domains,
            "threshold_split": threshold_split,
        },
        {
            "name": "calfit_none",
            "mode": "none",
            "factor": 1.0,
            "train_split": str(args.train_split),
            "train_domains": None,
            "threshold_split": threshold_split,
        },
    ]
    for factor in args.factors:
        variants.append(
            {
                "name": f"v31_error_w{factor:g}",
                "mode": "v31_error",
                "factor": factor,
                "train_split": str(args.train_split),
                "train_domains": None,
                "threshold_split": threshold_split,
            }
        )
    for factor in args.factors:
        variants.append(
            {
                "name": f"v31_margin_w{factor:g}",
                "mode": "v31_margin",
                "factor": factor,
                "train_split": str(args.train_split),
                "train_domains": None,
                "threshold_split": threshold_split,
            }
        )

    results = {}
    scores_by_variant = {}
    for variant in variants:
        result, scores = _variant_result(
            variant=variant,
            features=features,
            labels=labels,
            row_splits=row_splits,
            row_domains=row_domains,
            thresholds=thresholds,
            c_values=c_values,
            class_weight=str(fusion_cfg.get("class_weight", "balanced")),
            seed=seed,
            threshold_policy=threshold_policy,
            sensitivity_guard=sensitivity_guard,
            v31_scores=v31_scores,
            v31_threshold=float(args.v31_threshold),
            eval_split=eval_split,
        )
        results[str(variant["name"])] = result
        scores_by_variant[str(variant["name"])] = scores

    holdout_aurocs = {
        name: float(result["metrics_by_split"][eval_split]["auroc"])
        for name, result in results.items()
    }
    best_name = max(holdout_aurocs, key=lambda key: holdout_aurocs[key])
    none_scores = scores_by_variant["calfit_none"]
    labels_eval = labels[eval_mask]
    residual_names = [name for name in holdout_aurocs if name not in {trainfit_name, "calfit_none"}]
    best_residual_name = max(residual_names, key=lambda key: holdout_aurocs[key]) if residual_names else None
    best_residual_scores = (
        scores_by_variant[best_residual_name] if best_residual_name is not None else none_scores
    )
    residual_delta_ci = _bootstrap_delta_auc(
        labels_eval,
        best_residual_scores[eval_mask],
        none_scores[eval_mask],
        samples=int(args.bootstrap),
        seed=seed,
    )
    calfit_vs_trainfit_ci = _bootstrap_delta_auc(
        labels_eval,
        none_scores[eval_mask],
        scores_by_variant[trainfit_name][eval_mask],
        samples=int(args.bootstrap),
        seed=seed + 1,
    )

    output = {
        "config": str(config_path.relative_to(project_root)),
        "mode": "phase_c_calibration_fit_residual_ablation",
        "active_reference_auroc": float(args.active_reference_auroc),
        "feature_cache": cache_info,
        "calibration_split_info": calibration_info,
        "feature_names": feature_names,
        "n_images": int(len(items)),
        "split_counts": {
            split: int((row_splits == split).sum()) for split in sorted(set(row_splits.tolist()))
        },
        "v31_vs_calfit_fusion_holdout_complementarity": _q_statistic(
            labels_eval,
            v31_scores[eval_mask, 0],
            scores_by_variant["calfit_none"][eval_mask],
        ),
        "results": results,
        "holdout_aurocs": holdout_aurocs,
        "trainfit_baseline": {
            "name": trainfit_name,
            "auroc": holdout_aurocs[trainfit_name],
            "delta_vs_active_reference": holdout_aurocs[trainfit_name] - float(args.active_reference_auroc),
        },
        "calfit_none_vs_trainfit": {
            "delta_auc": holdout_aurocs["calfit_none"] - holdout_aurocs[trainfit_name],
            "paired_bootstrap_ci": calfit_vs_trainfit_ci,
        },
        "best_residual_vs_calfit_none": {
            "name": best_residual_name,
            "auroc": holdout_aurocs[best_residual_name] if best_residual_name else None,
            "delta_auc": (
                holdout_aurocs[best_residual_name] - holdout_aurocs["calfit_none"]
                if best_residual_name
                else None
            ),
            "paired_bootstrap_ci": residual_delta_ci,
        },
        "best_by_holdout_auroc": {
            "name": best_name,
            "auroc": holdout_aurocs[best_name],
            "delta_vs_active_reference": holdout_aurocs[best_name] - float(args.active_reference_auroc),
            "delta_vs_calfit_none_ci": (
                residual_delta_ci if best_name == best_residual_name else None
            ),
        },
    }
    residual_ci_low = residual_delta_ci.get("ci95_low")
    calfit_ci_low = calfit_vs_trainfit_ci.get("ci95_low")
    residual_pass = bool(
        best_residual_name is not None
        and holdout_aurocs[best_residual_name] > holdout_aurocs["calfit_none"]
        and residual_ci_low is not None
        and residual_ci_low > 0.0
    )
    calfit_policy_candidate = bool(
        holdout_aurocs["calfit_none"] > holdout_aurocs[trainfit_name]
        and calfit_ci_low is not None
        and calfit_ci_low > 0.0
    )
    output["decision"] = {
        "residual_complementarity_pass": residual_pass,
        "calfit_policy_candidate": calfit_policy_candidate,
        "proceed_to_B_or_A": residual_pass,
        "reason": (
            "Residual weighting significantly improves over unweighted cal-fit; B/A may be reconsidered."
            if residual_pass
            else "ABORT_CBA: residual weighting does not significantly improve over unweighted cal-fit."
        ),
        "calfit_policy_note": (
            "Unweighted calibration-fit beats the reproduced train-fit fusion on paired holdout; "
            "this is a threshold/meta-fit policy candidate, not residual-complementarity evidence."
            if calfit_policy_candidate
            else "Unweighted calibration-fit is not significantly better than the reproduced train-fit fusion."
        ),
    }

    output_path = resolve_project_path(project_root, args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output["best_by_holdout_auroc"], indent=2))
    print(json.dumps(output["decision"], indent=2))
    print(f"Saved: {output_path}")
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/v31_v8b_late_fusion_quickqual_v1.yaml")
    parser.add_argument(
        "--output",
        default=".omc/research/fusion_complementarity/phase_c_calfit_ablation.json",
    )
    parser.add_argument("--train-split", default="external_calibration")
    parser.add_argument("--trainfit-variant", default="classification_domains")
    parser.add_argument("--threshold-split", default="external_calibration")
    parser.add_argument("--eval-split", default="external_holdout")
    parser.add_argument("--factors", type=float, nargs="+", default=[2.0, 3.0, 5.0])
    parser.add_argument("--v31-threshold", type=float, default=0.5)
    parser.add_argument("--active-reference-auroc", type=float, default=0.9341)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument(
        "--cache-npz",
        default=".omc/research/fusion_complementarity/quickqual_v1_feature_cache.npz",
    )
    parser.add_argument("--reuse-cache", action="store_true")
    return parser.parse_args()


def main() -> None:
    run_calfit(parse_args())


if __name__ == "__main__":
    main()
