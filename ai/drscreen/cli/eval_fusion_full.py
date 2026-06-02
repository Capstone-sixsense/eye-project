"""Regression harness for the active v31+v8b fusion deployment.

This command intentionally starts from the compact deployment artifacts because
the active fusion checkpoint was promoted from a staged late-fusion sweep. It
records the current classification/XAI metrics, payload contract, optional
single-image inference smoke output, and optional local latency probe in one
JSON file.
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from dataclasses import fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    roc_auc_score,
)

from drscreen.cli.late_fusion_classifier import _apply_calibration_split
from drscreen.cli.lesion_evidence_classifier import _read_items
from drscreen.infer.payload import InferencePayload
from drscreen.infer.service import InferenceSession, _load_xai_eval_metrics
from drscreen.settings import (
    find_classification_metrics_path,
    load_app_config,
    resolve_project_path,
)

PATH_LIKE_PAYLOAD_KEYS = {
    "checkpoint_path",
    "prediction_path",
    "heatmap_path",
    "lesion_map_path",
}


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Expected object JSON: {path}")
    return data


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def _metric_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _payload_schema_keys() -> list[str]:
    return [field.name for field in fields(InferencePayload)]


def _payload_core(payload: dict[str, Any]) -> dict[str, Any]:
    """Return deterministic payload fields for contract/smoke checks."""

    core = {
        key: value
        for key, value in payload.items()
        if key not in PATH_LIKE_PAYLOAD_KEYS
    }
    if isinstance(core.get("lesion_summary"), dict):
        core["lesion_summary_keys"] = sorted(core["lesion_summary"].keys())
        # The full numeric summary is useful in manual debugging but too noisy
        # for the lightweight contract snapshot.
        del core["lesion_summary"]
    if isinstance(core.get("eval_metrics"), dict):
        core["eval_metrics_keys"] = sorted(core["eval_metrics"].keys())
        del core["eval_metrics"]
    return core


def _classification_snapshot(project_root: Path, version: str) -> dict[str, Any]:
    path = find_classification_metrics_path(
        project_root,
        version,
        split_name="external_test",
        checkpoint_stem="best",
        prefer_compact=True,
    )
    if not path.exists():
        raise FileNotFoundError(f"Classification metrics not found: {path}")
    data = _read_json(path)
    optimal = data.get("metrics_at_optimal_threshold", {})
    if not isinstance(optimal, dict):
        optimal = {}
    return {
        "source_path": str(path),
        "split": data.get("split"),
        "evaluation_split": data.get("evaluation_split"),
        "rows": data.get("rows"),
        "auroc": _metric_float(data.get("metrics", {}).get("auroc")),
        "optimal_threshold": _metric_float(data.get("optimal_threshold")),
        "accuracy": _metric_float(optimal.get("accuracy")),
        "sensitivity": _metric_float(optimal.get("sensitivity")),
        "specificity": _metric_float(optimal.get("specificity")),
        "precision": _metric_float(optimal.get("precision")),
        "f1": _metric_float(optimal.get("f1")),
        "confusion": {
            "tn": optimal.get("true_negative"),
            "fp": optimal.get("false_positive"),
            "fn": optimal.get("false_negative"),
            "tp": optimal.get("true_positive"),
        },
    }


def _xai_snapshot(project_root: Path, version: str, infer_cfg: dict[str, Any]) -> dict[str, Any]:
    metrics = _load_xai_eval_metrics(project_root, version, infer_cfg) or {}
    split = str(infer_cfg.get("xai_eval_split", "test"))
    evidence_type = str(infer_cfg.get("evidence_type", "")).strip().lower()
    compact_dir = project_root / "artifacts" / "evaluations"
    source_path = None
    if evidence_type in {"lesion_segmentation", "lesion_evidence", "segmentation"}:
        candidates = [
            compact_dir / f"xai_{version}_lesion_segmentation_{split}_best_metrics.json",
            compact_dir / f"xai_{version}_segmentation_{split}_best_metrics.json",
        ]
        source_path = next((str(path) for path in candidates if path.exists()), None)

    return {
        "source_path": source_path,
        "metrics": metrics,
    }


def _run_smoke(session: InferenceSession, image_path: Path) -> dict[str, Any]:
    prediction = session.predict_image_path(image_path, save_outputs=False)
    return {
        "image_path": str(image_path),
        "payload_keys": sorted(prediction.payload.keys()),
        "payload_core": _payload_core(prediction.payload),
        "has_heatmap_overlay": prediction.heatmap_overlay is not None,
    }


def _latency_probe(
    session: InferenceSession,
    image_path: Path,
    *,
    warmup: int,
    runs: int,
) -> dict[str, Any]:
    if runs <= 0:
        return {"enabled": False}

    for _ in range(max(0, warmup)):
        session.predict_image_path(image_path, save_outputs=False)
    if session.device.type == "cuda":
        torch.cuda.synchronize(session.device)

    durations_ms: list[float] = []
    for _ in range(runs):
        start = time.perf_counter()
        session.predict_image_path(image_path, save_outputs=False)
        if session.device.type == "cuda":
            torch.cuda.synchronize(session.device)
        durations_ms.append((time.perf_counter() - start) * 1000.0)

    durations_sorted = sorted(durations_ms)

    def percentile(pct: float) -> float:
        if not durations_sorted:
            return 0.0
        index = min(len(durations_sorted) - 1, max(0, round((pct / 100.0) * (len(durations_sorted) - 1))))
        return float(durations_sorted[index])

    return {
        "enabled": True,
        "image_path": str(image_path),
        "device": str(session.device),
        "warmup": int(warmup),
        "runs": int(runs),
        "mean_ms": float(statistics.fmean(durations_ms)),
        "p50_ms": float(statistics.median(durations_ms)),
        "p95_ms": percentile(95),
        "min_ms": float(min(durations_ms)),
        "max_ms": float(max(durations_ms)),
    }


def _predict_meta_probability(
    session: InferenceSession,
    image_tensor: torch.Tensor,
    *,
    option: str,
) -> float:
    if not hasattr(session.model, "predict_fusion_score"):
        raise RuntimeError("Option evaluation requires predict_fusion_score().")
    if option == "none":
        output = session.model.predict_fusion_score(image_tensor.unsqueeze(0), amp_enabled=False)
        probability = output.get("meta_probability")
        if probability is None:
            raise RuntimeError("Fusion model did not produce meta_probability.")
        return float(probability)
    if option == "amp":
        output = session.model.predict_fusion_score(image_tensor.unsqueeze(0), amp_enabled=True)
        probability = output.get("meta_probability")
        if probability is None:
            raise RuntimeError("Fusion model did not produce meta_probability.")
        return float(probability)
    if option == "hflip":
        output = session.model.predict_fusion_score(image_tensor.unsqueeze(0), amp_enabled=False)
        flipped = torch.flip(image_tensor, dims=[2])
        flipped_output = session.model.predict_fusion_score(flipped.unsqueeze(0), amp_enabled=False)
        probability = output.get("meta_probability")
        flipped_probability = flipped_output.get("meta_probability")
        if probability is None or flipped_probability is None:
            raise RuntimeError("Fusion model did not produce meta_probability.")
        return (float(probability) + float(flipped_probability)) / 2.0
    if option == "hflip_feature_recalc":
        output = session.model.predict_fusion_score(image_tensor.unsqueeze(0), amp_enabled=False)
        flipped = torch.flip(image_tensor, dims=[2])
        flipped_output = session.model.predict_fusion_score(flipped.unsqueeze(0), amp_enabled=False)
        cached_seg = output.get("seg_prob")
        flipped_seg = flipped_output.get("seg_prob")
        if not isinstance(cached_seg, torch.Tensor) or not isinstance(flipped_seg, torch.Tensor):
            raise RuntimeError("Fusion feature-recalc TTA requires seg_prob tensors.")
        averaged_seg = (cached_seg[0] + torch.flip(flipped_seg[0], dims=[2])) * 0.5
        recalc_output = session.model.predict_fusion_from_components(
            v31_probability=float(output["v31_probability"]),
            v31_logit=float(output["v31_logit"]),
            seg_prob=averaged_seg,
        )
        probability = recalc_output.get("meta_probability")
        if probability is None:
            raise RuntimeError("Fusion model did not produce recalculated meta_probability.")
        return float(probability)
    raise ValueError(f"Unsupported option: {option}")


def _stratified_limit_items(items: list[Any], *, max_images: int, seed: int) -> list[Any]:
    if max_images <= 0 or len(items) <= max_images:
        return items
    rng = np.random.default_rng(seed)
    by_label: dict[int, list[Any]] = {}
    for item in items:
        by_label.setdefault(int(item.label), []).append(item)
    selected: list[Any] = []
    remaining = max_images
    labels = sorted(by_label)
    for index, label in enumerate(labels):
        candidates = sorted(by_label[label], key=lambda item: item.image_id)
        quota = remaining if index == len(labels) - 1 else max(1, max_images // len(labels))
        quota = min(quota, len(candidates))
        chosen_indices = rng.choice(np.arange(len(candidates)), size=quota, replace=False)
        selected.extend(candidates[int(idx)] for idx in chosen_indices.tolist())
        remaining -= quota
    return sorted(selected[:max_images], key=lambda item: item.image_id)


def _option_eval(args: argparse.Namespace) -> dict[str, Any]:
    option = str(args.option_eval).strip().lower()
    if option not in {"amp", "hflip", "hflip_feature_recalc"}:
        raise ValueError("--option-eval must be one of: amp, hflip, hflip_feature_recalc")

    config_path = Path(args.config).resolve()
    project_root = config_path.parents[1]
    base_path = None
    candidate_base = config_path.parent / "base.yaml"
    if config_path.name != "base.yaml" and candidate_base.exists():
        base_path = candidate_base
    config = load_app_config(config_path, base_path=base_path)
    data_cfg = config.get("data", {})
    split_cfg = data_cfg.get("calibration_split") or {
        "source_split": "external_test",
        "calibration_split": "external_calibration",
        "holdout_split": "external_holdout",
        "fraction": 0.2,
        "seed": 43,
    }
    items = _read_items(config, project_root)
    source_split = str(split_cfg.get("source_split", "external_test"))
    if any(item.split == source_split for item in items):
        items, calibration_info = _apply_calibration_split(
            items,
            split_cfg,
            seed=int(split_cfg.get("seed", 43)),
        )
    else:
        calibration_info = None
    selected = [item for item in items if item.split == args.option_split]
    if not selected:
        raise ValueError(f"No rows selected for option split: {args.option_split}")
    selected = _stratified_limit_items(
        selected,
        max_images=int(args.max_images),
        seed=int(split_cfg.get("seed", 43)),
    )

    session = InferenceSession.from_config_path(config_path)
    baseline_scores: list[float] = []
    candidate_scores: list[float] = []
    labels: list[int] = []
    durations_ms: list[float] = []
    session.model.eval()
    with torch.no_grad():
        for item in selected:
            with Image.open(item.image_path) as image:
                tensor = session.eval_transform(image.convert("RGB")).to(session.device)
            if session.device.type == "cuda":
                torch.cuda.synchronize(session.device)
            start = time.perf_counter()
            baseline = _predict_meta_probability(session, tensor, option="none")
            candidate = _predict_meta_probability(session, tensor, option=option)
            if session.device.type == "cuda":
                torch.cuda.synchronize(session.device)
            durations_ms.append((time.perf_counter() - start) * 1000.0)
            baseline_scores.append(baseline)
            candidate_scores.append(candidate)
            labels.append(int(item.label))

    label_values = sorted(set(labels))
    baseline_auroc = roc_auc_score(labels, baseline_scores) if len(label_values) == 2 else None
    candidate_auroc = roc_auc_score(labels, candidate_scores) if len(label_values) == 2 else None
    deploy_threshold = float(config.get("infer", {}).get("threshold", 0.5))

    def report_at_threshold(scores: list[float]) -> dict[str, Any]:
        predictions = [int(score >= deploy_threshold) for score in scores]
        tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
        sensitivity = tp / (tp + fn) if (tp + fn) else None
        specificity = tn / (tn + fp) if (tn + fp) else None
        precision = precision_score(labels, predictions, zero_division=0)
        f1 = f1_score(labels, predictions, zero_division=0)
        return {
            "threshold": deploy_threshold,
            "accuracy": float(accuracy_score(labels, predictions)),
            "sensitivity": None if sensitivity is None else float(sensitivity),
            "specificity": None if specificity is None else float(specificity),
            "precision": float(precision),
            "f1": float(f1),
            "true_negative": int(tn),
            "false_positive": int(fp),
            "false_negative": int(fn),
            "true_positive": int(tp),
        }

    def select_by_sensitivity_guard(scores: list[float]) -> dict[str, Any] | None:
        guard = args.sensitivity_guard
        if guard is None:
            return None
        thresholds = np.linspace(
            float(args.threshold_min),
            float(args.threshold_max),
            int(args.threshold_steps),
        )
        reports = []
        for threshold in thresholds:
            original_threshold = deploy_threshold
            try:
                deploy_threshold_local = float(threshold)
                predictions = [int(score >= deploy_threshold_local) for score in scores]
                tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
                sensitivity = tp / (tp + fn) if (tp + fn) else None
                specificity = tn / (tn + fp) if (tn + fp) else None
                precision = precision_score(labels, predictions, zero_division=0)
                f1 = f1_score(labels, predictions, zero_division=0)
                reports.append(
                    {
                        "threshold": deploy_threshold_local,
                        "accuracy": float(accuracy_score(labels, predictions)),
                        "sensitivity": None if sensitivity is None else float(sensitivity),
                        "specificity": None if specificity is None else float(specificity),
                        "precision": float(precision),
                        "f1": float(f1),
                        "true_negative": int(tn),
                        "false_positive": int(fp),
                        "false_negative": int(fn),
                        "true_positive": int(tp),
                    }
                )
            finally:
                _ = original_threshold
        guarded = [
            report for report in reports
            if report["sensitivity"] is not None
            and float(report["sensitivity"]) >= float(guard)
        ]
        if not guarded:
            return {
                "policy": "sensitivity_guard",
                "guard": float(guard),
                "status": "guard_unmet",
                "selected": None,
            }
        selected = max(guarded, key=lambda row: (row["specificity"] or 0.0, row["f1"]))
        return {
            "policy": "sensitivity_guard",
            "guard": float(guard),
            "status": "guard_satisfied",
            "selected": selected,
        }

    baseline_report = report_at_threshold(baseline_scores)
    candidate_report = report_at_threshold(candidate_scores)
    drifts = np.abs(np.asarray(candidate_scores) - np.asarray(baseline_scores))
    sorted_durations = sorted(durations_ms)
    p95_index = min(
        len(sorted_durations) - 1,
        max(0, round(0.95 * (len(sorted_durations) - 1))),
    )
    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config_path": str(config_path),
        "version": str(config.get("project", {}).get("version", "")),
        "option": option,
        "split": args.option_split,
        "max_images": int(args.max_images),
        "n_images": len(selected),
        "label_counts": {
            str(label): int(sum(1 for value in labels if value == label))
            for label in label_values
        },
        "calibration_split_info": calibration_info,
        "baseline_auroc": None if baseline_auroc is None else float(baseline_auroc),
        "candidate_auroc": None if candidate_auroc is None else float(candidate_auroc),
        "auroc_delta": (
            None
            if baseline_auroc is None or candidate_auroc is None
            else float(candidate_auroc - baseline_auroc)
        ),
        "deploy_threshold": deploy_threshold,
        "baseline_at_deploy_threshold": baseline_report,
        "candidate_at_deploy_threshold": candidate_report,
        "baseline_sensitivity_guard_selection": select_by_sensitivity_guard(baseline_scores),
        "candidate_sensitivity_guard_selection": select_by_sensitivity_guard(candidate_scores),
        "probability_drift": {
            "mean_abs": float(drifts.mean()),
            "max_abs": float(drifts.max()),
            "p95_abs": float(np.quantile(drifts, 0.95)),
        },
        "paired_latency_ms": {
            "mean": float(statistics.fmean(durations_ms)),
            "p50": float(statistics.median(durations_ms)),
            "p95": float(sorted_durations[p95_index]),
        },
    }


def _metric_at(snapshot: dict[str, Any], path: tuple[str, ...]) -> float | None:
    current: Any = snapshot
    for key in path:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return _metric_float(current)


def _delta(candidate: dict[str, Any], baseline: dict[str, Any], path: tuple[str, ...]) -> dict[str, Any]:
    candidate_value = _metric_at(candidate, path)
    baseline_value = _metric_at(baseline, path)
    return {
        "candidate": candidate_value,
        "baseline": baseline_value,
        "delta": (
            None
            if candidate_value is None or baseline_value is None
            else candidate_value - baseline_value
        ),
    }


def _regression_check(baseline_path: Path, candidate_path: Path) -> dict[str, Any]:
    baseline = _read_json(baseline_path)
    candidate = _read_json(candidate_path)
    comparisons = {
        "classification.auroc": _delta(candidate, baseline, ("classification", "auroc")),
        "classification.sensitivity": _delta(candidate, baseline, ("classification", "sensitivity")),
        "classification.specificity": _delta(candidate, baseline, ("classification", "specificity")),
        "classification.optimal_threshold": _delta(candidate, baseline, ("classification", "optimal_threshold")),
        "xai.xai_auc_iou": _delta(candidate, baseline, ("xai", "metrics", "xai_auc_iou")),
        "xai.xai_seg_mdice": _delta(candidate, baseline, ("xai", "metrics", "xai_seg_mdice")),
        "xai.xai_seg_union_iou": _delta(candidate, baseline, ("xai", "metrics", "xai_seg_union_iou")),
    }
    schema_baseline = baseline.get("payload_schema_keys", [])
    schema_candidate = candidate.get("payload_schema_keys", [])
    schema_identical = schema_baseline == schema_candidate

    auroc_delta = comparisons["classification.auroc"]["delta"]
    xai_auc_delta = comparisons["xai.xai_auc_iou"]["delta"]
    xai_mdice_delta = comparisons["xai.xai_seg_mdice"]["delta"]
    xai_union_delta = comparisons["xai.xai_seg_union_iou"]["delta"]
    sensitivity = comparisons["classification.sensitivity"]["candidate"]
    specificity = comparisons["classification.specificity"]["candidate"]
    promotion_pass = (
        auroc_delta is not None
        and auroc_delta >= 0.001
        and sensitivity is not None
        and sensitivity >= 0.80
        and specificity is not None
        and specificity >= 0.91
        and xai_auc_delta is not None
        and xai_auc_delta >= -0.005
        and xai_mdice_delta is not None
        and xai_mdice_delta >= -0.005
        and xai_union_delta is not None
        and xai_union_delta >= -0.005
        and schema_identical
    )
    return {
        "baseline_path": str(baseline_path),
        "candidate_path": str(candidate_path),
        "comparisons": comparisons,
        "payload_schema_identical": schema_identical,
        "promotion_gate_pass": promotion_pass,
    }


def build_snapshot(args: argparse.Namespace) -> dict[str, Any]:
    config_path = Path(args.config).resolve()
    project_root = config_path.parents[1]
    base_path = None
    candidate_base = config_path.parent / "base.yaml"
    if config_path.name != "base.yaml" and candidate_base.exists():
        base_path = candidate_base
    config = load_app_config(config_path, base_path=base_path)
    version = str(config.get("project", {}).get("version", ""))
    infer_cfg = config.get("infer", {})
    snapshot: dict[str, Any] = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config_path": str(config_path),
        "project_root": str(project_root),
        "version": version,
        "infer": {
            "threshold": infer_cfg.get("threshold"),
            "checkpoint_path": str(resolve_project_path(project_root, infer_cfg.get("checkpoint_path", ""))),
            "use_meta_classifier": infer_cfg.get("use_meta_classifier"),
            "evidence_type": infer_cfg.get("evidence_type"),
            "lesion_threshold": infer_cfg.get("lesion_threshold"),
        },
        "classification": _classification_snapshot(project_root, version),
        "xai": _xai_snapshot(project_root, version, infer_cfg),
        "payload_schema_keys": _payload_schema_keys(),
    }

    smoke_image = Path(args.image).resolve() if args.image else None
    latency_image = Path(args.latency_image).resolve() if args.latency_image else smoke_image
    if smoke_image is not None or (latency_image is not None and args.latency_runs > 0):
        session = InferenceSession.from_config_path(config_path)
        if smoke_image is not None:
            snapshot["single_image_smoke"] = _run_smoke(session, smoke_image)
        if latency_image is not None:
            snapshot["latency"] = _latency_probe(
                session,
                latency_image,
                warmup=int(args.latency_warmup),
                runs=int(args.latency_runs),
            )
    else:
        snapshot["latency"] = {"enabled": False}

    return snapshot


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate active fusion deployment contract.")
    parser.add_argument("--config", default="configs/base.yaml", help="Runtime YAML config.")
    parser.add_argument("--output", help="Snapshot output JSON path.")
    parser.add_argument("--image", help="Optional single-image inference smoke input.")
    parser.add_argument("--latency-image", help="Optional latency probe image. Defaults to --image.")
    parser.add_argument("--latency-warmup", type=int, default=5)
    parser.add_argument("--latency-runs", type=int, default=0)
    parser.add_argument("--baseline", help="Baseline snapshot for promotion/regression check.")
    parser.add_argument("--candidate", help="Candidate snapshot for promotion/regression check.")
    parser.add_argument("--regression-check", action="store_true", help="Compare --candidate against --baseline.")
    parser.add_argument(
        "--option-eval",
        choices=["amp", "hflip", "hflip_feature_recalc"],
        help="Evaluate an inference option on manifest rows.",
    )
    parser.add_argument("--option-split", default="external_holdout")
    parser.add_argument("--max-images", type=int, default=0, help="0 means all selected rows.")
    parser.add_argument("--sensitivity-guard", type=float, default=None)
    parser.add_argument("--threshold-min", type=float, default=0.01)
    parser.add_argument("--threshold-max", type=float, default=0.99)
    parser.add_argument("--threshold-steps", type=int, default=99)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.regression_check:
        if not args.baseline or not args.candidate:
            raise SystemExit("--regression-check requires --baseline and --candidate.")
        result = _regression_check(Path(args.baseline).resolve(), Path(args.candidate).resolve())
        if args.output:
            _write_json(Path(args.output), result)
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return
    if args.option_eval:
        result = _option_eval(args)
        if args.output:
            _write_json(Path(args.output), result)
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return

    snapshot = build_snapshot(args)
    if args.output:
        _write_json(Path(args.output), snapshot)
    print(json.dumps(snapshot, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
