from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from drscreen.infer.late_fusion_features import FUSION_AREA_THRESHOLDS, FUSION_TOPK_FRACS


def _extract_state_dict(checkpoint: Any) -> dict[str, torch.Tensor]:
    if isinstance(checkpoint, dict):
        for key in ("model_state_dict", "state_dict", "model"):
            value = checkpoint.get(key)
            if isinstance(value, dict):
                return value
        if checkpoint and all(torch.is_tensor(v) for v in checkpoint.values()):
            return checkpoint
    raise ValueError("Unsupported checkpoint format; expected a state_dict payload.")


def _best_result(metrics: dict[str, Any], key: str | None) -> tuple[str, dict[str, Any]]:
    results = metrics.get("results")
    if not isinstance(results, dict) or not results:
        raise ValueError("metrics JSON does not contain results.")
    if key is None:
        key = (metrics.get("best_by_primary_eval_auroc") or {}).get("key")
    if not key:
        key = (metrics.get("best_by_external_auroc") or {}).get("key")
    if not key or key not in results:
        raise ValueError(f"Unable to resolve fusion result key: {key!r}")
    return key, results[key]


def _read_meta_classifier(metrics_path: Path, key: str | None) -> tuple[str, dict[str, Any], list[str]]:
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    result_key, result = _best_result(metrics, key)
    feature_names = list(result.get("feature_names") or [])
    if not feature_names:
        raise ValueError(f"{result_key} has no feature_names.")
    required = ("scaler_mean", "scaler_scale", "coef", "intercept", "classes")
    missing = [name for name in required if name not in result]
    if missing:
        raise ValueError(
            f"{metrics_path} is missing {missing}. Re-run late_fusion_classifier.py "
            "after the scaler export patch."
        )
    n_features = len(feature_names)
    if len(result["scaler_mean"]) != n_features or len(result["scaler_scale"]) != n_features:
        raise ValueError("Scaler parameter length does not match feature_names.")
    coef = result["coef"]
    if not coef or len(coef[0]) != n_features:
        raise ValueError("Coefficient length does not match feature_names.")
    meta_classifier = {
        "kind": "standardscaler_logisticregression_numeric",
        "scaler_mean": result["scaler_mean"],
        "scaler_scale": result["scaler_scale"],
        "coef": result["coef"],
        "intercept": result["intercept"],
        "classes": result["classes"],
    }
    return result_key, meta_classifier, feature_names


def build(args: argparse.Namespace) -> Path:
    classifier_path = Path(args.classifier_ckpt).resolve()
    segmenter_path = Path(args.segmenter_ckpt).resolve()
    metrics_path = Path(args.metrics_json).resolve()
    output_path = Path(args.output).resolve()

    classifier_ckpt = torch.load(classifier_path, map_location="cpu", weights_only=False)
    segmenter_ckpt = torch.load(segmenter_path, map_location="cpu", weights_only=False)
    result_key, meta_classifier, feature_schema = _read_meta_classifier(
        metrics_path,
        args.result_key,
    )

    payload = {
        "fusion_version": "v31_v8b_score_level_fusion_v2_1",
        "architecture": "v31_v8b_fusion",
        "num_outputs": 1,
        "label_names": ["normal", "abnormal"],
        "model_state_dict": {},
        "classifier_state_dict": _extract_state_dict(classifier_ckpt),
        "segmenter_state_dict": _extract_state_dict(segmenter_ckpt),
        "meta_classifier": meta_classifier,
        "feature_schema": feature_schema,
        "feature_extraction": {
            "area_thresholds": list(FUSION_AREA_THRESHOLDS),
            "topk_fracs": list(FUSION_TOPK_FRACS),
            "image_preprocessing": "preprocessed_manifest_eval_transform",
        },
        "decision_threshold": float(args.fusion_threshold),
        "optimal_threshold": float(args.fusion_threshold),
        "thresholds": {
            "fusion_score": float(args.fusion_threshold),
            "v31_legacy": float(args.v31_threshold),
            "lesion_global": float(args.lesion_threshold),
        },
        "sources": {
            "classifier_checkpoint": str(classifier_path),
            "segmenter_checkpoint": str(segmenter_path),
            "metrics_json": str(metrics_path),
            "metrics_result_key": result_key,
        },
        "provenance": {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "tool": "drscreen.cli.build_fusion_checkpoint",
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output_path)
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a v31+v8b score-fusion checkpoint.")
    parser.add_argument("--classifier-ckpt", required=True)
    parser.add_argument("--segmenter-ckpt", required=True)
    parser.add_argument("--metrics-json", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--result-key", default=None)
    parser.add_argument("--fusion-threshold", type=float, default=0.38)
    parser.add_argument("--v31-threshold", type=float, default=0.35)
    parser.add_argument("--lesion-threshold", type=float, default=0.5)
    return parser.parse_args()


def main() -> None:
    output_path = build(parse_args())
    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"Wrote {output_path} ({size_mb:.1f} MiB)")


if __name__ == "__main__":
    main()
