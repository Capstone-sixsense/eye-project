"""검증(validation)/holdout 분할 선택의 건전성 진단.

체크포인트 선택에 쓰는 검증 분할이 holdout/외부 테스트를 잘 대변하는지(선택 기준이
실제 일반화 성능과 어긋나지 않는지)를 점검한다. 검증-테스트 분포 괴리로 인한 잘못된
모델 선택을 사전에 잡기 위한 도구다.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from drscreen.settings import (
    build_effective_checkpoint_config,
    ensure_runtime_directories,
    load_app_config,
    resolve_checkpoint_path,
    resolve_project_path,
)
from drscreen.train.data_loader_factory import build_eval_dataset
from drscreen.train.engine import collect_logits_and_targets
from drscreen.train.metrics import compute_binary_classification_metrics
from drscreen.train.model_setup import (
    build_model_for_eval,
    resolve_device,
    validate_training_scope,
)
from drscreen.utils.checkpoint import load_state_from_checkpoint


def _average_ranks(values: list[float]) -> np.ndarray:
    order = np.argsort(np.asarray(values, dtype=float), kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2.0 + 1.0
        start = end
    return ranks


def _spearman(x: list[float], y: list[float]) -> float | None:
    if len(x) != len(y) or len(x) < 2:
        return None
    rx = _average_ranks(x)
    ry = _average_ranks(y)
    if float(rx.std()) == 0.0 or float(ry.std()) == 0.0:
        return None
    return float(np.corrcoef(rx, ry)[0, 1])


def _checkpoint_paths_from_summary(project_root: Path, summary_path: Path) -> list[Path]:
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    paths: list[Path] = []
    for value in summary.get("recent_checkpoint_paths") or []:
        paths.append(resolve_project_path(project_root, value))
    if paths:
        return paths
    paths.extend(sorted(summary_path.parent.glob("epoch_*.pt")))
    if paths:
        return paths
    for key in ("best_checkpoint_path", "last_checkpoint_path"):
        value = summary.get(key)
        if value:
            paths.append(resolve_project_path(project_root, value))
    return paths


def _evaluate_checkpoint_splits(
    config: dict[str, Any],
    *,
    config_path: Path,
    project_root: Path,
    checkpoint_path: Path,
    splits: list[str],
    threshold: float,
) -> list[dict[str, Any]]:
    resolved_checkpoint = resolve_checkpoint_path(project_root, checkpoint_path)
    checkpoint = torch.load(resolved_checkpoint, map_location="cpu", weights_only=False)
    effective_config = build_effective_checkpoint_config(config, checkpoint)
    validate_training_scope(effective_config)

    device_name = str(
        effective_config.get("infer", {}).get("device")
        or effective_config.get("train", {}).get("device", "cpu")
    )
    device = resolve_device(device_name)
    model = build_model_for_eval(effective_config, device)
    load_state_from_checkpoint(model, checkpoint)
    amp_enabled = bool(effective_config["train"].get("amp", False)) and device.type == "cuda"

    results: list[dict[str, Any]] = []
    for split in splits:
        dataset, manifest_path = build_eval_dataset(effective_config, project_root, split)
        loader = DataLoader(
            dataset,
            batch_size=int(effective_config["data"]["batch_size"]),
            shuffle=False,
            num_workers=int(effective_config["data"].get("num_workers", 0)),
            pin_memory=device.type == "cuda",
            persistent_workers=int(effective_config["data"].get("num_workers", 0)) > 0,
        )
        logits, targets = collect_logits_and_targets(model, loader, device, amp_enabled=amp_enabled)
        metrics = compute_binary_classification_metrics(logits, targets, threshold=threshold)
        results.append(
            {
                "checkpoint": str(resolved_checkpoint),
                "split": str(split),
                "manifest_path": str(manifest_path),
                "rows": len(dataset),
                "auroc": metrics.auroc,
                "metrics": metrics.to_dict(),
            }
        )
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnose validation/holdout selection sanity.")
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    parser.add_argument("--checkpoints", nargs="*", help="Checkpoint paths to evaluate.")
    parser.add_argument("--training-summary", help="Optional training_summary.json for best/last checkpoints.")
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["val_mixed", "external_calibration", "external_holdout"],
        help="Splits to evaluate for each checkpoint.",
    )
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--output", required=True, help="Output JSON path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).resolve()
    project_root = config_path.parents[1]
    base_path = None
    candidate_base = config_path.parent / "base.yaml"
    if config_path.name != "base.yaml" and candidate_base.exists():
        base_path = candidate_base

    config = load_app_config(config_path, base_path=base_path)
    ensure_runtime_directories(config, project_root)

    checkpoints = [
        Path(value).resolve() if Path(value).is_absolute() else resolve_project_path(project_root, value)
        for value in (args.checkpoints or [])
    ]
    if args.training_summary:
        checkpoints.extend(_checkpoint_paths_from_summary(project_root, Path(args.training_summary).resolve()))
    deduped: list[Path] = []
    seen: set[Path] = set()
    for checkpoint in checkpoints:
        if checkpoint in seen:
            continue
        seen.add(checkpoint)
        deduped.append(checkpoint)
    if not deduped:
        raise ValueError("Provide --checkpoints or --training-summary.")

    evaluations: list[dict[str, Any]] = []
    by_checkpoint: dict[str, dict[str, float]] = {}
    for checkpoint in deduped:
        split_scores: dict[str, float] = {}
        checkpoint_evaluations = _evaluate_checkpoint_splits(
            config,
            config_path=config_path,
            project_root=project_root,
            checkpoint_path=checkpoint,
            splits=[str(split) for split in args.splits],
            threshold=float(args.threshold),
        )
        for result in checkpoint_evaluations:
            split = str(result["split"])
            auroc = result.get("auroc")
            split_scores[str(split)] = float(auroc) if auroc is not None else float("nan")
            evaluations.append(result)
        by_checkpoint[str(checkpoint)] = split_scores

    correlations: dict[str, float | None] = {}
    holdout_split = "external_holdout"
    if holdout_split in args.splits:
        holdout = [scores[holdout_split] for scores in by_checkpoint.values()]
        for split in args.splits:
            if split == holdout_split:
                continue
            values = [scores[str(split)] for scores in by_checkpoint.values()]
            if any(np.isnan(v) for v in values + holdout):
                correlations[f"{split}_vs_{holdout_split}"] = None
            else:
                correlations[f"{split}_vs_{holdout_split}"] = _spearman(values, holdout)

    output = {
        "config_path": str(config_path),
        "checkpoints": [str(path) for path in deduped],
        "splits": [str(split) for split in args.splits],
        "evaluations": evaluations,
        "spearman": correlations,
    }
    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"Selection sanity written: {output_path}")


if __name__ == "__main__":
    main()
