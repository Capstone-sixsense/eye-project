from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from drscreen.settings import (
    build_effective_checkpoint_config,
    classification_metrics_filename,
    get_run_evaluation_dir,
    resolve_checkpoint_path,
    resolve_project_path,
)
from drscreen.train.data_loader_factory import build_eval_dataset
from drscreen.train.engine import collect_logits_and_targets, evaluate_one_epoch
from drscreen.train.metrics import (
    compute_binary_classification_metrics,
    find_optimal_threshold,
)
from drscreen.train.model_setup import (
    build_criterion,
    build_model_for_eval,
    resolve_device,
    validate_training_scope,
)
from drscreen.utils.checkpoint import load_state_from_checkpoint


def run_split_evaluation(
    config: dict[str, Any],
    *,
    config_path: Path,
    project_root: Path,
    split_name: str | None = None,
    checkpoint_path: Path | None = None,
    threshold: float = 0.5,
) -> dict[str, Any]:
    requested_split = split_name or str(config["data"]["test_split"])
    resolved_checkpoint_path = resolve_checkpoint_path(
        project_root,
        checkpoint_path or config["infer"]["checkpoint_path"],
    )
    checkpoint = torch.load(resolved_checkpoint_path, map_location="cpu", weights_only=False)
    effective_config = build_effective_checkpoint_config(config, checkpoint)
    validate_training_scope(effective_config)

    device_name = str(
        effective_config.get("infer", {}).get("device")
        or effective_config.get("train", {}).get("device", "cpu")
    )
    device = resolve_device(device_name)
    dataset, manifest_path = build_eval_dataset(effective_config, project_root, requested_split)

    loader = DataLoader(
        dataset,
        batch_size=int(effective_config["data"]["batch_size"]),
        shuffle=False,
        num_workers=int(effective_config["data"].get("num_workers", 0)),
        pin_memory=device.type == "cuda",
        persistent_workers=int(effective_config["data"].get("num_workers", 0)) > 0,
    )
    model = build_model_for_eval(effective_config, device)
    load_state_from_checkpoint(model, checkpoint)

    criterion = build_criterion(effective_config).to(device)
    amp_enabled = bool(effective_config["train"].get("amp", False)) and device.type == "cuda"

    logits, targets = collect_logits_and_targets(model, loader, device, amp_enabled=amp_enabled)
    optimal_threshold = find_optimal_threshold(logits, targets)

    metrics = evaluate_one_epoch(
        model, loader, criterion, device, amp_enabled=amp_enabled, threshold=threshold,
    )
    metrics_at_optimal = evaluate_one_epoch(
        model, loader, criterion, device, amp_enabled=amp_enabled, threshold=optimal_threshold,
    )

    domain_breakdown: dict[str, Any] | None = None
    if "domain" in dataset.frame.columns:
        domain_breakdown = {}
        domain_series = dataset.frame["domain"].tolist()
        unique_domains = sorted({str(d) for d in domain_series if d is not None and str(d) != "nan"})
        flat_logits = logits.view(-1)
        flat_targets = targets.view(-1)
        for domain in unique_domains:
            indices = torch.tensor(
                [i for i, d in enumerate(domain_series) if str(d) == domain],
                dtype=torch.long,
            )
            domain_metrics = compute_binary_classification_metrics(
                flat_logits[indices], flat_targets[indices], threshold=threshold
            )
            domain_breakdown[domain] = domain_metrics.to_dict()

    version = str(effective_config.get("project", {}).get("version", "")).strip()
    evaluation_dir = (
        get_run_evaluation_dir(project_root, version)
        if version
        else resolve_project_path(project_root, effective_config["project"]["output_root"]) / "evaluations"
    )
    evaluation_dir.mkdir(parents=True, exist_ok=True)
    output_path = evaluation_dir / classification_metrics_filename(
        requested_split,
        version or resolved_checkpoint_path.parent.name,
        resolved_checkpoint_path.stem,
    )

    summary: dict[str, Any] = {
        "project_root": str(project_root),
        "config_path": str(config_path),
        "manifest_path": str(manifest_path),
        "split": requested_split,
        "rows": len(dataset),
        "device": str(device),
        "checkpoint_path": str(resolved_checkpoint_path),
        "label_names": list(effective_config["labels"]["names"]),
        "metrics": metrics.to_dict(),
        "optimal_threshold": optimal_threshold,
        "metrics_at_optimal_threshold": metrics_at_optimal.to_dict(),
    }
    if domain_breakdown is not None:
        summary["domain_breakdown"] = domain_breakdown
    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    summary["output_path"] = str(output_path)
    return summary
