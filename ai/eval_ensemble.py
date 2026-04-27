"""Ensemble evaluation: average logits from two checkpoints on a given split.

Usage:
    python eval_ensemble.py \
        --config1 configs/v9_fda.yaml \
        --config2 configs/v10_swad.yaml \
        --split external_test \
        --out artifacts/evaluations/ensemble_v9v10/external_test.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from drscreen.data.datasets import ManifestDataset
from drscreen.data.transforms import build_eval_transform
from drscreen.models.build import build_model
from drscreen.models.profiles import get_model_profile
from drscreen.settings import (
    build_effective_checkpoint_config,
    load_app_config,
    resolve_project_path,
)
from drscreen.train.engine import collect_logits_and_targets
from drscreen.train.metrics import compute_binary_classification_metrics, find_optimal_threshold
from drscreen.train.runner import resolve_device
from drscreen.utils.checkpoint import load_state_from_checkpoint


def _load_model_and_logits(
    config_path: Path,
    project_root: Path,
    split_name: str,
    shared_dataset: ManifestDataset,
    device: torch.device,
    amp_enabled: bool,
) -> torch.Tensor:
    base_path = config_path.parent / "base.yaml"
    config = load_app_config(
        config_path,
        base_path=base_path if config_path.name != "base.yaml" and base_path.exists() else None,
    )
    ckpt_path = resolve_project_path(project_root, config["infer"]["checkpoint_path"])
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    eff_cfg = build_effective_checkpoint_config(config, checkpoint)

    model = build_model(
        str(eff_cfg["model"]["architecture"]),
        pretrained=False,
        num_outputs=int(eff_cfg["model"]["num_outputs"]),
        use_attention=bool(eff_cfg["model"].get("use_attention", False)),
        use_mixstyle=bool(eff_cfg["model"].get("use_mixstyle", False)),
        use_ibn=bool(eff_cfg["model"].get("use_ibn", False)),
        classifier_dropout=float(eff_cfg["model"].get("classifier_dropout", 0.0)),
    ).to(device)
    load_state_from_checkpoint(model, checkpoint)

    profile = get_model_profile(str(eff_cfg["model"]["architecture"]))
    eval_transform = build_eval_transform(
        crop_size=int(eff_cfg["data"]["image_size"]),
        resize_size=int(eff_cfg["data"].get("resize_size", eff_cfg["data"]["image_size"])),
        interpolation=profile.interpolation,
        use_preprocessing=bool(eff_cfg["data"].get("use_preprocessing", False)),
    )
    shared_dataset.transform = eval_transform

    loader = DataLoader(
        shared_dataset,
        batch_size=int(eff_cfg["data"].get("batch_size", 16)),
        shuffle=False,
        num_workers=int(eff_cfg["data"].get("num_workers", 0)),
        pin_memory=device.type == "cuda",
    )
    logits, targets = collect_logits_and_targets(model, loader, device, amp_enabled=amp_enabled)
    print(f"  [{ckpt_path.parent.name}] logits shape={logits.shape}, mean={logits.mean():.4f}")
    return logits, targets


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config1", required=True)
    parser.add_argument("--config2", required=True)
    parser.add_argument("--split", default="external_test")
    parser.add_argument("--out", default="artifacts/evaluations/ensemble_v9v10/external_test.json")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    config1_path = Path(args.config1).resolve()
    config2_path = Path(args.config2).resolve()
    project_root = config1_path.parents[1]

    device = resolve_device(args.device)
    amp_enabled = device.type == "cuda"

    # Build shared dataset using config1's manifest/image_root settings
    base_path = config1_path.parent / "base.yaml"
    base_config = load_app_config(
        config1_path,
        base_path=base_path if config1_path.name != "base.yaml" and base_path.exists() else None,
    )
    manifest_path = resolve_project_path(project_root, base_config["data"]["manifest_path"])
    image_root = resolve_project_path(project_root, base_config["data"]["image_root"])

    # Placeholder transform — overridden per model inside _load_model_and_logits
    shared_dataset = ManifestDataset(
        manifest_path=manifest_path,
        image_root=image_root,
        split=args.split,
        transform=None,
    )
    print(f"Split '{args.split}': {len(shared_dataset)} examples")

    print("Loading model 1...")
    logits1, targets = _load_model_and_logits(
        config1_path, project_root, args.split, shared_dataset, device, amp_enabled
    )
    print("Loading model 2...")
    logits2, _ = _load_model_and_logits(
        config2_path, project_root, args.split, shared_dataset, device, amp_enabled
    )

    # Average logits (pre-sigmoid), then find optimal threshold
    ensemble_logits = (logits1 + logits2) / 2.0
    print(f"\nEnsemble logits: mean={ensemble_logits.mean():.4f}, std={ensemble_logits.std():.4f}")

    optimal_threshold = find_optimal_threshold(ensemble_logits, targets)
    metrics_05 = compute_binary_classification_metrics(ensemble_logits, targets, threshold=0.5)
    metrics_opt = compute_binary_classification_metrics(ensemble_logits, targets, threshold=optimal_threshold)

    result = {
        "split": args.split,
        "rows": len(shared_dataset),
        "config1": str(config1_path),
        "config2": str(config2_path),
        "optimal_threshold": float(optimal_threshold),
        "metrics_at_threshold_0.5": metrics_05.to_dict(),
        "metrics_at_optimal_threshold": metrics_opt.to_dict(),
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2))

    print(f"\n--- Ensemble Results ({args.split}) ---")
    print(f"  AUROC:             {metrics_05.auroc:.4f}")
    print(f"  Optimal threshold: {optimal_threshold}")
    print(f"  Sensitivity@opt:   {metrics_opt.sensitivity:.4f}")
    print(f"  Specificity@opt:   {metrics_opt.specificity:.4f}")
    print(f"  F1@opt:            {metrics_opt.f1:.4f}")
    print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    main()
