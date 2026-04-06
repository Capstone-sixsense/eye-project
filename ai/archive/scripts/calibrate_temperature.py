"""Temperature scaling calibration for drscreen.

val 분할로 temperature T를 보정(NLL 최소화)한 뒤
external_test 지표 변화를 비교 출력한다.

Usage:
    python calibrate_temperature.py --config configs/base.yaml
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from drscreen.data.datasets import ManifestDataset
from drscreen.models.build import build_model
from drscreen.settings import load_app_config, resolve_project_path
from drscreen.train.engine import collect_logits_and_targets
from drscreen.train.metrics import compute_binary_classification_metrics, find_optimal_threshold
from drscreen.settings import build_effective_checkpoint_config
from drscreen.train.runner import _build_transforms, resolve_device


def _nll(logits: torch.Tensor, targets: torch.Tensor, temperature: float) -> float:
    return F.binary_cross_entropy_with_logits(
        logits / temperature, targets.float()
    ).item()


def find_temperature(logits: torch.Tensor, targets: torch.Tensor) -> float:
    """Grid search T in [0.1, 5.0] minimising NLL on given split."""
    best_t, best_nll = 1.0, float("inf")
    for t in np.linspace(0.1, 5.0, 490):
        nll = _nll(logits, targets, float(t))
        if nll < best_nll:
            best_nll, best_t = nll, float(t)
    return round(best_t, 3)


def _build_loader(
    config: dict,
    project_root: Path,
    split: str,
    eval_transform,
    device: torch.device,
) -> DataLoader:
    data_cfg = config["data"]
    manifest_path = resolve_project_path(project_root, data_cfg["manifest_path"])
    image_root = resolve_project_path(project_root, data_cfg["image_root"])
    dataset = ManifestDataset(
        manifest_path=manifest_path,
        image_root=image_root,
        split=split,
        transform=eval_transform,
    )
    if len(dataset) == 0:
        raise ValueError(f"Split is empty: {split}")
    return DataLoader(
        dataset,
        batch_size=int(data_cfg["batch_size"]),
        shuffle=False,
        num_workers=int(data_cfg.get("num_workers", 0)),
        pin_memory=device.type == "cuda",
        persistent_workers=int(data_cfg.get("num_workers", 0)) > 0,
    )


def _row(threshold: float, version: str, m) -> str:
    sens = m.sensitivity or 0.0
    spec = m.specificity or 0.0
    return f"  {threshold:<10.2f}  {version:<14}  {sens:.4f}         {spec:.4f}         {m.f1:.4f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--val-split", default=None)
    parser.add_argument("--eval-split", default=None)
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    project_root = config_path.parents[1]
    base_path = config_path.parent / "base.yaml"
    config = load_app_config(
        config_path,
        base_path=base_path if config_path.name != "base.yaml" and base_path.exists() else None,
    )

    checkpoint_path = resolve_project_path(
        project_root,
        args.checkpoint or config["infer"]["checkpoint_path"],
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    effective_config = build_effective_checkpoint_config(config, checkpoint)

    device_name = str(
        effective_config.get("infer", {}).get("device")
        or effective_config.get("train", {}).get("device", "cpu")
    )
    device = resolve_device(device_name)
    amp = bool(effective_config["train"].get("amp", False)) and device.type == "cuda"
    _, eval_transform = _build_transforms(effective_config)

    model = build_model(
        str(effective_config["model"]["architecture"]),
        pretrained=False,
        num_outputs=int(effective_config["model"]["num_outputs"]),
        use_attention=bool(effective_config["model"].get("use_attention", False)),
        classifier_dropout=0.0,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    val_split = args.val_split or str(effective_config["data"]["val_split"])
    eval_split = args.eval_split or str(effective_config["data"]["external_test_split"])

    # --- 1. Fit temperature on val -----------------------------------------
    print(f"[1/3] Collecting val logits ({val_split})...")
    val_loader = _build_loader(effective_config, project_root, val_split, eval_transform, device)
    val_logits, val_targets = collect_logits_and_targets(model, val_loader, device, amp_enabled=amp)

    print("[2/3] Fitting temperature...")
    T = find_temperature(val_logits, val_targets)
    nll_before = _nll(val_logits, val_targets, 1.0)
    nll_after = _nll(val_logits, val_targets, T)
    print(f"      T = {T}  |  val NLL: {nll_before:.4f} -> {nll_after:.4f}")

    # --- 2. Evaluate on external_test --------------------------------------
    print(f"[3/3] Collecting {eval_split} logits...")
    eval_loader = _build_loader(effective_config, project_root, eval_split, eval_transform, device)
    eval_logits, eval_targets = collect_logits_and_targets(
        model, eval_loader, device, amp_enabled=amp
    )
    cal_logits = eval_logits / T

    # --- 3. Report ---------------------------------------------------------
    print()
    print("=" * 70)
    print(f"  Temperature T = {T}   (val NLL: {nll_before:.4f} -> {nll_after:.4f})")
    print("=" * 70)
    print(f"  {'Threshold':<10}  {'Version':<14}  {'Sensitivity':<14}  {'Specificity':<14}  F1")
    print("  " + "-" * 66)

    results: dict = {
        "temperature": T,
        "val_nll_before": nll_before,
        "val_nll_after": nll_after,
        "metrics_by_threshold": {},
    }

    for t in [0.5, 0.3, 0.15, 0.11]:
        raw = compute_binary_classification_metrics(eval_logits, eval_targets, threshold=t)
        cal = compute_binary_classification_metrics(cal_logits, eval_targets, threshold=t)
        print(_row(t, "raw", raw))
        print(_row(t, "calibrated", cal))
        print()
        results["metrics_by_threshold"][str(t)] = {
            "raw": raw.to_dict(),
            "calibrated": cal.to_dict(),
        }

    opt_t_raw = find_optimal_threshold(eval_logits, eval_targets)
    opt_t_cal = find_optimal_threshold(cal_logits, eval_targets)
    opt_raw = compute_binary_classification_metrics(eval_logits, eval_targets, threshold=opt_t_raw)
    opt_cal = compute_binary_classification_metrics(cal_logits, eval_targets, threshold=opt_t_cal)

    print("  Youden's J optimal:")
    print(_row(opt_t_raw, f"raw (t={opt_t_raw})", opt_raw))
    print(_row(opt_t_cal, f"cal (t={opt_t_cal})", opt_cal))

    results["optimal_raw"] = {"threshold": opt_t_raw, "metrics": opt_raw.to_dict()}
    results["optimal_calibrated"] = {"threshold": opt_t_cal, "metrics": opt_cal.to_dict()}

    out = project_root / "artifacts" / "evaluations" / "temperature_calibration.json"
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\n  Saved -> {out}")


if __name__ == "__main__":
    main()
