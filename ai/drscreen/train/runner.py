from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image as PILImage
from torch import nn
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR, LRScheduler, LinearLR, SequentialLR
from torch.utils.data import DataLoader

from drscreen.data.datasets import ManifestDataset
from drscreen.data.transforms import FundusPreprocess, build_eval_transform, build_train_transform
from drscreen.models.build import build_model, get_classifier_module, split_model_parameters
from drscreen.quality.quickqual import QuickQualAssessor
from drscreen.models.profiles import get_model_profile
from drscreen.settings import merge_dicts, resolve_project_path
from drscreen.train.engine import evaluate_one_epoch, train_one_epoch
from drscreen.utils.logging import get_logger
from drscreen.utils.seed import set_seed


LOGGER = get_logger(__name__)

_DEFAULT_MIN_SENSITIVITY = 0.80


@dataclass(frozen=True, slots=True)
class TrainingPhase:
    name: str
    epochs: int
    head_only: bool


def resolve_device(device_name: str) -> torch.device:
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if device_name.startswith("cuda") and not torch.cuda.is_available():
        LOGGER.warning("CUDA was requested but is not available. Falling back to CPU.")
        return torch.device("cpu")

    return torch.device(device_name)


def _validate_training_scope(config: dict[str, Any]) -> None:
    task = str(config["project"]["task"])
    num_outputs = int(config["model"]["num_outputs"])
    label_names = list(config["labels"]["names"])

    if task != "binary_dr_screening" or num_outputs != 1:
        raise NotImplementedError(
            "The training loop currently supports only the binary_dr_screening task with "
            "model.num_outputs == 1."
        )

    if len(label_names) != 2:
        raise ValueError("Binary training expects exactly two label names.")


def _build_transforms(config: dict[str, Any]) -> tuple[Any, Any]:
    architecture = str(config["model"]["architecture"])
    profile = get_model_profile(architecture)
    data_cfg = config["data"]
    image_size = int(data_cfg["image_size"])
    resize_size = int(data_cfg["resize_size"])

    use_preprocessing = bool(data_cfg.get("use_preprocessing", False))
    train_transform = build_train_transform(
        crop_size=image_size,
        resize_size=resize_size,
        interpolation=profile.interpolation,
        mean=profile.mean,
        std=profile.std,
        use_preprocessing=use_preprocessing,
    )
    eval_transform = build_eval_transform(
        crop_size=image_size,
        resize_size=resize_size,
        interpolation=profile.interpolation,
        mean=profile.mean,
        std=profile.std,
        use_preprocessing=use_preprocessing,
    )
    return train_transform, eval_transform


def _build_datasets(
    config: dict[str, Any],
    project_root: Path,
) -> tuple[ManifestDataset, ManifestDataset, Path]:
    data_cfg = config["data"]
    manifest_path = resolve_project_path(project_root, data_cfg["manifest_path"])
    image_root = resolve_project_path(project_root, data_cfg["image_root"])
    train_transform, eval_transform = _build_transforms(config)

    train_dataset = ManifestDataset(
        manifest_path=manifest_path,
        image_root=image_root,
        split=data_cfg["train_split"],
        transform=train_transform,
    )
    val_dataset = ManifestDataset(
        manifest_path=manifest_path,
        image_root=image_root,
        split=data_cfg["val_split"],
        transform=eval_transform,
    )
    if len(train_dataset) == 0:
        raise ValueError("Training split is empty.")
    if len(val_dataset) == 0:
        raise ValueError("Validation split is empty.")
    return train_dataset, val_dataset, manifest_path


def _build_eval_dataset(
    config: dict[str, Any],
    project_root: Path,
    split_name: str,
) -> tuple[ManifestDataset, Path]:
    data_cfg = config["data"]
    manifest_path = resolve_project_path(project_root, data_cfg["manifest_path"])
    image_root = resolve_project_path(project_root, data_cfg["image_root"])
    _, eval_transform = _build_transforms(config)

    dataset = ManifestDataset(
        manifest_path=manifest_path,
        image_root=image_root,
        split=split_name,
        transform=eval_transform,
    )
    if len(dataset) == 0:
        raise ValueError(f"Evaluation split is empty: {split_name}")
    return dataset, manifest_path


def _build_dataloaders(
    config: dict[str, Any],
    project_root: Path,
    device: torch.device,
) -> tuple[DataLoader, DataLoader, Path]:
    train_dataset, val_dataset, manifest_path = _build_datasets(config, project_root)
    data_cfg = config["data"]
    batch_size = int(data_cfg["batch_size"])
    num_workers = int(data_cfg.get("num_workers", 0))
    persistent_workers = bool(data_cfg.get("persistent_workers", num_workers > 0)) and num_workers > 0
    generator = torch.Generator().manual_seed(int(config["train"]["seed"]))

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=persistent_workers,
        generator=generator,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=persistent_workers,
    )
    return train_loader, val_loader, manifest_path


def _build_training_phases(config: dict[str, Any]) -> list[TrainingPhase]:
    train_cfg = config["train"]
    phases: list[TrainingPhase] = []

    head_epochs = int(train_cfg.get("head_epochs", 0))
    if head_epochs > 0:
        phases.append(TrainingPhase(name="head", epochs=head_epochs, head_only=True))

    finetune_epochs = int(train_cfg.get("finetune_epochs", 0))
    if finetune_epochs > 0:
        phases.append(TrainingPhase(name="finetune", epochs=finetune_epochs, head_only=False))

    if not phases:
        raise ValueError("At least one training epoch is required.")

    return phases


def _set_phase_trainability(model: nn.Module, architecture: str, *, head_only: bool) -> None:
    backbone_parameters, head_parameters = split_model_parameters(architecture, model)
    for parameter in backbone_parameters:
        parameter.requires_grad = not head_only
    for parameter in head_parameters:
        parameter.requires_grad = True


def _prepare_model_for_head_only_training(model: nn.Module, architecture: str) -> None:
    classifier = get_classifier_module(architecture, model)
    for module in model.children():
        module.eval()
    classifier.train()


def _build_optimizer(
    config: dict[str, Any],
    model: nn.Module,
    *,
    architecture: str,
    head_only: bool,
) -> Optimizer:
    train_cfg = config["train"]
    optimizer_name = str(train_cfg["optimizer"]).lower()
    if optimizer_name != "adamw":
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")

    backbone_parameters, head_parameters = split_model_parameters(architecture, model)
    weight_decay = float(train_cfg["weight_decay"])
    head_learning_rate = float(train_cfg["head_learning_rate"])
    backbone_learning_rate = float(train_cfg["backbone_learning_rate"])

    if head_only:
        parameter_groups = [
            {
                "params": [parameter for parameter in head_parameters if parameter.requires_grad],
                "lr": head_learning_rate,
            }
        ]
    else:
        parameter_groups = []
        if head_parameters:
            parameter_groups.append(
                {
                    "params": [parameter for parameter in head_parameters if parameter.requires_grad],
                    "lr": head_learning_rate,
                }
            )
        if backbone_parameters:
            parameter_groups.append(
                {
                    "params": [
                        parameter for parameter in backbone_parameters if parameter.requires_grad
                    ],
                    "lr": backbone_learning_rate,
                }
            )

    if not any(group["params"] for group in parameter_groups):
        raise ValueError("No trainable parameters were available for the requested phase.")

    return AdamW(parameter_groups, weight_decay=weight_decay)


def _build_scheduler(config: dict[str, Any], optimizer: Optimizer, epochs: int) -> LRScheduler | None:
    scheduler_name = str(config["train"]["scheduler"]).lower()
    if scheduler_name != "cosine":
        raise ValueError(f"Unsupported scheduler: {scheduler_name}")

    if epochs <= 1:
        return None

    warmup_epochs = int(config["train"].get("warmup_epochs", 0))
    warmup_epochs = min(warmup_epochs, max(epochs - 1, 0))
    if warmup_epochs <= 0:
        return CosineAnnealingLR(optimizer, T_max=epochs)

    warmup = LinearLR(optimizer, start_factor=0.2, end_factor=1.0, total_iters=warmup_epochs)
    cosine = CosineAnnealingLR(optimizer, T_max=max(epochs - warmup_epochs, 1))
    return SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[warmup_epochs])


def _build_criterion(config: dict[str, Any]) -> nn.Module:
    _validate_training_scope(config)
    return nn.BCEWithLogitsLoss()


def _checkpoint_payload(
    *,
    config: dict[str, Any],
    model: nn.Module,
    optimizer: Optimizer,
    scheduler: LRScheduler | None,
    epoch: int,
    phase: str,
    train_metrics: dict[str, Any],
    val_metrics: dict[str, Any],
) -> dict[str, Any]:
    return {
        "epoch": epoch,
        "phase": phase,
        "architecture": config["model"]["architecture"],
        "num_outputs": config["model"]["num_outputs"],
        "label_names": list(config["labels"]["names"]),
        "config": config,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
    }


def _prefixed_metric_fields(prefix: str, metrics: Any) -> dict[str, Any]:
    return {f"{prefix}_{key}": value for key, value in metrics.to_dict().items()}


def _build_effective_eval_config(
    runtime_config: dict[str, Any],
    checkpoint: dict[str, Any],
) -> dict[str, Any]:
    checkpoint_config = checkpoint.get("config")
    effective_config = runtime_config
    if isinstance(checkpoint_config, dict):
        effective_config = merge_dicts(checkpoint_config, runtime_config)

    effective_config = merge_dicts(
        effective_config,
        {
            "model": {
                "architecture": checkpoint.get(
                    "architecture",
                    effective_config["model"]["architecture"],
                ),
                "num_outputs": checkpoint.get(
                    "num_outputs",
                    effective_config["model"]["num_outputs"],
                ),
                "pretrained": False,
            },
            "labels": {
                "names": checkpoint.get(
                    "label_names",
                    effective_config["labels"]["names"],
                )
            },
        },
    )
    return effective_config


def describe_training_setup(
    config: dict[str, Any],
    *,
    config_path: Path,
    project_root: Path,
) -> dict[str, Any]:
    _validate_training_scope(config)
    train_dataset, val_dataset, manifest_path = _build_datasets(config, project_root)
    profile = get_model_profile(str(config["model"]["architecture"]))
    phases = _build_training_phases(config)

    return {
        "project_root": str(project_root),
        "config_path": str(config_path),
        "manifest_path": str(manifest_path),
        "manifest_exists": manifest_path.exists(),
        "train_rows": len(train_dataset),
        "val_rows": len(val_dataset),
        "architecture": profile.architecture,
        "recommended_profile": profile.to_dict(),
        "phases": [
            {"name": phase.name, "epochs": phase.epochs, "head_only": phase.head_only}
            for phase in phases
        ],
    }


def run_training(
    config: dict[str, Any],
    *,
    config_path: Path,
    project_root: Path,
) -> dict[str, Any]:
    _validate_training_scope(config)
    set_seed(int(config["train"]["seed"]))

    device = resolve_device(str(config["train"]["device"]))
    architecture = str(config["model"]["architecture"])
    train_loader, val_loader, manifest_path = _build_dataloaders(config, project_root, device)

    model = build_model(
        architecture,
        pretrained=bool(config["model"]["pretrained"]),
        num_outputs=int(config["model"]["num_outputs"]),
        use_attention=bool(config["model"].get("use_attention", False)),
        grad_checkpointing=bool(config["model"].get("grad_checkpointing", False)),
    ).to(device)
    criterion = _build_criterion(config)
    version = str(config["project"].get("version", "")).strip()
    checkpoint_dir = resolve_project_path(project_root, config["train"]["checkpoint_dir"])
    if version:
        checkpoint_dir = checkpoint_dir / version
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    amp_enabled = bool(config["train"].get("amp", False)) and device.type == "cuda"
    # BF16 has the same exponent range as FP32, so GradScaler is not needed.
    # FP16 still requires scaling to avoid underflow in gradients.
    _amp_needs_scaler = amp_enabled and not (
        torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    )
    scaler = torch.amp.GradScaler("cuda", enabled=_amp_needs_scaler)
    gradient_clip_norm = float(config["train"].get("gradient_clip_norm", 0.0)) or None

    best_val_auroc = 0.0
    best_epoch = 0
    best_checkpoint_path = checkpoint_dir / "best.pt"
    last_checkpoint_path = checkpoint_dir / "last.pt"
    history: list[dict[str, Any]] = []
    global_epoch = 0

    for phase in _build_training_phases(config):
        _set_phase_trainability(model, architecture, head_only=phase.head_only)
        optimizer = _build_optimizer(
            config,
            model,
            architecture=architecture,
            head_only=phase.head_only,
        )
        scheduler = _build_scheduler(config, optimizer, phase.epochs)

        for phase_epoch in range(1, phase.epochs + 1):
            global_epoch += 1
            train_metrics = train_one_epoch(
                model,
                train_loader,
                criterion,
                optimizer,
                device,
                model_train_setup=(
                    (lambda current_model: _prepare_model_for_head_only_training(current_model, architecture))
                    if phase.head_only
                    else None
                ),
                amp_enabled=amp_enabled,
                scaler=scaler,
                gradient_clip_norm=gradient_clip_norm,
            )
            val_metrics = evaluate_one_epoch(
                model,
                val_loader,
                criterion,
                device,
                amp_enabled=amp_enabled,
            )
            if scheduler is not None:
                scheduler.step()

            epoch_record = {
                "epoch": global_epoch,
                "phase": phase.name,
                "phase_epoch": phase_epoch,
                "learning_rates": [float(group["lr"]) for group in optimizer.param_groups],
            }
            epoch_record.update(_prefixed_metric_fields("train", train_metrics))
            epoch_record.update(_prefixed_metric_fields("val", val_metrics))
            history.append(epoch_record)

            checkpoint_payload = _checkpoint_payload(
                config=config,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=global_epoch,
                phase=phase.name,
                train_metrics=train_metrics.to_dict(),
                val_metrics=val_metrics.to_dict(),
            )
            torch.save(checkpoint_payload, last_checkpoint_path)
            val_sensitivity = val_metrics.sensitivity or 0.0
            val_auroc = val_metrics.auroc or 0.0
            min_sensitivity = float(config["train"].get("min_checkpoint_sensitivity", _DEFAULT_MIN_SENSITIVITY))
            if val_sensitivity >= min_sensitivity and val_auroc > best_val_auroc:
                best_val_auroc = val_auroc
                best_epoch = global_epoch
                torch.save(checkpoint_payload, best_checkpoint_path)

            LOGGER.info(
                "phase=%s epoch=%s/%s train_loss=%.4f val_loss=%.4f val_sensitivity=%.4f val_auroc=%.4f",
                phase.name,
                phase_epoch,
                phase.epochs,
                train_metrics.loss,
                val_metrics.loss,
                val_sensitivity,
                val_auroc,
            )

    summary = {
        "project_root": str(project_root),
        "config_path": str(config_path),
        "manifest_path": str(manifest_path),
        "train_rows": len(train_loader.dataset),
        "val_rows": len(val_loader.dataset),
        "device": str(device),
        "amp_enabled": amp_enabled,
        "best_epoch": best_epoch,
        "best_val_auroc": best_val_auroc,
        "best_checkpoint_path": str(best_checkpoint_path),
        "last_checkpoint_path": str(last_checkpoint_path),
        "history": history,
    }
    summary_path = checkpoint_dir / "training_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    summary["summary_path"] = str(summary_path)
    return summary


def run_split_evaluation(
    config: dict[str, Any],
    *,
    config_path: Path,
    project_root: Path,
    split_name: str | None = None,
    checkpoint_path: Path | None = None,
) -> dict[str, Any]:
    requested_split = split_name or str(config["data"]["test_split"])
    resolved_checkpoint_path = resolve_project_path(
        project_root,
        checkpoint_path or config["infer"]["checkpoint_path"],
    )
    checkpoint = torch.load(resolved_checkpoint_path, map_location="cpu")
    effective_config = _build_effective_eval_config(config, checkpoint)
    _validate_training_scope(effective_config)

    device_name = str(
        effective_config.get("infer", {}).get("device")
        or effective_config.get("train", {}).get("device", "cpu")
    )
    device = resolve_device(device_name)
    dataset, manifest_path = _build_eval_dataset(effective_config, project_root, requested_split)

    quality_assessor = QuickQualAssessor.from_config(effective_config, project_root, device)
    if quality_assessor is not None:
        use_preprocessing = bool(effective_config["data"].get("use_preprocessing", False))
        preprocessor = FundusPreprocess() if use_preprocessing else None
        rejected_indices: list[int] = []
        for i, row in dataset.frame.iterrows():
            image_path = dataset.image_root / str(row["image_path"])
            pil_img = PILImage.open(image_path).convert("RGB")
            if preprocessor is not None:
                pil_img = preprocessor(pil_img)
            quality_result = quality_assessor.assess(np.asarray(pil_img))
            if quality_result.is_reject:
                rejected_indices.append(i)
        if rejected_indices:
            LOGGER.info(
                "EyeQ pre-scan: %d/%d images rejected and excluded from evaluation.",
                len(rejected_indices),
                len(dataset),
            )
            dataset.frame = dataset.frame.drop(index=rejected_indices).reset_index(drop=True)

    loader = DataLoader(
        dataset,
        batch_size=int(effective_config["data"]["batch_size"]),
        shuffle=False,
        num_workers=int(effective_config["data"].get("num_workers", 0)),
        pin_memory=device.type == "cuda",
        persistent_workers=int(effective_config["data"].get("num_workers", 0)) > 0,
    )
    model = build_model(
        str(effective_config["model"]["architecture"]),
        pretrained=False,
        num_outputs=int(effective_config["model"]["num_outputs"]),
        use_attention=bool(effective_config["model"].get("use_attention", False)),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    criterion = _build_criterion(effective_config)
    amp_enabled = bool(effective_config["train"].get("amp", False)) and device.type == "cuda"
    metrics = evaluate_one_epoch(
        model,
        loader,
        criterion,
        device,
        amp_enabled=amp_enabled,
    )

    evaluation_dir = resolve_project_path(project_root, effective_config["project"]["output_root"])
    evaluation_dir = evaluation_dir / "evaluations"
    evaluation_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_parent = resolved_checkpoint_path.parent.name
    output_path = evaluation_dir / (
        f"{requested_split}_{checkpoint_parent}_{resolved_checkpoint_path.stem}_metrics.json"
    )

    summary = {
        "project_root": str(project_root),
        "config_path": str(config_path),
        "manifest_path": str(manifest_path),
        "split": requested_split,
        "rows": len(dataset),
        "device": str(device),
        "checkpoint_path": str(resolved_checkpoint_path),
        "label_names": list(effective_config["labels"]["names"]),
        "metrics": metrics.to_dict(),
    }
    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    summary["output_path"] = str(output_path)
    return summary
