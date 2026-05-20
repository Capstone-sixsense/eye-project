from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from drscreen.data.datasets import ManifestDataset, SegmentationManifestDataset
from drscreen.data.mask_providers import (
    CompositeMaskProvider,
    IDRiDPerLesionMaskProvider,
    MAPLESTrainMaskProvider,
    TJDRMaskProvider,
)
from drscreen.data.transforms import (
    build_segmentation_eval_transform,
    build_segmentation_train_transform,
)
from drscreen.models.profiles import get_model_profile
from drscreen.models.seg_evidence import LesionSegEvidence
from drscreen.settings import get_run_checkpoint_dir, resolve_project_path
from drscreen.train.loss import DiceBCELoss, FocalTverskyBCELoss
from drscreen.utils.seed import set_seed
from drscreen.xai.seg_metrics import dice_iou_from_logits


def _amp_dtype(device: torch.device) -> torch.dtype:
    if (
        device.type == "cuda"
        and torch.cuda.is_available()
        and torch.cuda.is_bf16_supported()
    ):
        return torch.bfloat16
    return torch.float16


def _build_seg_mask_provider(
    config: dict[str, Any],
    project_root: Path,
) -> CompositeMaskProvider:
    data_cfg = config["data"]
    channels = int(config["model"].get("out_channels", data_cfg.get("seg_mask_channels", 4)))
    if channels != 4:
        raise ValueError("segmentation evidence training currently expects 4 mask channels")

    seg_mask_dir = resolve_project_path(
        project_root,
        data_cfg.get(
            "seg_mask_dir",
            "data/raw/IDRiD/A. Segmentation/2. All Segmentation Groundtruths",
        ),
    )
    maples_ann_dir = resolve_project_path(
        project_root,
        data_cfg.get(
            "maples_annotations_dir",
            "data/raw/MAPLES-DR/AdditionalData/annotations",
        ),
    )
    tjdr_root = resolve_project_path(
        project_root,
        data_cfg.get("tjdr_root", "data/raw/TJDR"),
    )
    raw_root = resolve_project_path(project_root, data_cfg.get("image_root", "data/raw"))
    providers = [
        IDRiDPerLesionMaskProvider(seg_mask_dir, raw_root=raw_root),
        MAPLESTrainMaskProvider(maples_ann_dir, channels=4, raw_root=raw_root),
        TJDRMaskProvider(tjdr_root, channels=4, raw_root=raw_root),
    ]
    return CompositeMaskProvider(providers)


def _build_transforms(config: dict[str, Any]) -> tuple[Any, Any]:
    data_cfg = config["data"]
    encoder = str(config["model"].get("encoder", "resnet50"))
    profile = get_model_profile(encoder)
    image_size = int(data_cfg.get("image_size", profile.crop_size))
    resize_size = int(data_cfg.get("resize_size", image_size))
    train_transform = build_segmentation_train_transform(
        crop_size=image_size,
        resize_size=resize_size,
        interpolation=profile.interpolation,
        mean=profile.mean,
        std=profile.std,
        use_preprocessing=bool(data_cfg.get("use_preprocessing", False)),
        use_random_resized_crop=bool(data_cfg.get("use_random_resized_crop", True)),
    )
    eval_transform = build_segmentation_eval_transform(
        crop_size=image_size,
        resize_size=resize_size,
        interpolation=profile.interpolation,
        mean=profile.mean,
        std=profile.std,
        use_preprocessing=bool(data_cfg.get("use_preprocessing", False)),
    )
    return train_transform, eval_transform


def _make_dataset(
    config: dict[str, Any],
    project_root: Path,
    *,
    transform,
    mask_provider,
) -> ManifestDataset:
    data_cfg = config["data"]
    return SegmentationManifestDataset(
        manifest_path=resolve_project_path(project_root, data_cfg["manifest_path"]),
        image_root=resolve_project_path(project_root, data_cfg.get("image_root", "data/raw")),
        split=data_cfg.get("train_split", "train"),
        transform=transform,
        seg_mask_size=int(data_cfg.get("image_size", 512)),
        mask_provider=mask_provider,
    )


def _valid_mask_indices(dataset: ManifestDataset) -> list[int]:
    valid: list[int] = []
    for idx, row in dataset.frame.iterrows():
        domain = str(row["domain"]) if "domain" in dataset.frame.columns else ""
        _mask, is_valid = dataset._mask_provider.load(  # noqa: SLF001 - internal runner
            str(row["image_path"]),
            domain,
            dataset._seg_mask_size,  # noqa: SLF001 - internal runner
        )
        if is_valid:
            valid.append(int(idx))
    return valid


def _build_seg_datasets(
    config: dict[str, Any],
    project_root: Path,
) -> tuple[ManifestDataset, ManifestDataset, Path, dict[str, int]]:
    data_cfg = config["data"]
    manifest_path = resolve_project_path(project_root, data_cfg["manifest_path"])
    mask_provider = _build_seg_mask_provider(config, project_root)
    train_transform, eval_transform = _build_transforms(config)

    base_dataset = _make_dataset(
        config,
        project_root,
        transform=eval_transform,
        mask_provider=mask_provider,
    )
    valid_indices = _valid_mask_indices(base_dataset)
    if not valid_indices:
        raise ValueError("No mask-valid rows found for segmentation evidence training.")

    seed = int(config["train"].get("seed", 42))
    rng = np.random.default_rng(seed)
    order = np.array(valid_indices, dtype=np.int64)
    rng.shuffle(order)
    val_fraction = float(data_cfg.get("seg_val_fraction", 0.15))
    val_count = max(1, int(round(len(order) * val_fraction))) if len(order) > 1 else 1
    val_idx = sorted(int(i) for i in order[:val_count])
    train_idx = sorted(int(i) for i in order[val_count:])
    if not train_idx:
        train_idx = val_idx

    train_dataset = _make_dataset(
        config,
        project_root,
        transform=train_transform,
        mask_provider=mask_provider,
    )
    val_dataset = _make_dataset(
        config,
        project_root,
        transform=eval_transform,
        mask_provider=mask_provider,
    )
    train_dataset.frame = base_dataset.frame.iloc[train_idx].reset_index(drop=True)
    val_dataset.frame = base_dataset.frame.iloc[val_idx].reset_index(drop=True)

    domain_counts = {
        str(k): int(v)
        for k, v in base_dataset.frame.iloc[valid_indices]["domain"].value_counts().items()
    }
    return train_dataset, val_dataset, manifest_path, domain_counts


def _build_loaders(
    config: dict[str, Any],
    project_root: Path,
    device: torch.device,
) -> tuple[DataLoader, DataLoader, Path, dict[str, int]]:
    train_dataset, val_dataset, manifest_path, domain_counts = _build_seg_datasets(
        config, project_root
    )
    data_cfg = config["data"]
    batch_size = int(data_cfg.get("batch_size", 4))
    num_workers = int(data_cfg.get("num_workers", 0))
    persistent_workers = bool(data_cfg.get("persistent_workers", False)) and num_workers > 0
    generator = torch.Generator().manual_seed(int(config["train"].get("seed", 42)))
    drop_last = bool(data_cfg.get("drop_last", False))
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=persistent_workers,
        generator=generator,
        drop_last=drop_last,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=persistent_workers,
    )
    return train_loader, val_loader, manifest_path, domain_counts


def _build_criterion(config: dict[str, Any]) -> torch.nn.Module:
    train_cfg = config["train"]
    loss_type = str(train_cfg.get("seg_loss_type", "dice_bce")).strip().lower()
    if loss_type == "dice_bce":
        return DiceBCELoss(
            dice_weight=float(train_cfg.get("dice_weight", 0.5))
        )
    if loss_type in {"focal_tversky", "focal_tversky_bce"}:
        return FocalTverskyBCELoss(
            alpha=float(train_cfg.get("tversky_alpha", 0.7)),
            beta=float(train_cfg.get("tversky_beta", 0.3)),
            gamma=float(train_cfg.get("tversky_gamma", 4.0 / 3.0)),
            bce_weight=float(train_cfg.get("bce_weight", 0.2)),
        )
    raise ValueError(
        f"Unsupported segmentation loss {loss_type!r}; expected dice_bce or focal_tversky_bce."
    )


def describe_segmentation_setup(
    config: dict[str, Any],
    *,
    config_path: Path,
    project_root: Path,
) -> dict[str, Any]:
    device = torch.device(str(config["train"].get("device", "cuda")))
    train_loader, val_loader, manifest_path, domain_counts = _build_loaders(
        config, project_root, device
    )
    return {
        "project_root": str(project_root),
        "config_path": str(config_path),
        "manifest_path": str(manifest_path),
        "train_rows": len(train_loader.dataset),
        "val_rows": len(val_loader.dataset),
        "mask_valid_domain_counts": domain_counts,
        "encoder": str(config["model"].get("encoder", "resnet50")),
        "out_channels": int(config["model"].get("out_channels", 4)),
    }


def _run_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
    *,
    optimizer: torch.optim.Optimizer | None = None,
    amp_enabled: bool = False,
    scaler: torch.amp.GradScaler | None = None,
    threshold: float = 0.5,
) -> dict[str, float | int | None]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_examples = 0
    metric_records: list[dict[str, float | None]] = []

    for batch in loader:
        images = batch["image"].to(device)
        targets = batch["seg_mask"].to(device)
        valid = batch["seg_mask_valid"].to(device).bool()
        if not valid.any():
            continue
        images = images[valid]
        targets = targets[valid]
        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type=device.type, dtype=_amp_dtype(device), enabled=amp_enabled):
            logits = model(images)
            loss = criterion(logits, targets)

        if training:
            assert optimizer is not None
            if scaler is not None and scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

        batch_size = int(images.shape[0])
        total_loss += float(loss.detach().item()) * batch_size
        total_examples += batch_size
        metric_records.append(dice_iou_from_logits(logits.detach(), targets, threshold=threshold))

    def _mean(key: str) -> float | None:
        vals = [m[key] for m in metric_records if m.get(key) is not None]
        return float(np.mean(vals)) if vals else None

    return {
        "loss": total_loss / max(total_examples, 1),
        "mdice": _mean("mdice"),
        "miou": _mean("miou"),
        "union_dice": _mean("union_dice"),
        "union_iou": _mean("union_iou"),
        "num_examples": total_examples,
    }


def run_segmentation_training(
    config: dict[str, Any],
    *,
    config_path: Path,
    project_root: Path,
) -> dict[str, Any]:
    set_seed(int(config["train"].get("seed", 42)))
    device = torch.device(str(config["train"].get("device", "cuda")))
    if device.type == "cuda" and not torch.cuda.is_available():
        device = torch.device("cpu")

    train_loader, val_loader, manifest_path, domain_counts = _build_loaders(
        config, project_root, device
    )
    model = LesionSegEvidence(
        encoder=str(config["model"].get("encoder", "resnet50")),
        out_channels=int(config["model"].get("out_channels", 4)),
        pretrained=bool(config["model"].get("pretrained", True)),
        decoder_channels=tuple(config["model"].get("decoder_channels", [256, 128, 64, 32])),
    ).to(device)

    criterion = _build_criterion(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["train"].get("learning_rate", 1e-4)),
        weight_decay=float(config["train"].get("weight_decay", 1e-4)),
    )
    amp_enabled = bool(config["train"].get("amp", False)) and device.type == "cuda"
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=amp_enabled and not torch.cuda.is_bf16_supported(),
    )
    threshold = float(config.get("infer", {}).get("lesion_threshold", 0.5))

    version = str(config["project"].get("version", "")).strip()
    checkpoint_dir = get_run_checkpoint_dir(project_root, version)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_path = checkpoint_dir / "best.pt"
    last_path = checkpoint_dir / "last.pt"

    best_val_mdice = -1.0
    best_epoch = 0
    history: list[dict[str, Any]] = []
    epochs = int(config["train"].get("epochs", 40))
    es_patience = int(config["train"].get("early_stopping_patience", 0))
    es_min_delta = float(config["train"].get("early_stopping_min_delta", 0.0))
    es_best_score = -1.0
    es_no_improve = 0
    stopped_early = False
    early_stop_epoch: int | None = None
    for epoch in range(1, epochs + 1):
        train_metrics = _run_epoch(
            model,
            train_loader,
            criterion,
            device,
            optimizer=optimizer,
            amp_enabled=amp_enabled,
            scaler=scaler,
            threshold=threshold,
        )
        with torch.no_grad():
            val_metrics = _run_epoch(
                model,
                val_loader,
                criterion,
                device,
                optimizer=None,
                amp_enabled=amp_enabled,
                threshold=threshold,
            )

        record = {
            "epoch": epoch,
            **{f"train_{k}": v for k, v in train_metrics.items()},
            **{f"val_{k}": v for k, v in val_metrics.items()},
        }
        history.append(record)
        print(
            "epoch "
            f"{epoch}/{epochs} "
            f"train_loss={train_metrics.get('loss'):.4f} "
            f"val_loss={val_metrics.get('loss'):.4f} "
            f"val_mdice={val_metrics.get('mdice')}",
            flush=True,
        )

        payload = {
            "epoch": epoch,
            "architecture": "lesion_seg_evidence",
            "encoder": str(config["model"].get("encoder", "resnet50")),
            "out_channels": int(config["model"].get("out_channels", 4)),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": config,
            "metrics": {"train": train_metrics, "val": val_metrics},
        }
        torch.save(payload, last_path)
        val_mdice = val_metrics.get("mdice")
        score = float(val_mdice) if val_mdice is not None else -1.0
        if score > best_val_mdice:
            best_val_mdice = score
            best_epoch = epoch
            torch.save(payload, best_path)

        if es_patience > 0:
            if score > es_best_score + es_min_delta:
                es_best_score = score
                es_no_improve = 0
            else:
                es_no_improve += 1
                if es_no_improve >= es_patience:
                    stopped_early = True
                    early_stop_epoch = epoch
                    print(
                        "early stopping "
                        f"at epoch {epoch}: val_mdice did not improve by "
                        f"{es_min_delta:.4f} for {es_patience} epochs",
                        flush=True,
                    )
                    break

    summary = {
        "project_root": str(project_root),
        "config_path": str(config_path),
        "manifest_path": str(manifest_path),
        "train_rows": len(train_loader.dataset),
        "val_rows": len(val_loader.dataset),
        "mask_valid_domain_counts": domain_counts,
        "device": str(device),
        "amp_enabled": amp_enabled,
        "seg_loss_type": str(config["train"].get("seg_loss_type", "dice_bce")),
        "best_epoch": best_epoch,
        "best_val_mdice": best_val_mdice if best_val_mdice >= 0 else None,
        "stopped_early": stopped_early,
        "early_stop_epoch": early_stop_epoch,
        "early_stopping_patience": es_patience,
        "early_stopping_min_delta": es_min_delta,
        "best_checkpoint_path": str(best_path),
        "last_checkpoint_path": str(last_path),
        "history": history,
    }
    summary_path = checkpoint_dir / "training_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    summary["summary_path"] = str(summary_path)
    return summary
