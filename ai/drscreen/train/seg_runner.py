from __future__ import annotations

import json
import math
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.optim.swa_utils import SWALR, AveragedModel
from torch.utils.data import DataLoader, WeightedRandomSampler

from drscreen.data.datasets import (
    ManifestDataset,
    SegmentationFDAManifestDataset,
    SegmentationManifestDataset,
)
from drscreen.data.mask_providers import (
    CompositeMaskProvider,
    DDRSegMaskProvider,
    IDRiDPerLesionMaskProvider,
    MAPLESTrainMaskProvider,
    TJDRMaskProvider,
)
from drscreen.data.transforms import (
    build_segmentation_eval_transform,
    build_segmentation_train_transform,
    preprocess_kwargs_from_config,
)
from drscreen.models.profiles import get_model_profile
from drscreen.models.seg_evidence import LesionSegEvidence
from drscreen.settings import get_run_checkpoint_dir, resolve_project_path
from drscreen.train.loss import DiceBCELoss, FocalTverskyBCELoss
from drscreen.utils.checkpoint import load_state_from_checkpoint
from drscreen.utils.seed import configure_determinism, environment_snapshot, set_seed
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
    ddr_seg_root = resolve_project_path(
        project_root,
        data_cfg.get("ddr_seg_root", "data/raw/ddr/lesion_segmentation"),
    )
    raw_root = resolve_project_path(project_root, data_cfg.get("image_root", "data/raw"))
    providers = [
        IDRiDPerLesionMaskProvider(seg_mask_dir, raw_root=raw_root),
        MAPLESTrainMaskProvider(maples_ann_dir, channels=4, raw_root=raw_root),
        TJDRMaskProvider(tjdr_root, channels=4, raw_root=raw_root),
        DDRSegMaskProvider(ddr_seg_root, channels=4, raw_root=raw_root),
    ]
    return CompositeMaskProvider(providers)


def _build_transforms(config: dict[str, Any]) -> tuple[Any, Any]:
    data_cfg = config["data"]
    encoder = str(config["model"].get("encoder", "resnet50"))
    profile = get_model_profile(encoder)
    image_size = int(data_cfg.get("image_size", profile.crop_size))
    resize_size = int(data_cfg.get("resize_size", image_size))
    preprocess_kwargs = preprocess_kwargs_from_config(data_cfg)
    train_transform = build_segmentation_train_transform(
        crop_size=image_size,
        resize_size=resize_size,
        interpolation=profile.interpolation,
        mean=profile.mean,
        std=profile.std,
        use_preprocessing=bool(data_cfg.get("use_preprocessing", False)),
        use_random_resized_crop=bool(data_cfg.get("use_random_resized_crop", True)),
        preprocess_kwargs=preprocess_kwargs,
        gin_config=data_cfg.get("gin"),
    )
    eval_transform = build_segmentation_eval_transform(
        crop_size=image_size,
        resize_size=resize_size,
        interpolation=profile.interpolation,
        mean=profile.mean,
        std=profile.std,
        use_preprocessing=bool(data_cfg.get("use_preprocessing", False)),
        preprocess_kwargs=preprocess_kwargs,
    )
    return train_transform, eval_transform


def _make_dataset(
    config: dict[str, Any],
    project_root: Path,
    *,
    transform,
    mask_provider,
    use_fda: bool = False,
) -> ManifestDataset:
    data_cfg = config["data"]
    dataset_cls = SegmentationFDAManifestDataset if use_fda else SegmentationManifestDataset
    kwargs: dict[str, Any] = {}
    if use_fda:
        ampmix_cfg = data_cfg.get("ampmix") or {}
        kwargs.update(
            {
                "fda_alpha": float(data_cfg.get("fda_alpha", 0.05)),
                "fda_probability": float(data_cfg.get("fda_probability", 1.0)),
                "ampmix_mode": bool(ampmix_cfg.get("enable", False)),
                "ampmix_alpha_low": float(ampmix_cfg.get("alpha_low", 0.0)),
                "ampmix_alpha_high": float(ampmix_cfg.get("alpha_high", 0.5)),
                "fda_target_domain": data_cfg.get("fda_target_domain"),
                "fda_apply_to_target_domain": bool(
                    data_cfg.get("fda_apply_to_target_domain", False)
                ),
            }
        )
    return dataset_cls(
        manifest_path=resolve_project_path(project_root, data_cfg["manifest_path"]),
        image_root=resolve_project_path(project_root, data_cfg.get("image_root", "data/raw")),
        split=data_cfg.get("train_split", "train"),
        transform=transform,
        seg_mask_size=int(data_cfg.get("image_size", 512)),
        mask_provider=mask_provider,
        **kwargs,
    )


def _valid_mask_indices(dataset: ManifestDataset) -> list[int]:
    valid: list[int] = []
    has_valid_mask = getattr(dataset._mask_provider, "has_valid_mask", None)  # noqa: SLF001
    for idx, row in dataset.frame.iterrows():
        domain = str(row["domain"]) if "domain" in dataset.frame.columns else ""
        if callable(has_valid_mask):
            is_valid = bool(has_valid_mask(str(row["image_path"]), domain))
        else:
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
        use_fda=bool(data_cfg.get("use_fda", False)),
    )
    val_dataset = _make_dataset(
        config,
        project_root,
        transform=eval_transform,
        mask_provider=mask_provider,
    )
    train_dataset.frame = base_dataset.frame.iloc[train_idx].reset_index(drop=True)
    val_dataset.frame = base_dataset.frame.iloc[val_idx].reset_index(drop=True)
    if isinstance(train_dataset, SegmentationFDAManifestDataset):
        train_dataset.rebuild_domain_indices()

    domain_counts = {
        str(k): int(v)
        for k, v in base_dataset.frame.iloc[valid_indices]["domain"].value_counts().items()
    }
    return train_dataset, val_dataset, manifest_path, domain_counts


def _build_domain_sampler(
    dataset: ManifestDataset,
    config: dict[str, Any],
    *,
    generator: torch.Generator,
) -> WeightedRandomSampler | None:
    data_cfg = config["data"]
    domain_sampling = str(data_cfg.get("domain_sampling", "")).strip().lower()
    domain_weights_cfg = data_cfg.get("domain_sample_weights") or {}
    if not domain_sampling and not domain_weights_cfg:
        return None

    if "domain" not in dataset.frame.columns:
        raise ValueError("domain_sampling requires a domain column in the training manifest.")

    domains = [str(v) for v in dataset.frame["domain"].tolist()]
    counts = {domain: domains.count(domain) for domain in sorted(set(domains))}
    weights: list[float] = []
    if domain_sampling == "balanced":
        weights = [1.0 / max(counts[domain], 1) for domain in domains]
    else:
        weights = [1.0 for _ in domains]

    if domain_weights_cfg:
        domain_weights = {
            str(domain): float(weight) for domain, weight in domain_weights_cfg.items()
        }
        weights = [
            weight * domain_weights.get(domain, 1.0)
            for weight, domain in zip(weights, domains, strict=True)
        ]

    if not any(weight > 0 for weight in weights):
        raise ValueError("domain_sample_weights produced all-zero training weights.")

    return WeightedRandomSampler(
        weights=torch.as_tensor(weights, dtype=torch.double),
        num_samples=len(weights),
        replacement=True,
        generator=generator,
    )


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
    sampler = _build_domain_sampler(train_dataset, config, generator=generator)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=sampler is None,
        sampler=sampler,
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


def _adverin_enabled(config: dict[str, Any]) -> bool:
    cfg = config.get("train", {}).get("adverin", {})
    return bool(isinstance(cfg, dict) and cfg.get("enable", False))


def _apply_adverin_mapping(
    images: torch.Tensor,
    *,
    mean: torch.Tensor,
    std: torch.Tensor,
    scale_param: torch.Tensor,
    bias_param: torch.Tensor,
    gamma_param: torch.Tensor,
    max_scale_delta: float,
    max_bias_delta: float,
    max_gamma_log_delta: float,
) -> torch.Tensor:
    x = (images * std + mean).clamp(0.0, 1.0)
    scale = 1.0 + float(max_scale_delta) * torch.tanh(scale_param)
    bias = float(max_bias_delta) * torch.tanh(bias_param)
    gamma = torch.exp(float(max_gamma_log_delta) * torch.tanh(gamma_param))
    x = (x * scale + bias).clamp(1.0e-4, 1.0)
    x = torch.pow(x, gamma).clamp(0.0, 1.0)
    return (x - mean) / std


def _adverin_batch(
    model: torch.nn.Module,
    images: torch.Tensor,
    targets: torch.Tensor,
    criterion: torch.nn.Module,
    *,
    mean: torch.Tensor,
    std: torch.Tensor,
    amp_enabled: bool,
    config: dict[str, Any],
) -> torch.Tensor:
    steps = int(config.get("inner_steps", 1))
    if steps <= 0:
        return images
    inner_lr = float(config.get("inner_lr", 0.1))
    max_scale_delta = float(config.get("max_scale_delta", 0.3))
    max_bias_delta = float(config.get("max_bias_delta", 0.12))
    max_gamma_log_delta = float(config.get("max_gamma_log_delta", 0.5))
    grad_mode = str(config.get("grad_mode", "sign")).strip().lower()

    shape = (images.shape[0], images.shape[1], 1, 1)
    scale_param = torch.zeros(shape, device=images.device, requires_grad=True)
    bias_param = torch.zeros(shape, device=images.device, requires_grad=True)
    gamma_param = torch.zeros(shape, device=images.device, requires_grad=True)
    params = (scale_param, bias_param, gamma_param)

    for _ in range(steps):
        adv_images = _apply_adverin_mapping(
            images,
            mean=mean,
            std=std,
            scale_param=scale_param,
            bias_param=bias_param,
            gamma_param=gamma_param,
            max_scale_delta=max_scale_delta,
            max_bias_delta=max_bias_delta,
            max_gamma_log_delta=max_gamma_log_delta,
        )
        with torch.autocast(device_type=images.device.type, dtype=_amp_dtype(images.device), enabled=amp_enabled):
            adv_loss = criterion(model(adv_images), targets)
        grads = torch.autograd.grad(adv_loss, params, only_inputs=True)
        with torch.no_grad():
            for param, grad in zip(params, grads, strict=True):
                update = grad.sign() if grad_mode == "sign" else grad
                param.add_(inner_lr * update)
                param.clamp_(-1.0, 1.0)
        for param in params:
            param.requires_grad_(True)

    return _apply_adverin_mapping(
        images,
        mean=mean,
        std=std,
        scale_param=scale_param.detach(),
        bias_param=bias_param.detach(),
        gamma_param=gamma_param.detach(),
        max_scale_delta=max_scale_delta,
        max_bias_delta=max_bias_delta,
        max_gamma_log_delta=max_gamma_log_delta,
    ).detach()


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
        "reproducibility_requested": {
            "seed": int(config["train"].get("seed", 42)),
            "deterministic": bool(config["train"].get("deterministic", False)),
            "deterministic_warn_only": bool(config["train"].get("deterministic_warn_only", True)),
            "cudnn_benchmark": bool(config["train"].get("cudnn_benchmark", False)),
        },
        "environment": environment_snapshot(project_root),
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
    adverin_config: dict[str, Any] | None = None,
    norm_mean: torch.Tensor | None = None,
    norm_std: torch.Tensor | None = None,
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
            if adverin_config and norm_mean is not None and norm_std is not None:
                images = _adverin_batch(
                    model,
                    images,
                    targets,
                    criterion,
                    mean=norm_mean,
                    std=norm_std,
                    amp_enabled=amp_enabled,
                    config=adverin_config,
                )

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


def _reset_and_update_bn_for_dict_loader(
    loader: DataLoader,
    model: torch.nn.Module,
    device: torch.device,
) -> None:
    bn_modules = [
        module for module in model.modules()
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm)
    ]
    if not bn_modules:
        return

    momenta = {module: module.momentum for module in bn_modules}
    for module in bn_modules:
        module.running_mean.zero_()
        module.running_var.fill_(1)
        module.num_batches_tracked.zero_()

    was_training = model.training
    model.train()
    examples_seen = 0
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            valid = batch["seg_mask_valid"].to(device).bool()
            if not valid.any():
                continue
            images = images[valid]
            batch_size = int(images.shape[0])
            momentum = batch_size / float(examples_seen + batch_size)
            for module in bn_modules:
                module.momentum = momentum
            model(images)
            examples_seen += batch_size

    for module, momentum in momenta.items():
        module.momentum = momentum
    model.train(was_training)


def _segmentation_checkpoint_payload(
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    config: dict[str, Any],
    epoch: int,
    train_metrics: dict[str, float | int | None],
    val_metrics: dict[str, float | int | None],
    initial_checkpoint_path: Path | None,
    swa: bool = False,
) -> dict[str, Any]:
    state_source = model.module if isinstance(model, AveragedModel) else model
    return {
        "epoch": epoch,
        "architecture": "lesion_seg_evidence",
        "encoder": str(config["model"].get("encoder", "resnet50")),
        "out_channels": int(config["model"].get("out_channels", 4)),
        "model_state_dict": state_source.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": config,
        "metrics": {"train": train_metrics, "val": val_metrics},
        "initial_checkpoint_path": str(initial_checkpoint_path) if initial_checkpoint_path else None,
        "swa": swa,
    }


def run_segmentation_training(
    config: dict[str, Any],
    *,
    config_path: Path,
    project_root: Path,
) -> dict[str, Any]:
    train_cfg = config["train"]
    determinism = configure_determinism(train_cfg)
    set_seed(int(train_cfg.get("seed", 42)))
    environment = environment_snapshot(project_root)
    device = torch.device(str(train_cfg.get("device", "cuda")))
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
    initial_checkpoint_path: Path | None = None
    initial_missing: list[str] = []
    initial_unexpected: list[str] = []
    initial_path_value = str(config["train"].get("initial_checkpoint_path", "")).strip()
    if initial_path_value:
        initial_checkpoint_path = resolve_project_path(project_root, initial_path_value)
        if not initial_checkpoint_path.exists():
            raise FileNotFoundError(f"initial_checkpoint_path not found: {initial_checkpoint_path}")
        checkpoint = torch.load(initial_checkpoint_path, map_location="cpu", weights_only=False)
        initial_missing, initial_unexpected = load_state_from_checkpoint(
            model,
            checkpoint,
            strict=bool(config["train"].get("initial_checkpoint_strict", True)),
        )
        print(
            "loaded initial segmentation checkpoint "
            f"{initial_checkpoint_path} "
            f"(missing={len(initial_missing)} unexpected={len(initial_unexpected)})",
            flush=True,
        )

    criterion = _build_criterion(config).to(device)
    profile = get_model_profile(str(config["model"].get("encoder", "resnet50")))
    norm_mean = torch.tensor(profile.mean, dtype=torch.float32, device=device).view(1, 3, 1, 1)
    norm_std = torch.tensor(profile.std, dtype=torch.float32, device=device).view(1, 3, 1, 1)
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
    adverin_config = config["train"].get("adverin", {}) if _adverin_enabled(config) else None

    version = str(config["project"].get("version", "")).strip()
    checkpoint_dir = get_run_checkpoint_dir(project_root, version)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_path = checkpoint_dir / "best.pt"
    last_path = checkpoint_dir / "last.pt"
    pre_swa_best_path = checkpoint_dir / "best_pre_swa.pt"
    swa_path = checkpoint_dir / "swa.pt"

    best_val_mdice = -1.0
    best_epoch = 0
    history: list[dict[str, Any]] = []
    epochs = int(config["train"].get("epochs", 40))
    swa_enable = bool(config["train"].get("swa_enable", False))
    swa_start_fraction = float(config["train"].get("swa_start_fraction", 0.8))
    swa_start_epoch = max(1, min(epochs, int(math.ceil(epochs * swa_start_fraction))))
    swa_lr = float(config["train"].get("swa_lr", config["train"].get("learning_rate", 1e-4)))
    swa_model = AveragedModel(model) if swa_enable else None
    swa_scheduler = SWALR(optimizer, swa_lr=swa_lr) if swa_enable else None
    swa_updates = 0
    swa_val_metrics: dict[str, float | int | None] | None = None
    swa_selected_as_best = False
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
            adverin_config=adverin_config,
            norm_mean=norm_mean,
            norm_std=norm_std,
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

        payload = _segmentation_checkpoint_payload(
            model=model,
            optimizer=optimizer,
            config=config,
            epoch=epoch,
            train_metrics=train_metrics,
            val_metrics=val_metrics,
            initial_checkpoint_path=initial_checkpoint_path,
        )
        torch.save(payload, last_path)
        val_mdice = val_metrics.get("mdice")
        score = float(val_mdice) if val_mdice is not None else -1.0
        if score > best_val_mdice:
            best_val_mdice = score
            best_epoch = epoch
            torch.save(payload, best_path)

        if swa_model is not None and epoch >= swa_start_epoch:
            swa_model.update_parameters(model)
            swa_updates += 1
            if swa_scheduler is not None:
                swa_scheduler.step()

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

    if swa_model is not None and swa_updates > 0:
        if best_path.exists():
            shutil.copy2(best_path, pre_swa_best_path)
        _reset_and_update_bn_for_dict_loader(train_loader, swa_model, device)
        with torch.no_grad():
            swa_val_metrics = _run_epoch(
                swa_model,
                val_loader,
                criterion,
                device,
                optimizer=None,
                amp_enabled=amp_enabled,
                threshold=threshold,
            )
        swa_payload = _segmentation_checkpoint_payload(
            model=swa_model,
            optimizer=optimizer,
            config=config,
            epoch=early_stop_epoch or epoch,
            train_metrics={},
            val_metrics=swa_val_metrics,
            initial_checkpoint_path=initial_checkpoint_path,
            swa=True,
        )
        torch.save(swa_payload, swa_path)
        swa_score_value = swa_val_metrics.get("mdice")
        swa_score = float(swa_score_value) if swa_score_value is not None else -1.0
        if swa_score > best_val_mdice:
            best_val_mdice = swa_score
            best_epoch = int(swa_payload["epoch"])
            swa_selected_as_best = True
            torch.save(swa_payload, best_path)

    summary = {
        "project_root": str(project_root),
        "config_path": str(config_path),
        "manifest_path": str(manifest_path),
        "train_rows": len(train_loader.dataset),
        "val_rows": len(val_loader.dataset),
        "mask_valid_domain_counts": domain_counts,
        "device": str(device),
        "amp_enabled": amp_enabled,
        "reproducibility": {
            "seed": int(train_cfg.get("seed", 42)),
            "determinism": determinism,
            "environment": environment,
        },
        "initial_checkpoint_path": str(initial_checkpoint_path) if initial_checkpoint_path else None,
        "initial_checkpoint_missing": initial_missing,
        "initial_checkpoint_unexpected": initial_unexpected,
        "seg_loss_type": str(config["train"].get("seg_loss_type", "dice_bce")),
        "adverin": adverin_config,
        "best_epoch": best_epoch,
        "best_val_mdice": best_val_mdice if best_val_mdice >= 0 else None,
        "stopped_early": stopped_early,
        "early_stop_epoch": early_stop_epoch,
        "early_stopping_patience": es_patience,
        "early_stopping_min_delta": es_min_delta,
        "swa_enable": swa_enable,
        "swa_start_epoch": swa_start_epoch if swa_enable else None,
        "swa_updates": swa_updates,
        "swa_checkpoint_path": str(swa_path) if swa_updates > 0 else None,
        "pre_swa_best_checkpoint_path": str(pre_swa_best_path) if pre_swa_best_path.exists() else None,
        "swa_val_metrics": swa_val_metrics,
        "swa_selected_as_best": swa_selected_as_best,
        "best_checkpoint_path": str(best_path),
        "last_checkpoint_path": str(last_path),
        "history": history,
    }
    summary_path = checkpoint_dir / "training_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    summary["summary_path"] = str(summary_path)
    return summary
