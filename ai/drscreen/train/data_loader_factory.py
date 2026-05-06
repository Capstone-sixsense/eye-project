from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from drscreen.data.datasets import FDAManifestDataset, ManifestDataset
from drscreen.data.transforms import build_eval_transform, build_train_transform
from drscreen.models.profiles import get_model_profile
from drscreen.settings import resolve_project_path


def _build_transforms(config: dict[str, Any]) -> tuple[Any, Any]:
    architecture = str(config["model"]["architecture"])
    profile = get_model_profile(architecture)
    data_cfg = config["data"]
    image_size = int(data_cfg["image_size"])
    resize_size = int(data_cfg["resize_size"])
    use_preprocessing = bool(data_cfg.get("use_preprocessing", False))
    use_random_resized_crop = bool(data_cfg.get("use_random_resized_crop", True))
    train_transform = build_train_transform(
        crop_size=image_size, resize_size=resize_size, interpolation=profile.interpolation,
        mean=profile.mean, std=profile.std, use_preprocessing=use_preprocessing,
        use_random_resized_crop=use_random_resized_crop,
    )
    eval_transform = build_eval_transform(
        crop_size=image_size, resize_size=resize_size, interpolation=profile.interpolation,
        mean=profile.mean, std=profile.std, use_preprocessing=use_preprocessing,
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
    excluded_domains = {
        str(domain).strip()
        for domain in data_cfg.get("train_exclude_domains", [])
        if str(domain).strip()
    }

    seg_mask_dir_cfg = data_cfg.get("seg_mask_dir")
    seg_mask_dir = (
        resolve_project_path(project_root, seg_mask_dir_cfg)
        if seg_mask_dir_cfg
        else project_root / "data" / "raw" / "IDRiD" / "A. Segmentation" / "2. All Segmentation Groundtruths"
    )
    seg_mask_size = int(data_cfg.get("image_size", 512))

    use_fda = bool(data_cfg.get("use_fda", False))
    if use_fda:
        fda_alpha = float(data_cfg.get("fda_alpha", 0.05))
        train_dataset: ManifestDataset = FDAManifestDataset(
            manifest_path=manifest_path, image_root=image_root,
            split=data_cfg["train_split"], transform=train_transform, fda_alpha=fda_alpha,
            seg_mask_dir=seg_mask_dir, seg_mask_size=seg_mask_size,
        )
    else:
        train_dataset = ManifestDataset(
            manifest_path=manifest_path, image_root=image_root,
            split=data_cfg["train_split"], transform=train_transform,
            seg_mask_dir=seg_mask_dir, seg_mask_size=seg_mask_size,
        )
    val_dataset = ManifestDataset(
        manifest_path=manifest_path, image_root=image_root,
        split=data_cfg["val_split"], transform=eval_transform,
    )

    if excluded_domains:
        for split_name, dataset in (("train", train_dataset), ("val", val_dataset)):
            if "domain" not in dataset.frame.columns:
                raise ValueError(
                    f"Cannot apply data.train_exclude_domains on {split_name} split: "
                    "manifest has no 'domain' column."
                )
            dataset.frame = dataset.frame[
                ~dataset.frame["domain"].astype(str).isin(excluded_domains)
            ].reset_index(drop=True)
        if isinstance(train_dataset, FDAManifestDataset):
            train_dataset.rebuild_domain_indices()

    if len(train_dataset) == 0:
        raise ValueError("Training split is empty.")
    if len(val_dataset) == 0:
        raise ValueError("Validation split is empty.")
    return train_dataset, val_dataset, manifest_path


def build_eval_dataset(
    config: dict[str, Any],
    project_root: Path,
    split_name: str,
) -> tuple[ManifestDataset, Path]:
    data_cfg = config["data"]
    manifest_path = resolve_project_path(project_root, data_cfg["manifest_path"])
    image_root = resolve_project_path(project_root, data_cfg["image_root"])
    _, eval_transform = _build_transforms(config)
    dataset = ManifestDataset(
        manifest_path=manifest_path, image_root=image_root,
        split=split_name, transform=eval_transform,
    )
    if len(dataset) == 0:
        raise ValueError(f"Evaluation split is empty: {split_name}")
    return dataset, manifest_path


def build_dataloaders(
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
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=device.type == "cuda",
        persistent_workers=persistent_workers, generator=generator,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=device.type == "cuda",
        persistent_workers=persistent_workers,
    )
    return train_loader, val_loader, manifest_path
