from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from drscreen.data.mask_providers import (
    CompositeMaskProvider,
    DDRSegMaskProvider,
    IDRiDMaskProvider,
    IDRiDPerLesionMaskProvider,
    LesionMaskProvider,
    MAPLESTrainMaskProvider,
    NullMaskProvider,
    TJDRMaskProvider,
)
from drscreen.data.datasets import (
    FDAManifestDataset,
    ManifestDataset,
    SegmentationFDAManifestDataset,
    SegmentationManifestDataset,
)
from drscreen.data.transforms import (
    build_eval_transform,
    build_segmentation_train_transform,
    build_train_transform,
)
from drscreen.models.profiles import get_model_profile
from drscreen.settings import resolve_project_path


def _uses_mask_supervision(config: dict[str, Any]) -> bool:
    train_cfg = config.get("train", {})
    return float(train_cfg.get("lambda_aux_seg", 0.0) or 0.0) > 0.0


def _build_transforms(config: dict[str, Any]) -> tuple[Any, Any]:
    architecture = str(config["model"]["architecture"])
    profile = get_model_profile(architecture)
    data_cfg = config["data"]
    image_size = int(data_cfg["image_size"])
    resize_size = int(data_cfg["resize_size"])
    use_preprocessing = bool(data_cfg.get("use_preprocessing", False))
    use_random_resized_crop = bool(data_cfg.get("use_random_resized_crop", True))
    if _uses_mask_supervision(config):
        train_transform = build_segmentation_train_transform(
            crop_size=image_size, resize_size=resize_size,
            interpolation=profile.interpolation,
            mean=profile.mean, std=profile.std, use_preprocessing=use_preprocessing,
            use_random_resized_crop=use_random_resized_crop,
        )
    else:
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


def _build_mask_provider(
    config: dict[str, Any],
    seg_mask_dir,
    maples_ann_dir=None,
    tjdr_root=None,
    ddr_seg_root=None,
    raw_root=None,
) -> LesionMaskProvider:
    data_cfg = config["data"]
    model_cfg = config.get("model", {})
    mask_mode = str(data_cfg.get("seg_mask_mode", "union")).strip().lower()
    seg_channels = int(
        model_cfg.get("aux_seg_channels", data_cfg.get("seg_mask_channels", 1))
    )

    if mask_mode in {"none", "null", "off"}:
        return NullMaskProvider(channels=seg_channels)
    if mask_mode in {"per_lesion", "per-lesion", "multi", "multichannel"}:
        return IDRiDPerLesionMaskProvider(seg_mask_dir, raw_root=raw_root)
    if mask_mode in {"union", "binary"}:
        return IDRiDMaskProvider(seg_mask_dir, raw_root=raw_root)
    if mask_mode in {
        "composite",
        "idrid_maples",
        "idrid+maples",
        "idrid_maples_tjdr",
        "idrid+maples+tjdr",
        "idrid_maples_tjdr_ddr",
        "idrid+maples+tjdr+ddr",
    }:
        if maples_ann_dir is None:
            raise ValueError(
                "seg_mask_mode='composite' requires a resolved MAPLES annotations directory"
            )
        if seg_channels == 4:
            idrid: LesionMaskProvider = IDRiDPerLesionMaskProvider(seg_mask_dir, raw_root=raw_root)
        else:
            idrid = IDRiDMaskProvider(seg_mask_dir, raw_root=raw_root)
        maples = MAPLESTrainMaskProvider(maples_ann_dir, channels=seg_channels, raw_root=raw_root)
        providers: list[LesionMaskProvider] = [idrid, maples]
        if tjdr_root is not None:
            providers.append(TJDRMaskProvider(tjdr_root, channels=seg_channels, raw_root=raw_root))
        if ddr_seg_root is not None:
            providers.append(DDRSegMaskProvider(ddr_seg_root, channels=seg_channels, raw_root=raw_root))
        return CompositeMaskProvider(providers)
    raise ValueError(f"Unsupported data.seg_mask_mode: {mask_mode!r}")


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
    maples_ann_dir_cfg = data_cfg.get("maples_annotations_dir")
    maples_ann_dir = (
        resolve_project_path(project_root, maples_ann_dir_cfg)
        if maples_ann_dir_cfg
        else project_root / "data" / "raw" / "MAPLES-DR" / "AdditionalData" / "annotations"
    )
    tjdr_root_cfg = data_cfg.get("tjdr_root")
    tjdr_root = (
        resolve_project_path(project_root, tjdr_root_cfg)
        if tjdr_root_cfg
        else project_root / "data" / "raw" / "TJDR"
    )
    ddr_seg_root_cfg = data_cfg.get("ddr_seg_root")
    ddr_seg_root = (
        resolve_project_path(project_root, ddr_seg_root_cfg)
        if ddr_seg_root_cfg
        else project_root / "data" / "raw" / "ddr" / "lesion_segmentation"
    )
    concept_label_path_cfg = data_cfg.get("concept_label_path")
    concept_label_path = (
        resolve_project_path(project_root, concept_label_path_cfg)
        if concept_label_path_cfg
        else None
    )
    seg_mask_size = int(data_cfg.get("image_size", 512))
    mask_provider = _build_mask_provider(
        config,
        seg_mask_dir,
        maples_ann_dir=maples_ann_dir,
        tjdr_root=tjdr_root,
        ddr_seg_root=ddr_seg_root,
        raw_root=image_root,
    )

    use_fda = bool(data_cfg.get("use_fda", False))
    use_sync_mask_transform = _uses_mask_supervision(config)
    if use_fda:
        fda_alpha = float(data_cfg.get("fda_alpha", 0.05))
        if use_sync_mask_transform:
            train_dataset: ManifestDataset = SegmentationFDAManifestDataset(
                manifest_path=manifest_path, image_root=image_root,
                split=data_cfg["train_split"], transform=train_transform,
                fda_alpha=fda_alpha,
                fda_probability=float(data_cfg.get("fda_probability", 1.0)),
                fda_target_domain=data_cfg.get("fda_target_domain"),
                fda_apply_to_target_domain=bool(data_cfg.get("fda_apply_to_target_domain", False)),
                seg_mask_size=seg_mask_size, mask_provider=mask_provider,
                concept_label_path=concept_label_path,
            )
        else:
            train_dataset = FDAManifestDataset(
                manifest_path=manifest_path, image_root=image_root,
                split=data_cfg["train_split"], transform=train_transform, fda_alpha=fda_alpha,
                seg_mask_size=seg_mask_size, mask_provider=mask_provider,
                concept_label_path=concept_label_path,
            )
    else:
        dataset_cls = SegmentationManifestDataset if use_sync_mask_transform else ManifestDataset
        train_dataset = dataset_cls(
            manifest_path=manifest_path, image_root=image_root,
            split=data_cfg["train_split"], transform=train_transform,
            seg_mask_size=seg_mask_size, mask_provider=mask_provider,
            concept_label_path=concept_label_path,
        )
    val_dataset = ManifestDataset(
        manifest_path=manifest_path, image_root=image_root,
        split=data_cfg["val_split"], transform=eval_transform,
        concept_label_path=concept_label_path,
    )

    if bool(data_cfg.get("train_mask_valid_only", False)):
        valid_positions: list[int] = []
        for iloc_pos, row in train_dataset.frame.iterrows():
            _mask, is_valid = train_dataset._mask_provider.load(  # noqa: SLF001 - internal factory filter
                str(row["image_path"]),
                str(row["domain"]) if "domain" in train_dataset.frame.columns else "",
                seg_mask_size,
            )
            if is_valid:
                valid_positions.append(iloc_pos)
        train_dataset.frame = train_dataset.frame.iloc[valid_positions].reset_index(drop=True)
        if hasattr(train_dataset, "rebuild_domain_indices"):
            train_dataset.rebuild_domain_indices()

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
        if hasattr(train_dataset, "rebuild_domain_indices"):
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
