from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from drscreen.models.profiles import get_model_profile
from drscreen.settings import get_run_checkpoint_dir, resolve_project_path
from drscreen.train.checkpointing import (
    apply_swad,
    check_promotion_candidate,
    checkpoint_payload,
    prefixed_metric_fields,
)
from drscreen.train.data_loader_factory import _build_datasets, build_dataloaders
from drscreen.train.engine import SWADBuffer, evaluate_one_epoch, train_one_epoch
from drscreen.train.model_setup import (
    _DEFAULT_MIN_SENSITIVITY,
    build_criterion,
    build_model_for_training,
    build_optimizer,
    build_scheduler,
    build_training_phases,
    load_pretrained_backbone,
    prepare_model_for_decoder_only_training,
    prepare_model_for_head_only_training,
    resolve_device,
    set_phase_trainability,
    validate_training_scope,
)
from drscreen.utils.logging import get_logger
from drscreen.utils.seed import set_seed

LOGGER = get_logger(__name__)


def describe_training_setup(
    config: dict[str, Any],
    *,
    config_path: Path,
    project_root: Path,
) -> dict[str, Any]:
    validate_training_scope(config)
    train_dataset, val_dataset, manifest_path = _build_datasets(config, project_root)
    profile = get_model_profile(str(config["model"]["architecture"]))
    phases = build_training_phases(config)
    training_mode = str(config["train"].get("training_mode", "standard")).lower()
    return {
        "project_root": str(project_root),
        "config_path": str(config_path),
        "manifest_path": str(manifest_path),
        "manifest_exists": manifest_path.exists(),
        "train_rows": len(train_dataset),
        "val_rows": len(val_dataset),
        "architecture": profile.architecture,
        "training_mode": training_mode,
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
    validate_training_scope(config)
    set_seed(int(config["train"]["seed"]))

    device = resolve_device(str(config["train"]["device"]))
    architecture = str(config["model"]["architecture"])
    train_loader, val_loader, manifest_path = build_dataloaders(config, project_root, device)

    model = build_model_for_training(config, device)
    load_pretrained_backbone(model, config, project_root)
    training_mode = str(config["train"].get("training_mode", "standard")).lower()
    if training_mode not in {"standard", "decoder_only"}:
        raise ValueError(f"Unsupported train.training_mode: {training_mode!r}")

    criterion = build_criterion(config).to(device)
    version = str(config["project"].get("version", "")).strip()
    checkpoint_dir = (
        get_run_checkpoint_dir(project_root, version)
        if version
        else resolve_project_path(project_root, config["train"]["checkpoint_dir"])
    )
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    use_coral = bool(config["train"].get("use_coral", False))
    lambda_coral = float(config["train"].get("lambda_coral", 1.0))
    lambda_aux_seg = float(config["train"].get("lambda_aux_seg", 0.0))
    lambda_cam_align = float(config["train"].get("lambda_cam_align", 0.0))
    lambda_patch_l1 = float(config["train"].get("lambda_patch_l1", 0.0))
    lambda_concept = float(config["train"].get("lambda_concept", 0.0))
    coral_block_cfg = config["train"].get("coral_block")
    coral_block = int(coral_block_cfg) if coral_block_cfg is not None else None
    seg_loss_type = str(config["train"].get("seg_loss_type", "bce"))
    coral_criterion: torch.nn.Module | None = None
    if use_coral:
        from drscreen.train.loss import CoralLoss
        coral_criterion = CoralLoss().to(device)
        LOGGER.info(
            "CORAL enabled: lambda=%.4f block=%s",
            lambda_coral,
            "final_pooled" if coral_block is None else f"block{coral_block}",
        )
    if lambda_cam_align > 0.0:
        LOGGER.info("CAM alignment enabled: lambda=%.4f", lambda_cam_align)

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
    optimizer: Optimizer | None = None
    scheduler: LRScheduler | None = None
    swad_last_n = int(config["train"].get("swad_last_n_epochs", 0))
    swad_buffer: SWADBuffer | None = SWADBuffer(swad_last_n) if swad_last_n > 0 else None

    es_patience = int(config["train"].get("early_stopping_patience", 0))
    es_min_delta = float(config["train"].get("early_stopping_min_delta", 0.0))
    es_best_auroc = 0.0
    es_no_improve = 0
    should_stop = False

    for phase in build_training_phases(config):
        if should_stop:
            break
        if training_mode == "decoder_only":
            if phase.head_only:
                raise ValueError("decoder_only training does not support head_only phases.")
            prepare_model_for_decoder_only_training(model)
            optimizer_head_only = False
            model_train_setup = prepare_model_for_decoder_only_training
        else:
            set_phase_trainability(model, architecture, head_only=phase.head_only)
            optimizer_head_only = phase.head_only
            model_train_setup = (
                (lambda m: prepare_model_for_head_only_training(m, architecture))
                if phase.head_only else None
            )
        optimizer = build_optimizer(
            config,
            model,
            architecture=architecture,
            head_only=optimizer_head_only,
        )
        scheduler = build_scheduler(config, optimizer, phase.epochs)

        for phase_epoch in range(1, phase.epochs + 1):
            global_epoch += 1
            train_metrics = train_one_epoch(
                model, train_loader, criterion, optimizer, device,
                model_train_setup=model_train_setup,
                amp_enabled=amp_enabled, scaler=scaler, gradient_clip_norm=gradient_clip_norm,
                coral_criterion=coral_criterion,
                lambda_coral=lambda_coral,
                lambda_aux_seg=lambda_aux_seg,
                seg_loss_type=seg_loss_type,
                lambda_cam_align=lambda_cam_align,
                coral_block=coral_block,
                lambda_patch_l1=lambda_patch_l1,
                lambda_concept=lambda_concept,
            )
            val_metrics = evaluate_one_epoch(model, val_loader, criterion, device, amp_enabled=amp_enabled)
            if scheduler is not None:
                scheduler.step()

            epoch_record = {
                "epoch": global_epoch, "phase": phase.name, "phase_epoch": phase_epoch,
                "learning_rates": [float(g["lr"]) for g in optimizer.param_groups],
            }
            epoch_record.update(prefixed_metric_fields("train", train_metrics))
            epoch_record.update(prefixed_metric_fields("val", val_metrics))
            history.append(epoch_record)

            ckpt = checkpoint_payload(
                config=config, model=model, optimizer=optimizer, scheduler=scheduler,
                epoch=global_epoch, phase=phase.name,
                train_metrics=train_metrics.to_dict(), val_metrics=val_metrics.to_dict(),
            )
            torch.save(ckpt, last_checkpoint_path)

            val_sensitivity = val_metrics.sensitivity or 0.0
            val_auroc = val_metrics.auroc or 0.0
            min_sensitivity = float(config["train"].get("min_checkpoint_sensitivity", _DEFAULT_MIN_SENSITIVITY))
            if val_sensitivity >= min_sensitivity and val_auroc > best_val_auroc:
                best_val_auroc = val_auroc
                best_epoch = global_epoch
                torch.save(ckpt, best_checkpoint_path)

            if swad_buffer is not None and not phase.head_only:
                swad_buffer.update(model)

            if es_patience > 0 and not phase.head_only:
                if val_auroc > es_best_auroc + es_min_delta:
                    es_best_auroc = val_auroc
                    es_no_improve = 0
                else:
                    es_no_improve += 1
                    if es_no_improve >= es_patience:
                        LOGGER.info(
                            "Early stopping at epoch %d — val AUROC did not improve by %.4f for %d epochs",
                            global_epoch, es_min_delta, es_patience,
                        )
                        should_stop = True
                        break

            LOGGER.info(
                "phase=%s epoch=%s/%s train_loss=%.4f val_loss=%.4f val_sensitivity=%.4f val_auroc=%.4f",
                phase.name, phase_epoch, phase.epochs,
                train_metrics.loss, val_metrics.loss, val_sensitivity, val_auroc,
            )

    if swad_buffer is not None and len(swad_buffer) > 0:
        best_val_auroc, best_epoch = apply_swad(
            swad_buffer=swad_buffer, model=model, config=config, project_root=project_root,
            device=device, val_loader=val_loader, criterion=criterion, amp_enabled=amp_enabled,
            best_val_auroc=best_val_auroc, best_epoch=best_epoch, global_epoch=global_epoch,
            best_checkpoint_path=best_checkpoint_path, optimizer=optimizer, scheduler=scheduler,
        )

    promotion_candidate, global_best_auroc_at_run = check_promotion_candidate(
        version=version, best_val_auroc=best_val_auroc, best_epoch=best_epoch,
        project_root=project_root, config=config, best_checkpoint_path=best_checkpoint_path,
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
        "promoted_to_global_best": False,
        "promotion_candidate": promotion_candidate,
        "global_best_auroc_at_run": global_best_auroc_at_run,
        "history": history,
    }
    summary_path = checkpoint_dir / "training_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    summary["summary_path"] = str(summary_path)
    return summary
