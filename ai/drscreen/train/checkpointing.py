"""체크포인트 payload 구성 + SWAD 가중치 평균 + 전역 best 승격 후보 판정.

- checkpoint_payload: 모델/옵티마이저/스케줄러 상태 + config + 메트릭을 한 dict로 묶어
  저장(추론 시 settings.build_effective_checkpoint_config가 이 config/architecture를 읽음).
- apply_swad: 마지막 N개 에폭 스냅샷을 평균낸 뒤 BatchNorm 통계를 재보정하고, 민감도
  하한을 만족하며 best AUROC를 넘으면 best로 저장.
- check_promotion_candidate: 전역 best 체크포인트보다 좋으면 '승격 후보'로 로깅만 한다.
  실제 배포 승격(파일 복사)은 수동이다(docs/AI_HANDOFF.md 4절 정책).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader

from drscreen.settings import resolve_project_path
from drscreen.train.engine import SWADBuffer, evaluate_one_epoch
from drscreen.train.model_setup import _DEFAULT_MIN_SENSITIVITY
from drscreen.utils.checkpoint import read_checkpoint_auroc
from drscreen.utils.logging import get_logger

LOGGER = get_logger(__name__)


def checkpoint_payload(
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


def prefixed_metric_fields(prefix: str, metrics: Any) -> dict[str, Any]:
    return {f"{prefix}_{key}": value for key, value in metrics.to_dict().items()}


def apply_swad(
    *,
    swad_buffer: SWADBuffer,
    model: nn.Module,
    config: dict[str, Any],
    project_root: Path,
    device: torch.device,
    val_loader: DataLoader,
    criterion: nn.Module,
    amp_enabled: bool,
    best_val_auroc: float,
    best_epoch: int,
    global_epoch: int,
    best_checkpoint_path: Path,
    optimizer: Optimizer,
    scheduler: LRScheduler | None,
) -> tuple[float, int]:
    from drscreen.train.data_loader_factory import build_eval_dataset

    LOGGER.info("Applying SWAD: averaging last %d epoch snapshots.", len(swad_buffer))
    avg_state = swad_buffer.get_averaged_state_dict()
    assert avg_state is not None
    model.load_state_dict(avg_state)
    model.to(device)

    _bn_dataset, _ = build_eval_dataset(config, project_root, str(config["data"]["train_split"]))
    _bn_loader = DataLoader(
        _bn_dataset,
        batch_size=int(config["data"]["batch_size"]),
        shuffle=False,
        num_workers=int(config["data"].get("num_workers", 0)),
        pin_memory=device.type == "cuda",
    )
    # 가중치를 평균내면 BatchNorm running stats가 어긋나므로, BN만 train 모드로 두고
    # 학습 데이터를 한 번 흘려 running mean/var를 재보정한다(파라미터는 갱신 안 됨).
    model.eval()
    for module in model.modules():
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            module.train()
    with torch.no_grad():
        for batch in _bn_loader:
            model(batch["image"].to(device))

    swad_val = evaluate_one_epoch(model, val_loader, criterion, device, amp_enabled=amp_enabled)
    swad_sensitivity = swad_val.sensitivity or 0.0
    swad_auroc = swad_val.auroc or 0.0
    min_sensitivity = float(config["train"].get("min_checkpoint_sensitivity", _DEFAULT_MIN_SENSITIVITY))
    LOGGER.info(
        "SWAD val: sensitivity=%.4f auroc=%.4f (best_before_swad=%.4f)",
        swad_sensitivity, swad_auroc, best_val_auroc,
    )
    if swad_sensitivity >= min_sensitivity and swad_auroc > best_val_auroc:
        best_val_auroc = swad_auroc
        best_epoch = global_epoch
        swad_payload = checkpoint_payload(
            config=config, model=model, optimizer=optimizer, scheduler=scheduler,
            epoch=global_epoch, phase="swad", train_metrics={}, val_metrics=swad_val.to_dict(),
        )
        torch.save(swad_payload, best_checkpoint_path)
        LOGGER.info("SWAD model saved as best: val_auroc=%.4f", swad_auroc)

    return best_val_auroc, best_epoch


def check_promotion_candidate(
    *,
    version: str,
    best_val_auroc: float,
    best_epoch: int,
    project_root: Path,
    config: dict[str, Any],
    best_checkpoint_path: Path,
) -> tuple[bool, float]:
    if not version or best_epoch == 0:
        return False, 0.0
    global_best_value = config["train"].get("global_best_checkpoint_path")
    global_best_path = (
        resolve_project_path(project_root, global_best_value)
        if global_best_value
        else resolve_project_path(project_root, config["train"]["checkpoint_dir"]) / "best.pt"
    )
    global_best_auroc = read_checkpoint_auroc(global_best_path) if global_best_path.exists() else 0.0
    if best_val_auroc > global_best_auroc:
        LOGGER.info(
            "Promotion candidate: val_auroc=%.4f > current global best=%.4f. "
            "Manual promotion required — review results before running: cp %s %s",
            best_val_auroc, global_best_auroc, best_checkpoint_path, global_best_path,
        )
        return True, global_best_auroc
    LOGGER.info(
        "No promotion: val_auroc=%.4f <= current global best=%.4f",
        best_val_auroc, global_best_auroc,
    )
    return False, global_best_auroc
