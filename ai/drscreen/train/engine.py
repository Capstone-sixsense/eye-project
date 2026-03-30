from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Callable

import torch

from drscreen.train.metrics import compute_binary_classification_metrics


def _amp_dtype(device: torch.device) -> torch.dtype:
    """Return BF16 on Ampere/Blackwell (SM >= 8.0) where BF16 is hardware-supported
    and avoids FP16 overflow. Fall back to FP16 for older GPUs."""
    if (
        device.type == "cuda"
        and torch.cuda.is_available()
        and torch.cuda.is_bf16_supported()
    ):
        return torch.bfloat16
    return torch.float16


@dataclass(slots=True)
class EpochMetrics:
    loss: float
    accuracy: float
    auroc: float | None
    f1: float
    sensitivity: float | None
    specificity: float | None
    precision: float
    threshold: float
    true_positive: int
    true_negative: int
    false_positive: int
    false_negative: int
    num_examples: int
    positive_examples: int
    negative_examples: int

    def to_dict(self) -> dict[str, float | int | None]:
        return asdict(self)


def train_one_epoch(
    model: torch.nn.Module,
    loader,
    criterion,
    optimizer,
    device: torch.device,
    *,
    model_train_setup: Callable[[torch.nn.Module], None] | None = None,
    amp_enabled: bool = False,
    scaler: torch.amp.GradScaler | None = None,
    gradient_clip_norm: float | None = None,
) -> EpochMetrics:
    model.train()
    if model_train_setup is not None:
        model_train_setup(model)
    total_loss = 0.0
    total_examples = 0
    all_logits: list[torch.Tensor] = []
    all_targets: list[torch.Tensor] = []

    for batch in loader:
        images = batch["image"].to(device)
        targets = batch["label"].float().to(device).view(-1, 1)

        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type=device.type, dtype=_amp_dtype(device), enabled=amp_enabled):
            logits = model(images)
            loss = criterion(logits, targets)

        if scaler is not None and scaler.is_enabled():
            scaler.scale(loss).backward()
            if gradient_clip_norm is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if gradient_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
            optimizer.step()

        batch_size = int(targets.shape[0])
        total_loss += float(loss.detach().item()) * batch_size
        total_examples += batch_size
        all_logits.append(logits.detach().float().cpu().view(-1))
        all_targets.append(targets.detach().long().cpu().view(-1))

    binary_metrics = compute_binary_classification_metrics(
        logits=torch.cat(all_logits),
        targets=torch.cat(all_targets),
    )

    return EpochMetrics(
        loss=total_loss / max(total_examples, 1),
        accuracy=binary_metrics.accuracy,
        auroc=binary_metrics.auroc,
        f1=binary_metrics.f1,
        sensitivity=binary_metrics.sensitivity,
        specificity=binary_metrics.specificity,
        precision=binary_metrics.precision,
        threshold=binary_metrics.threshold,
        true_positive=binary_metrics.true_positive,
        true_negative=binary_metrics.true_negative,
        false_positive=binary_metrics.false_positive,
        false_negative=binary_metrics.false_negative,
        num_examples=binary_metrics.num_examples,
        positive_examples=binary_metrics.positive_examples,
        negative_examples=binary_metrics.negative_examples,
    )


def evaluate_one_epoch(
    model: torch.nn.Module,
    loader,
    criterion,
    device: torch.device,
    *,
    amp_enabled: bool = False,
) -> EpochMetrics:
    model.eval()
    total_loss = 0.0
    total_examples = 0
    all_logits: list[torch.Tensor] = []
    all_targets: list[torch.Tensor] = []

    with torch.inference_mode():
        for batch in loader:
            images = batch["image"].to(device)
            targets = batch["label"].float().to(device).view(-1, 1)
            with torch.autocast(device_type=device.type, dtype=_amp_dtype(device), enabled=amp_enabled):
                logits = model(images)
                loss = criterion(logits, targets)

            batch_size = int(targets.shape[0])
            total_loss += float(loss.item()) * batch_size
            total_examples += batch_size
            all_logits.append(logits.detach().float().cpu().view(-1))
            all_targets.append(targets.detach().long().cpu().view(-1))

    binary_metrics = compute_binary_classification_metrics(
        logits=torch.cat(all_logits),
        targets=torch.cat(all_targets),
    )

    return EpochMetrics(
        loss=total_loss / max(total_examples, 1),
        accuracy=binary_metrics.accuracy,
        auroc=binary_metrics.auroc,
        f1=binary_metrics.f1,
        sensitivity=binary_metrics.sensitivity,
        specificity=binary_metrics.specificity,
        precision=binary_metrics.precision,
        threshold=binary_metrics.threshold,
        true_positive=binary_metrics.true_positive,
        true_negative=binary_metrics.true_negative,
        false_positive=binary_metrics.false_positive,
        false_negative=binary_metrics.false_negative,
        num_examples=binary_metrics.num_examples,
        positive_examples=binary_metrics.positive_examples,
        negative_examples=binary_metrics.negative_examples,
    )
