from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass
from typing import Callable

import torch

from drscreen.train.metrics import compute_binary_classification_metrics


class SWADBuffer:
    """Rolling buffer for Stochastic Weight Averaging Dense (SWAD).

    Maintains the last ``n`` model state dicts and computes their
    parameter-wise mean. Non-floating-point buffers (e.g. BatchNorm
    ``num_batches_tracked``) are taken from the most recent snapshot.

    Args:
        n: Window size. Only the last n model snapshots are retained.
    """

    def __init__(self, n: int) -> None:
        if n < 1:
            raise ValueError(f"SWADBuffer requires n >= 1, got {n}")
        self._buffer: deque[dict[str, torch.Tensor]] = deque(maxlen=n)

    def update(self, model: torch.nn.Module) -> None:
        """Append a snapshot of the current model weights to the buffer."""
        self._buffer.append({k: v.cpu().clone() for k, v in model.state_dict().items()})

    def get_averaged_state_dict(self) -> dict[str, torch.Tensor] | None:
        """Return the parameter-wise mean of all buffered snapshots.

        Floating-point tensors are averaged. Non-floating-point tensors
        (e.g. BatchNorm num_batches_tracked) are taken from the latest snapshot.
        Returns None if the buffer is empty.
        """
        if not self._buffer:
            return None
        latest = self._buffer[-1]
        avg: dict[str, torch.Tensor] = {}
        for key in latest:
            tensors = [s[key] for s in self._buffer if key in s]
            if tensors[0].is_floating_point():
                avg[key] = (
                    torch.stack([t.float() for t in tensors])
                    .mean(0)
                    .to(tensors[0].dtype)
                )
            else:
                avg[key] = latest[key].clone()
        return avg

    def __len__(self) -> int:
        return len(self._buffer)


def _unpack_batch(
    batch: dict[str, torch.Tensor], device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    return batch["image"].to(device), batch["label"].float().to(device).view(-1, 1)


def _has_timm_feature_api(model: torch.nn.Module) -> bool:
    return (
        hasattr(model, "forward_features")
        and hasattr(model, "forward_head")
        and hasattr(model, "classifier")
    )


def _forward_with_features(
    model: torch.nn.Module, images: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (pooled_features, logits) using timm's split forward API."""
    feat_map = model.forward_features(images)
    pooled = model.forward_head(feat_map, pre_logits=True)
    logits = model.classifier(pooled)
    return pooled, logits


def _compute_coral_loss(
    coral_criterion: torch.nn.Module,
    pooled: torch.Tensor,
    domains: list[str],
) -> torch.Tensor:
    domain_to_indices: dict[str, list[int]] = {}
    for i, d in enumerate(domains):
        domain_to_indices.setdefault(d, []).append(i)

    unique_domains = list(domain_to_indices.keys())
    if len(unique_domains) < 2:
        return pooled.new_tensor(0.0)

    losses: list[torch.Tensor] = []
    for i in range(len(unique_domains)):
        for j in range(i + 1, len(unique_domains)):
            idx1 = torch.tensor(domain_to_indices[unique_domains[i]], device=pooled.device)
            idx2 = torch.tensor(domain_to_indices[unique_domains[j]], device=pooled.device)
            f1, f2 = pooled[idx1], pooled[idx2]
            if f1.size(0) >= 2 and f2.size(0) >= 2:
                losses.append(coral_criterion(f1, f2))

    return torch.stack(losses).mean() if losses else pooled.new_tensor(0.0)


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
    coral_criterion: torch.nn.Module | None = None,
    lambda_coral: float = 0.0,
    lambda_aux_seg: float = 0.0,
) -> EpochMetrics:
    model.train()
    if model_train_setup is not None:
        model_train_setup(model)

    use_coral = coral_criterion is not None and lambda_coral > 0.0 and _has_timm_feature_api(model)
    use_aux_seg = lambda_aux_seg > 0.0
    total_loss = 0.0
    total_examples = 0
    all_logits: list[torch.Tensor] = []
    all_targets: list[torch.Tensor] = []

    for batch in loader:
        images, targets = _unpack_batch(batch, device)
        domains: list[str] | None = batch.get("domain") if use_coral else None

        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type=device.type, dtype=_amp_dtype(device), enabled=amp_enabled):
            if use_coral and domains is not None:
                pooled, logits = _forward_with_features(model, images)
                cls_loss = criterion(logits, targets)
                coral_loss = _compute_coral_loss(coral_criterion, pooled, domains)
                loss = cls_loss + lambda_coral * coral_loss
            else:
                output = model(images)
                if isinstance(output, tuple):
                    logits, seg_logits = output
                else:
                    logits, seg_logits = output, None
                loss = criterion(logits, targets)
                if use_aux_seg and seg_logits is not None:
                    valid = batch.get("seg_mask_valid")
                    if valid is not None:
                        valid = valid.to(device)
                        if valid.any():
                            seg_targets = batch["seg_mask"].to(device)
                            import torch.nn.functional as F
                            seg_loss = F.binary_cross_entropy_with_logits(
                                seg_logits[valid], seg_targets[valid]
                            )
                            loss = loss + lambda_aux_seg * seg_loss

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


def collect_logits_and_targets(
    model: torch.nn.Module,
    loader,
    device: torch.device,
    *,
    amp_enabled: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run inference and return raw (logits, targets) tensors without computing metrics."""
    model.eval()
    all_logits: list[torch.Tensor] = []
    all_targets: list[torch.Tensor] = []
    with torch.inference_mode():
        for batch in loader:
            images, targets = _unpack_batch(batch, device)
            with torch.autocast(device_type=device.type, dtype=_amp_dtype(device), enabled=amp_enabled):
                logits = model(images)
            all_logits.append(logits.detach().float().cpu().view(-1))
            all_targets.append(targets.detach().long().cpu().view(-1))
    return torch.cat(all_logits), torch.cat(all_targets)


def evaluate_one_epoch(
    model: torch.nn.Module,
    loader,
    criterion,
    device: torch.device,
    *,
    amp_enabled: bool = False,
    threshold: float = 0.5,
) -> EpochMetrics:
    model.eval()
    total_loss = 0.0
    total_examples = 0
    all_logits: list[torch.Tensor] = []
    all_targets: list[torch.Tensor] = []

    with torch.inference_mode():
        for batch in loader:
            images, targets = _unpack_batch(batch, device)
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
        threshold=threshold,
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
