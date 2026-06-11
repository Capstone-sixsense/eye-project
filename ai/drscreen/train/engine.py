"""에폭 단위 학습/평가 루프와 SWAD 가중치 버퍼.

train_one_epoch이 핵심이다. 기본은 분류 손실이지만, config 플래그에 따라 여러 보조
손실을 선택적으로 더한다(모두 0이면 순수 분류 학습):
- aux_seg: 병변 마스크 분할 손실(멀티태스크 supervision).
- cam_align: Layer-CAM 어트리뷰션을 병변 마스크에 정렬.
- coral: 도메인 간 특징 공분산 정렬(도메인 일반화).
- rsc: Representation Self-Challenging — 가장 기여 큰 특징을 마스킹해 robust feature 학습.
- concept / patch_l1: CBM 개념 손실 / BagNet patch-logit 희소화.

AMP는 _amp_dtype로 GPU 세대에 맞춰 bf16/fp16을 고른다. evaluate_one_epoch /
collect_logits_and_targets는 추론 전용(메트릭 계산 / 원시 logit 수집).
"""

from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass
from collections.abc import Callable

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


def _rsc_feature_mask(
    pooled: torch.Tensor,
    grad: torch.Tensor,
    *,
    p_feature: float,
) -> torch.Tensor:
    if not 0.0 < p_feature < 1.0:
        raise ValueError(f"rsc p_feature must be between 0 and 1, got {p_feature}")
    scores = (pooled.detach() * grad.detach()).abs()
    n_features = scores.shape[1]
    k = max(1, min(n_features - 1, int(round(n_features * p_feature))))
    threshold = torch.topk(scores, k, dim=1).values[:, -1].unsqueeze(1)
    return (scores < threshold).to(dtype=pooled.dtype)


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


def _decoder_only_allowed_parameter_ids(model: torch.nn.Module) -> set[int]:
    allowed: set[int] = set()
    seg_head = getattr(model, "seg_head", None)
    if seg_head is not None:
        allowed.update(id(parameter) for parameter in seg_head.parameters())
    lesion_weights = getattr(model, "lesion_weights", None)
    if lesion_weights is not None:
        allowed.add(id(lesion_weights))
    return allowed


def _assert_decoder_only_freeze(model: torch.nn.Module) -> None:
    allowed_ids = _decoder_only_allowed_parameter_ids(model)
    leaked = [
        name
        for name, parameter in model.named_parameters()
        if id(parameter) not in allowed_ids and parameter.grad is not None
    ]
    if leaked:
        raise AssertionError(f"decoder_only freeze leaked into: {leaked[:5]}")


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
    seg_loss_type: str = "bce",
    lambda_cam_align: float = 0.0,
    coral_block: int | None = None,
    lambda_patch_l1: float = 0.0,
    lambda_concept: float = 0.0,
    rsc_p_feature: float = 0.0,
    rsc_p_batch: float = 0.0,
) -> EpochMetrics:
    model.train()
    if model_train_setup is not None:
        model_train_setup(model)

    use_coral = coral_criterion is not None and lambda_coral > 0.0
    use_aux_seg = lambda_aux_seg > 0.0
    use_cam_align = lambda_cam_align > 0.0
    use_rsc = (
        rsc_p_feature > 0.0
        and rsc_p_batch > 0.0
        and hasattr(model, "forward_with_gated_features")
        and hasattr(model, "classify_pooled_features")
    )

    _seg_criterion: torch.nn.Module | None = None
    if use_aux_seg:
        if seg_loss_type == "dice_bce":
            from drscreen.train.loss import DiceBCELoss
            _seg_criterion = DiceBCELoss().to(device)
        else:
            _seg_criterion = None  # use F.binary_cross_entropy_with_logits inline

    _cam_align_criterion: torch.nn.Module | None = None
    if use_cam_align:
        from drscreen.train.loss import CamAlignmentLoss
        _cam_align_criterion = CamAlignmentLoss().to(device)
    total_loss = 0.0
    total_examples = 0
    all_logits: list[torch.Tensor] = []
    all_targets: list[torch.Tensor] = []
    decoder_only_checked = False

    for batch in loader:
        images, targets = _unpack_batch(batch, device)
        domains: list[str] | None = batch.get("domain") if use_coral else None

        # 배치마다: forward -> 분류 손실 -> (켜진 보조 손실들 누적) -> backward -> step.
        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type=device.type, dtype=_amp_dtype(device), enabled=amp_enabled):
            use_intermediate_coral = (
                use_coral
                and coral_block is not None
                and domains is not None
            )
            if use_coral and not use_intermediate_coral and not use_aux_seg and _has_timm_feature_api(model):
                # Legacy CORAL-only path: pool the final pre-classifier feature.
                pooled, logits = _forward_with_features(model, images)
                seg_logits = None
                loss = criterion(logits, targets)
                coral_loss = _compute_coral_loss(coral_criterion, pooled, domains)
                loss = loss + lambda_coral * coral_loss
            else:
                if use_rsc:
                    logits, seg_logits, pooled = model.forward_with_gated_features(images)
                    cls_loss = criterion(logits, targets)
                    apply_rsc = bool(
                        torch.rand((), device=device).item() < min(1.0, rsc_p_batch)
                    )
                    if apply_rsc:
                        pooled_grad = torch.autograd.grad(
                            cls_loss,
                            pooled,
                            retain_graph=True,
                            create_graph=False,
                        )[0]
                        rsc_mask = _rsc_feature_mask(
                            pooled,
                            pooled_grad,
                            p_feature=min(0.999, rsc_p_feature),
                        )
                        logits = model.classify_pooled_features(pooled * rsc_mask)
                        loss = criterion(logits, targets)
                    else:
                        loss = cls_loss
                else:
                    output = model(images)
                    if isinstance(output, tuple):
                        logits, seg_logits = output
                    else:
                        logits, seg_logits = output, None
                    loss = criterion(logits, targets)

                if lambda_patch_l1 > 0.0 and hasattr(model, "latest_patch_logits"):
                    patch_logits = model.latest_patch_logits()
                    if patch_logits is not None:
                        loss = loss + lambda_patch_l1 * patch_logits.abs().mean()

                if lambda_concept > 0.0 and hasattr(model, "latest_concept_logits"):
                    concept_logits = model.latest_concept_logits()
                    concept_valid = batch.get("concept_valid")
                    if concept_logits is not None and concept_valid is not None:
                        concept_valid = concept_valid.to(device).bool()
                        if concept_valid.any():
                            import torch.nn.functional as F

                            concept_targets = batch["concept_labels"].to(device).float()
                            concept_conf = batch.get("concept_confidence")
                            if concept_conf is None:
                                concept_conf = torch.ones_like(concept_valid, dtype=concept_logits.dtype)
                            else:
                                concept_conf = concept_conf.to(device).float()
                            raw_concept_loss = F.binary_cross_entropy_with_logits(
                                concept_logits[concept_valid],
                                concept_targets[concept_valid],
                                reduction="none",
                            ).mean(dim=1)
                            weights = concept_conf[concept_valid].clamp_min(0.0)
                            if weights.sum() > 0:
                                concept_loss = (raw_concept_loss * weights).sum() / weights.sum()
                                loss = loss + lambda_concept * concept_loss

                if use_intermediate_coral:
                    decoder_feats = getattr(model, "_decoder_feats", None)
                    coral_feat: torch.Tensor | None = None
                    if decoder_feats is not None and coral_block in decoder_feats:
                        coral_act = decoder_feats[coral_block]
                        coral_feat = torch.nn.functional.adaptive_avg_pool2d(
                            coral_act, 1
                        ).flatten(1)
                    if coral_feat is not None:
                        coral_loss = _compute_coral_loss(
                            coral_criterion, coral_feat, domains
                        )
                        loss = loss + lambda_coral * coral_loss

                if use_aux_seg and seg_logits is not None:
                    # 마스크가 유효한 샘플(seg_mask_valid)에만 분할 손실을 건다. 마스크 없는
                    # 도메인 행이 0 마스크로 잘못된 음성 supervision을 주지 않도록 필터링.
                    valid = batch.get("seg_mask_valid")
                    if valid is not None:
                        valid = valid.to(device)
                        if valid.any():
                            seg_targets = batch["seg_mask"].to(device)
                            import torch.nn.functional as F
                            if _seg_criterion is not None:
                                seg_loss = _seg_criterion(seg_logits[valid], seg_targets[valid])
                            else:
                                seg_loss = F.binary_cross_entropy_with_logits(
                                    seg_logits[valid], seg_targets[valid]
                                )
                            loss = loss + lambda_aux_seg * seg_loss

                            if (
                                use_cam_align
                                and _cam_align_criterion is not None
                                and getattr(model, "_feat", None) is not None
                            ):
                                act = model._feat.get("x")
                                if act is not None and act.requires_grad:
                                    grad_act = torch.autograd.grad(
                                        logits.sum(),
                                        act,
                                        create_graph=True,
                                        retain_graph=True,
                                    )[0]
                                    cam_loss = _cam_align_criterion(
                                        act, grad_act, seg_targets, valid
                                    )
                                    loss = loss + lambda_cam_align * cam_loss

        if loss.requires_grad:
            if scaler is not None and scaler.is_enabled():
                scaler.scale(loss).backward()
                if getattr(model, "_decoder_only", False) and not decoder_only_checked:
                    _assert_decoder_only_freeze(model)
                    decoder_only_checked = True
                if gradient_clip_norm is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if getattr(model, "_decoder_only", False) and not decoder_only_checked:
                    _assert_decoder_only_freeze(model)
                    decoder_only_checked = True
                if gradient_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
                optimizer.step()

        batch_size = int(targets.shape[0])
        total_loss += float(loss.detach().item()) * batch_size
        total_examples += batch_size
        all_logits.append(logits.detach().float().cpu().view(-1))
        all_targets.append(targets.detach().long().cpu().view(-1))

    if getattr(model, "_decoder_only", False) and use_aux_seg and not decoder_only_checked:
        raise RuntimeError("decoder_only training did not encounter a trainable segmentation batch.")

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
