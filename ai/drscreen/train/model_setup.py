"""학습/평가 실행에 필요한 구성요소 빌더 모음.

장치(device) 결정, 학습 단계(phase) 정의, 단계별 학습 가능 파라미터 설정, 옵티마이저/
스케줄러/손실 구성, 그리고 학습/평가용 모델 빌드와 사전학습 backbone 로딩을 담당한다.

학습 단계: head(분류 헤드만 학습) -> finetune(backbone까지 학습)의 2단계. head 단계에서는
backbone을 freeze하되 BatchNorm은 train 모드로 둬, ImageNet 통계와 안저(512px) 분포의
불일치로 인한 활성값 폭주를 막는다(prepare_model_for_head_only_training 참조).
옵티마이저는 head/backbone에 서로 다른 학습률을 주는 파라미터 그룹을 구성한다.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    LinearLR,
    LRScheduler,
    SequentialLR,
)

from drscreen.models.build import (
    build_model,
    get_classifier_module,
    split_model_parameters,
)
from drscreen.settings import resolve_checkpoint_path
from drscreen.utils.checkpoint import (
    load_state_dict_with_shape_filter,
    load_state_from_checkpoint,
)
from drscreen.utils.logging import get_logger

LOGGER = get_logger(__name__)

# 체크포인트 승격을 위한 기본 민감도 하한. 이 값 미만인 에폭은 best 후보에서 제외된다.
_DEFAULT_MIN_SENSITIVITY = 0.80


def resolve_device(device_name: str) -> torch.device:
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        LOGGER.warning("CUDA was requested but is not available. Falling back to CPU.")
        return torch.device("cpu")
    return torch.device(device_name)


@dataclass(frozen=True, slots=True)
class TrainingPhase:
    name: str
    epochs: int
    head_only: bool


def validate_training_scope(config: dict[str, Any]) -> None:
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


def build_training_phases(config: dict[str, Any]) -> list[TrainingPhase]:
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


def set_phase_trainability(model: nn.Module, architecture: str, *, head_only: bool) -> None:
    backbone_parameters, head_parameters = split_model_parameters(architecture, model)
    for parameter in backbone_parameters:
        parameter.requires_grad = not head_only
    for parameter in head_parameters:
        parameter.requires_grad = True


def prepare_model_for_head_only_training(model: nn.Module, architecture: str) -> None:
    classifier = get_classifier_module(architecture, model)
    for module in model.children():
        module.eval()
    classifier.train()
    seg_head = getattr(model, "seg_head", None)
    if seg_head is not None:
        seg_head.train()
    # Keep BN layers in train mode so they use batch statistics.
    # When backbone starts from ImageNet weights, the running stats are
    # calibrated for ImageNet distribution. Fundus images at 448px are
    # out-of-distribution, causing activation explosion in eval() mode.
    # Using batch stats (train mode) avoids this without unfreezing parameters.
    for module in model.modules():
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            module.train()


def prepare_model_for_decoder_only_training(model: nn.Module) -> None:
    """Freeze classifier/backbone path and train only the auxiliary decoder head."""
    model.train()
    for parameter in model.parameters():
        parameter.requires_grad = False

    backbone = getattr(model, "backbone", None)
    if backbone is not None:
        backbone.eval()

    seg_head = getattr(model, "seg_head", None)
    if seg_head is not None:
        seg_head.train()
        for parameter in seg_head.parameters():
            parameter.requires_grad = True
    lesion_weights = getattr(model, "lesion_weights", None)
    if lesion_weights is not None:
        lesion_weights.requires_grad = True

    model._decoder_only = True


def build_optimizer(
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

    weight_decay = float(train_cfg["weight_decay"])
    head_learning_rate = float(train_cfg["head_learning_rate"])
    backbone_learning_rate = float(train_cfg["backbone_learning_rate"])
    backbone_parameters, head_parameters = split_model_parameters(architecture, model)

    if head_only:
        parameter_groups = [
            {"params": [p for p in head_parameters if p.requires_grad], "lr": head_learning_rate}
        ]
    else:
        parameter_groups = []
        if head_parameters:
            parameter_groups.append(
                {"params": [p for p in head_parameters if p.requires_grad], "lr": head_learning_rate}
            )
        if backbone_parameters:
            parameter_groups.append(
                {"params": [p for p in backbone_parameters if p.requires_grad], "lr": backbone_learning_rate}
            )

    if not any(group["params"] for group in parameter_groups):
        raise ValueError("No trainable parameters were available for the requested phase.")
    return AdamW(parameter_groups, weight_decay=weight_decay)


def build_scheduler(config: dict[str, Any], optimizer: Optimizer, epochs: int) -> LRScheduler | None:
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


def build_criterion(config: dict[str, Any]) -> nn.Module:
    validate_training_scope(config)
    loss_name = str(config["train"].get("loss", "bce")).lower()
    if loss_name == "focal":
        from drscreen.train.loss import BinaryFocalLoss
        gamma = float(config["train"].get("focal_gamma", 2.0))
        alpha_cfg = config["train"].get("focal_alpha")
        alpha = float(alpha_cfg) if alpha_cfg is not None else None
        return BinaryFocalLoss(gamma=gamma, alpha=alpha)
    if loss_name == "bce":
        pos_weight_cfg = config["train"].get("pos_weight")
        if pos_weight_cfg is not None:
            return nn.BCEWithLogitsLoss(pos_weight=torch.tensor([float(pos_weight_cfg)]))
        return nn.BCEWithLogitsLoss()
    raise ValueError(f"Unsupported loss '{loss_name}'. Supported: 'bce', 'focal'.")


def _resolve_decoder_options(config: dict[str, Any]) -> tuple[str, list[int] | None]:
    model_cfg = config["model"]
    decoder_type = str(model_cfg.get("decoder_type", "single_block"))
    raw_blocks = model_cfg.get("decoder_blocks")
    decoder_blocks: list[int] | None = (
        [int(b) for b in raw_blocks] if raw_blocks is not None else None
    )
    return decoder_type, decoder_blocks


def build_model_for_training(config: dict[str, Any], device: torch.device) -> nn.Module:
    architecture = str(config["model"]["architecture"])
    decoder_type, decoder_blocks = _resolve_decoder_options(config)
    model_cfg = config["model"]
    return build_model(
        architecture,
        pretrained=bool(model_cfg["pretrained"]),
        num_outputs=int(model_cfg["num_outputs"]),
        use_attention=bool(model_cfg.get("use_attention", False)),
        attention_mode=model_cfg.get("attention_mode"),
        use_ibn=bool(model_cfg.get("use_ibn", False)),
        grad_checkpointing=bool(model_cfg.get("grad_checkpointing", False)),
        use_aux_seg=bool(model_cfg.get("use_aux_seg", False)),
        aux_seg_block=int(model_cfg.get("aux_seg_block", 2)),
        aux_seg_output_size=int(config["data"].get("image_size", 512)),
        aux_seg_channels=int(
            model_cfg.get(
                "aux_seg_channels",
                config.get("data", {}).get("seg_mask_channels", 1),
            )
        ),
        use_gated_pooling=bool(model_cfg.get("use_gated_pooling", False)),
        use_mil_attention=bool(model_cfg.get("use_mil_attention", False)),
        in_channels=int(model_cfg.get("in_channels", 3)),
        decoder_type=decoder_type,
        decoder_blocks=decoder_blocks,
        bagnet_patch_size=int(model_cfg.get("bagnet_patch_size", 33)),
        bagnet_patch_stride=int(model_cfg.get("bagnet_patch_stride", 8)),
        bagnet_hidden_channels=int(model_cfg.get("bagnet_hidden_channels", 128)),
        bagnet_depth=int(model_cfg.get("bagnet_depth", 4)),
        bagnet_dropout=float(model_cfg.get("bagnet_dropout", 0.15)),
        bagnet_aggregation=str(model_cfg.get("bagnet_aggregation", "mean")),
        concept_block=int(model_cfg.get("concept_block", 4)),
        concept_channels=int(model_cfg.get("concept_channels", 4)),
        concept_head_hidden_channels=(
            int(model_cfg["concept_head_hidden_channels"])
            if model_cfg.get("concept_head_hidden_channels") is not None
            else None
        ),
        concept_dropout=float(model_cfg.get("concept_dropout", 0.3)),
        segmenter_encoder=str(model_cfg.get("segmenter_encoder", "resnet50")),
        segmenter_out_channels=int(model_cfg.get("segmenter_out_channels", 4)),
        segmenter_decoder_channels=(
            [int(channel) for channel in model_cfg["segmenter_decoder_channels"]]
            if model_cfg.get("segmenter_decoder_channels") is not None
            else None
        ),
    ).to(device)


def build_model_for_eval(config: dict[str, Any], device: torch.device) -> nn.Module:
    decoder_type, decoder_blocks = _resolve_decoder_options(config)
    model_cfg = config["model"]
    return build_model(
        str(model_cfg["architecture"]),
        pretrained=False,
        num_outputs=int(model_cfg["num_outputs"]),
        use_attention=bool(model_cfg.get("use_attention", False)),
        attention_mode=model_cfg.get("attention_mode"),
        use_ibn=bool(model_cfg.get("use_ibn", False)),
        use_aux_seg=bool(model_cfg.get("use_aux_seg", False)),
        aux_seg_block=int(model_cfg.get("aux_seg_block", 2)),
        aux_seg_output_size=int(model_cfg.get("aux_seg_output_size", 512)),
        aux_seg_channels=int(
            model_cfg.get(
                "aux_seg_channels",
                config.get("data", {}).get("seg_mask_channels", 1),
            )
        ),
        use_gated_pooling=bool(model_cfg.get("use_gated_pooling", False)),
        use_mil_attention=bool(model_cfg.get("use_mil_attention", False)),
        in_channels=int(model_cfg.get("in_channels", 3)),
        decoder_type=decoder_type,
        decoder_blocks=decoder_blocks,
        bagnet_patch_size=int(model_cfg.get("bagnet_patch_size", 33)),
        bagnet_patch_stride=int(model_cfg.get("bagnet_patch_stride", 8)),
        bagnet_hidden_channels=int(model_cfg.get("bagnet_hidden_channels", 128)),
        bagnet_depth=int(model_cfg.get("bagnet_depth", 4)),
        bagnet_dropout=float(model_cfg.get("bagnet_dropout", 0.15)),
        bagnet_aggregation=str(model_cfg.get("bagnet_aggregation", "mean")),
        concept_block=int(model_cfg.get("concept_block", 4)),
        concept_channels=int(model_cfg.get("concept_channels", 4)),
        concept_head_hidden_channels=(
            int(model_cfg["concept_head_hidden_channels"])
            if model_cfg.get("concept_head_hidden_channels") is not None
            else None
        ),
        concept_dropout=float(model_cfg.get("concept_dropout", 0.3)),
        segmenter_encoder=str(model_cfg.get("segmenter_encoder", "resnet50")),
        segmenter_out_channels=int(model_cfg.get("segmenter_out_channels", 4)),
        segmenter_decoder_channels=(
            [int(channel) for channel in model_cfg["segmenter_decoder_channels"]]
            if model_cfg.get("segmenter_decoder_channels") is not None
            else None
        ),
    ).to(device)


def load_pretrained_backbone(
    model: nn.Module,
    config: dict[str, Any],
    project_root: Path,
) -> None:
    path_str = str(config["train"].get("pretrained_backbone_path", "")).strip()
    if not path_str:
        return
    path = resolve_checkpoint_path(project_root, path_str)
    if not path.exists():
        LOGGER.warning("pretrained_backbone_path not found: %s", path)
        return
    backbone_ckpt = torch.load(path, map_location="cpu", weights_only=False)
    state = backbone_ckpt.get("model_state_dict", backbone_ckpt)
    if hasattr(model, "backbone") and not any(
        str(key).startswith("backbone.") for key in state
    ):
        missing, unexpected = load_state_dict_with_shape_filter(
            model.backbone,
            state,
            strict=False,
        )
    else:
        missing, unexpected = load_state_from_checkpoint(model, backbone_ckpt, strict=False)
    LOGGER.info(
        "Loaded pretrained backbone from %s (missing=%d unexpected=%d)",
        path, len(missing), len(unexpected),
    )
