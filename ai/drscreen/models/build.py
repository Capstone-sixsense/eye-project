from __future__ import annotations

import timm
import torch
import torch.nn as nn
from timm.layers import EcaModule
from timm.layers.cbam import SpatialAttn
from torchvision import models

from drscreen.models.profiles import get_weights_enum


class IBN(nn.Module):
    """IBN-a: per-channel split into InstanceNorm (first half) + BatchNorm (second half).

    Applied to BN layers in shallow EfficientNet-B5 blocks to remove domain
    style statistics while preserving discriminative features in the BN half.
    Reference: Pan et al., ECCV 2018 -- IBN-Net.
    """

    def __init__(self, num_features: int) -> None:
        super().__init__()
        half = num_features // 2
        self.in_norm = nn.InstanceNorm2d(half, affine=True)
        self.bn_norm = nn.BatchNorm2d(num_features - half)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        half = x.size(1) // 2
        return torch.cat([self.in_norm(x[:, :half]), self.bn_norm(x[:, half:])], dim=1)


def _inject_ibn(model: nn.Module, num_blocks: int = 3) -> None:
    """Replace BatchNorm2d with IBN-a in the first *num_blocks* block groups.

    State dict keys change after injection (e.g. bn1.weight -> bn1.in_norm.weight +
    bn1.bn_norm.weight), so pretrained checkpoints must be loaded with strict=False.
    Deep blocks (num_blocks and beyond) keep standard BN to preserve discriminative
    features needed for DR lesion classification.
    """
    for i in range(min(num_blocks, len(model.blocks))):
        for block in model.blocks[i]:
            for name, module in list(block.named_children()):
                if isinstance(module, nn.BatchNorm2d):
                    setattr(block, name, IBN(module.num_features))


class _EcaSpatialAttn(nn.Module):
    """Combined ECA channel attention + CBAM spatial attention as a timm se_layer.

    Passed via se_layer=_EcaSpatialAttn so attention is integrated inside each
    MBConv block at the SE position, not applied externally on top of the block
    output. This keeps Grad-CAM target layers clean -- model.blocks[-1] output
    is the standard residual block output, not an attention-modulated surface.
    """

    def __init__(self, channels: int, **kwargs) -> None:
        super().__init__()
        self.eca = EcaModule(channels, **kwargs)
        self.spatial = SpatialAttn(kernel_size=7)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.spatial(self.eca(x))


# ---------------------------------------------------------------------------
# Variant builders (internal) — called by build_model() dispatcher
# ---------------------------------------------------------------------------

def _build_efficientnet_b5(
    *,
    pretrained: bool,
    num_outputs: int,
    use_attention: bool,
    use_ibn: bool,
    grad_checkpointing: bool,
    in_channels: int = 3,
) -> nn.Module:
    se_layer = _EcaSpatialAttn if use_attention else EcaModule
    model = timm.create_model(
        "efficientnet_b5",
        pretrained=pretrained,
        se_layer=se_layer,
        num_classes=num_outputs,
        in_chans=in_channels,
    )
    if use_ibn:
        _inject_ibn(model)
    if grad_checkpointing:
        model.set_grad_checkpointing(True)
    return model


def _build_multitask_aux_seg(
    *,
    pretrained: bool,
    num_outputs: int,
    use_attention: bool,
    use_ibn: bool,
    grad_checkpointing: bool,
    aux_seg_block: int,
    aux_seg_output_size: int,
) -> nn.Module:
    from drscreen.models.aux_seg import MultiTaskModel
    backbone = _build_efficientnet_b5(
        pretrained=pretrained,
        num_outputs=num_outputs,
        use_attention=use_attention,
        use_ibn=use_ibn,
        grad_checkpointing=grad_checkpointing,
    )
    return MultiTaskModel(backbone, block_index=aux_seg_block, output_size=aux_seg_output_size)


def _build_mil_attention(
    *,
    pretrained: bool,
    num_outputs: int,
    use_attention: bool,
    use_ibn: bool,
    grad_checkpointing: bool,
) -> nn.Module:
    from drscreen.models.mil_attention import MILAttentionModel
    backbone = _build_efficientnet_b5(
        pretrained=pretrained,
        num_outputs=num_outputs,
        use_attention=use_attention,
        use_ibn=use_ibn,
        grad_checkpointing=grad_checkpointing,
    )
    return MILAttentionModel(backbone, num_outputs=num_outputs)


# ---------------------------------------------------------------------------
# Public API — signature preserved for all existing callers
# ---------------------------------------------------------------------------

def build_model(
    model_name: str,
    pretrained: bool = True,
    num_outputs: int = 1,
    use_attention: bool = False,
    use_ibn: bool = False,
    grad_checkpointing: bool = False,
    use_aux_seg: bool = False,
    aux_seg_block: int = 2,
    aux_seg_output_size: int = 512,
    use_mil_attention: bool = False,
    in_channels: int = 3,
) -> nn.Module:
    if model_name == "efficientnet_b5":
        if use_aux_seg:
            return _build_multitask_aux_seg(
                pretrained=pretrained,
                num_outputs=num_outputs,
                use_attention=use_attention,
                use_ibn=use_ibn,
                grad_checkpointing=grad_checkpointing,
                aux_seg_block=aux_seg_block,
                aux_seg_output_size=aux_seg_output_size,
            )
        if use_mil_attention:
            return _build_mil_attention(
                pretrained=pretrained,
                num_outputs=num_outputs,
                use_attention=use_attention,
                use_ibn=use_ibn,
                grad_checkpointing=grad_checkpointing,
            )
        return _build_efficientnet_b5(
            pretrained=pretrained,
            num_outputs=num_outputs,
            use_attention=use_attention,
            use_ibn=use_ibn,
            grad_checkpointing=grad_checkpointing,
            in_channels=in_channels,
        )

    weights = get_weights_enum(model_name) if pretrained else None

    if model_name == "resnet50":
        model = models.resnet50(weights=weights)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_outputs)
        return model

    if model_name == "convnext_tiny":
        model = models.convnext_tiny(weights=weights)
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_outputs)
        return model

    raise ValueError(f"Unsupported model architecture: {model_name}")


def get_classifier_module(model_name: str, model: nn.Module) -> nn.Module:
    # Unwrap MultiTaskModel so attribute access targets the backbone
    backbone = getattr(model, "backbone", model)

    if model_name in {"efficientnet_b5", "convnext_tiny"}:
        return backbone.classifier

    if model_name == "resnet50":
        return backbone.fc

    raise ValueError(f"Unsupported model architecture: {model_name}")


def split_model_parameters(
    model_name: str,
    model: nn.Module,
) -> tuple[list[nn.Parameter], list[nn.Parameter]]:
    from drscreen.models.mil_attention import MILAttentionModel

    if isinstance(model, MILAttentionModel):
        head_ids = (
            {id(p) for p in model.attn_pool.parameters()}
            | {id(p) for p in model.classifier.parameters()}
        )
    else:
        classifier = get_classifier_module(model_name, model)
        head_ids = {id(p) for p in classifier.parameters()}

        seg_head = getattr(model, "seg_head", None)
        if seg_head is not None:
            head_ids |= {id(p) for p in seg_head.parameters()}

    backbone_parameters: list[nn.Parameter] = []
    head_parameters: list[nn.Parameter] = []
    for parameter in model.parameters():
        if id(parameter) in head_ids:
            head_parameters.append(parameter)
        else:
            backbone_parameters.append(parameter)

    return backbone_parameters, head_parameters
