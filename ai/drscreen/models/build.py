from __future__ import annotations

import timm
import torch
import torch.nn as nn
from timm.layers import EcaModule
from timm.layers.cbam import SpatialAttn
from torchvision import models

from drscreen.models.profiles import get_weights_enum


class _SpatialAttnWrapper(nn.Module):
    """Wraps a single MBConv block and applies spatial attention to its output."""

    def __init__(self, block: nn.Module) -> None:
        super().__init__()
        self.block = block
        self.spatial_attn = SpatialAttn(kernel_size=7)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.spatial_attn(self.block(x))


def _inject_spatial_attention(model: nn.Module) -> None:
    """Replace every MBConv block in model.blocks with a spatially-attended wrapper.

    timm EfficientNet has model.blocks: Sequential of 7 Sequential block groups.
    Each group contains one or more MBConv blocks.  We wrap each individual block
    so that spatial attention is applied after the block output (including its
    skip connection), matching the paper's design.
    """
    for group_idx in range(len(model.blocks)):
        group = model.blocks[group_idx]
        new_group: nn.Sequential = nn.Sequential()
        for block_idx in range(len(group)):
            new_group.add_module(str(block_idx), _SpatialAttnWrapper(group[block_idx]))
        model.blocks[group_idx] = new_group


def build_model(
    model_name: str,
    pretrained: bool = True,
    num_outputs: int = 1,
    use_attention: bool = False,
    grad_checkpointing: bool = False,
) -> nn.Module:
    if model_name == "efficientnet_b5":
        # timm build: ECA replaces every SE block via se_layer=EcaModule.
        # num_classes sets the final Linear head to the correct output size.
        model = timm.create_model(
            "efficientnet_b5",
            pretrained=pretrained,
            se_layer=EcaModule,
            num_classes=num_outputs,
        )
        if use_attention:
            _inject_spatial_attention(model)
        if grad_checkpointing:
            model.set_grad_checkpointing(True)
        return model

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
    if model_name in {"efficientnet_b5", "convnext_tiny"}:
        return model.classifier

    if model_name == "resnet50":
        return model.fc

    raise ValueError(f"Unsupported model architecture: {model_name}")


def split_model_parameters(
    model_name: str,
    model: nn.Module,
) -> tuple[list[nn.Parameter], list[nn.Parameter]]:
    classifier = get_classifier_module(model_name, model)
    classifier_parameter_ids = {id(parameter) for parameter in classifier.parameters()}

    backbone_parameters: list[nn.Parameter] = []
    head_parameters: list[nn.Parameter] = []
    for parameter in model.parameters():
        if id(parameter) in classifier_parameter_ids:
            head_parameters.append(parameter)
        else:
            backbone_parameters.append(parameter)

    return backbone_parameters, head_parameters
