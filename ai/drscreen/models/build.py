from __future__ import annotations

from pathlib import Path

import timm
import torch
import torch.nn as nn
from timm.layers import EcaModule
from timm.layers.cbam import SpatialAttn
from torchvision import models

from drscreen.models.mixstyle import MixStyle
from drscreen.models.profiles import get_weights_enum


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


def _inject_mixstyle(model: nn.Module, num_blocks: int = 3) -> None:
    """Attach MixStyle to the first *num_blocks* block groups via forward hooks.

    Uses ``register_forward_hook`` so state_dict keys stay unchanged --
    pretrained / SSL checkpoints load without key remapping.  MixStyle is
    registered as a submodule on each block group (not on the top-level
    model) so ``model.train()`` / ``model.eval()`` propagates correctly
    without polluting ``model.children()`` iteration.
    """
    def _make_hook(ms: MixStyle):
        def hook(_module: nn.Module, _inp: tuple, out: torch.Tensor) -> torch.Tensor:
            return ms(out)
        return hook

    for i in range(min(num_blocks, len(model.blocks))):
        ms = MixStyle()
        model.blocks[i].add_module("_mixstyle", ms)
        model.blocks[i].register_forward_hook(_make_hook(ms))


def build_model(
    model_name: str,
    pretrained: bool = True,
    num_outputs: int = 1,
    use_attention: bool = False,
    use_mixstyle: bool = False,
    grad_checkpointing: bool = False,
    classifier_dropout: float = 0.0,
) -> nn.Module:
    if model_name == "efficientnet_b5":
        # timm build: se_layer controls the attention module inside each MBConv.
        # use_attention=True swaps EcaModule for _EcaSpatialAttn (ECA + spatial),
        # keeping attention integrated inside the block rather than wrapping it
        # externally. num_classes sets the final Linear head; drop_rate applies
        # dropout before the classifier during training.
        se_layer = _EcaSpatialAttn if use_attention else EcaModule
        model = timm.create_model(
            "efficientnet_b5",
            pretrained=pretrained,
            se_layer=se_layer,
            num_classes=num_outputs,
            drop_rate=classifier_dropout,
        )
        if use_mixstyle:
            _inject_mixstyle(model)
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
