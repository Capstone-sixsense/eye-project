from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(slots=True)
class GradCamResult:
    heatmap: torch.Tensor


def resolve_default_target_layer(model: torch.nn.Module) -> torch.nn.Module:
    if hasattr(model, "features"):
        return model.features[-1]
    if hasattr(model, "blocks"):
        return model.blocks[-1]
    if hasattr(model, "layer4"):
        return model.layer4[-1]
    raise ValueError("Could not infer Grad-CAM target layer from model.")


def generate_gradcam(
    model: torch.nn.Module,
    inputs: torch.Tensor,
    target_layer: torch.nn.Module | None = None,
    target_layer_name: str | None = None,
    class_index: int | None = None,
) -> GradCamResult:
    if target_layer is not None:
        layer = target_layer
    elif target_layer_name:
        layer = model.get_submodule(target_layer_name)
    else:
        layer = resolve_default_target_layer(model)

    activations = {}
    gradients = {}

    def forward_hook(_, __, output):
        activations["value"] = output.detach()

    def backward_hook(_, grad_input, grad_output):
        del grad_input
        gradients["value"] = grad_output[0].detach()

    forward_handle = layer.register_forward_hook(forward_hook)
    backward_handle = layer.register_full_backward_hook(backward_hook)

    try:
        model.zero_grad(set_to_none=True)
        outputs = model(inputs)
        if outputs.ndim != 2:
            raise ValueError("Expected logits shape [batch, classes or 1].")

        if outputs.shape[1] == 1:
            score = outputs[:, 0].sum()
        else:
            target = class_index if class_index is not None else int(outputs.argmax(dim=1).item())
            score = outputs[:, target].sum()

        score.backward()

        activation = activations["value"]
        gradient = gradients["value"]
        weights = gradient.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((weights * activation).sum(dim=1, keepdim=True))
        cam = F.interpolate(
            cam,
            size=inputs.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        cam_min = cam.amin(dim=(2, 3), keepdim=True)
        cam_max = cam.amax(dim=(2, 3), keepdim=True)
        normalized = (cam - cam_min) / (cam_max - cam_min + 1e-8)
        return GradCamResult(heatmap=normalized.squeeze(1))
    finally:
        forward_handle.remove()
        backward_handle.remove()
