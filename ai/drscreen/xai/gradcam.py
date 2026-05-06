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


def _normalize_cam(cam: torch.Tensor) -> torch.Tensor:
    """Spatial min-max normalization per image in the batch."""
    cam_min = cam.amin(dim=(2, 3), keepdim=True)
    cam_max = cam.amax(dim=(2, 3), keepdim=True)
    return (cam - cam_min) / (cam_max - cam_min + 1e-8)


def _gradcam_core(
    model: torch.nn.Module,
    inputs: torch.Tensor,
    layer: torch.nn.Module,
    class_index: int | None,
    method: str,
) -> GradCamResult:
    """Shared forward+backward pass for Grad-CAM and Grad-CAM++."""
    activations: dict = {}
    gradients: dict = {}

    fwd = layer.register_forward_hook(lambda m, i, o: activations.update({"v": o.detach()}))
    bwd = layer.register_full_backward_hook(
        lambda m, gi, go: gradients.update({"v": go[0].detach()})
    )
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
    finally:
        fwd.remove()
        bwd.remove()

    act = activations["v"]   # [B, C, H, W]
    grad = gradients["v"]    # [B, C, H, W]

    if method == "layercam":
        weights = torch.relu(grad)
        cam = torch.relu((weights * act).sum(dim=1, keepdim=True))
    elif method == "gradcam++":
        grad_sq = grad ** 2
        grad_cu = grad ** 3
        alpha = grad_sq / (
            2 * grad_sq + (act * grad_cu).sum(dim=(2, 3), keepdim=True) + 1e-8
        )
        weights = (alpha * F.relu(grad)).sum(dim=(2, 3), keepdim=True)
        cam = F.relu((weights * act).sum(dim=1, keepdim=True))
    else:  # gradcam
        weights = grad.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((weights * act).sum(dim=1, keepdim=True))

    cam = F.interpolate(cam, size=inputs.shape[-2:], mode="bilinear", align_corners=False)
    return GradCamResult(heatmap=_normalize_cam(cam).squeeze(1))


def _score_cam(
    model: torch.nn.Module,
    inputs: torch.Tensor,
    layer: torch.nn.Module,
    class_index: int | None,
    top_k: int = 64,
) -> GradCamResult:
    """Score-CAM (Wang et al., CVPR Workshop 2020).

    Avoids gradients entirely: each channel's upsampled activation mask is used
    to perturb the input; the resulting score change is the channel weight.
    Limited to top_k channels by activation L1-norm to stay tractable.
    """
    activations: dict = {}
    fwd = layer.register_forward_hook(lambda m, i, o: activations.update({"v": o.detach()}))
    with torch.no_grad():
        base_outputs = model(inputs)
    fwd.remove()

    act = activations["v"]  # [1, C, h, w]
    C = act.shape[1]

    if class_index is None:
        if base_outputs.shape[1] == 1:
            class_index = 0
        else:
            class_index = int(base_outputs.argmax(dim=1).item())

    # Select top-k channels by activation magnitude
    channel_norms = act[0].abs().mean(dim=(1, 2))  # [C]
    k = min(top_k, C)
    top_indices = channel_norms.topk(k).indices

    # Baseline: zero input
    baseline = torch.zeros_like(inputs)
    with torch.no_grad():
        base_score = torch.sigmoid(model(baseline)[:, class_index]).item() if base_outputs.shape[1] == 1 \
            else torch.softmax(model(baseline), dim=1)[:, class_index].item()

    scores = torch.zeros(k, device=inputs.device)
    H, W = inputs.shape[-2:]

    with torch.no_grad():
        for i, c in enumerate(top_indices):
            channel_map = act[0, c:c+1, :, :]  # [1, h, w]
            mask = F.interpolate(channel_map.unsqueeze(0), size=(H, W), mode="bilinear", align_corners=False)
            mn, mx = mask.amin(), mask.amax()
            mask = (mask - mn) / (mx - mn + 1e-8)
            masked_input = inputs * mask
            out = model(masked_input)
            if out.shape[1] == 1:
                s = torch.sigmoid(out[:, 0]).item()
            else:
                s = torch.softmax(out, dim=1)[:, class_index].item()
            scores[i] = s - base_score

    # Build CAM: weighted sum of selected activation channels
    weights = F.relu(scores)  # only positive contributions
    cam = torch.zeros(1, 1, act.shape[2], act.shape[3], device=inputs.device)
    for i, c in enumerate(top_indices):
        cam += weights[i] * act[0, c:c+1, :, :].unsqueeze(0)

    cam = F.interpolate(cam, size=(H, W), mode="bilinear", align_corners=False)
    return GradCamResult(heatmap=_normalize_cam(cam).squeeze(1))


def _integrated_gradients(
    model: torch.nn.Module,
    inputs: torch.Tensor,
    class_index: int | None,
    n_steps: int = 50,
) -> GradCamResult:
    """Integrated Gradients (Sundararajan et al., ICML 2017).

    Pixel-level attribution: (input - baseline) * mean_gradient over n_steps
    interpolated inputs. Baseline = zero (black image).
    Output is averaged across channels to produce a single heatmap.
    """
    baseline = torch.zeros_like(inputs)

    # Determine target class from a forward pass
    with torch.no_grad():
        outputs = model(inputs)
    if class_index is None:
        class_index = 0 if outputs.shape[1] == 1 else int(outputs.argmax(dim=1).item())

    alphas = torch.linspace(0, 1, n_steps, device=inputs.device)
    integrated_grads = torch.zeros_like(inputs)

    for alpha in alphas:
        interpolated = (baseline + alpha * (inputs - baseline)).requires_grad_(True)
        out = model(interpolated)
        score = out[:, class_index].sum()
        model.zero_grad(set_to_none=True)
        score.backward()
        if interpolated.grad is not None:
            integrated_grads += interpolated.grad.detach()

    ig = (inputs - baseline) * integrated_grads / n_steps  # [1, 3, H, W]
    # Average absolute attribution across colour channels
    cam = ig.abs().mean(dim=1, keepdim=True)  # [1, 1, H, W]
    return GradCamResult(heatmap=_normalize_cam(cam).squeeze(1))


def generate_gradcam(
    model: torch.nn.Module,
    inputs: torch.Tensor,
    target_layer: torch.nn.Module | None = None,
    target_layer_name: str | None = None,
    class_index: int | None = None,
    method: str = "gradcam",
    score_cam_top_k: int = 64,
    ig_n_steps: int = 50,
) -> GradCamResult:
    """Dispatch to the requested XAI method.

    Supported methods
    -----------------
    gradcam        : Grad-CAM (Selvaraju et al., ICCV 2017)
    layercam       : Layer-CAM (Jiang et al., IEEE TIP 2021)
    gradcam++      : Grad-CAM++ (Chattopadhay et al., WACV 2018)
    scorecam       : Score-CAM (Wang et al., CVPR Workshop 2020)
    ig             : Integrated Gradients (Sundararajan et al., ICML 2017)
    """
    if method == "ig":
        return _integrated_gradients(model, inputs, class_index, n_steps=ig_n_steps)

    if target_layer is not None:
        layer = target_layer
    elif target_layer_name:
        layer = model.get_submodule(target_layer_name)
    else:
        layer = resolve_default_target_layer(model)

    if method == "scorecam":
        return _score_cam(model, inputs, layer, class_index, top_k=score_cam_top_k)

    return _gradcam_core(model, inputs, layer, class_index, method)
