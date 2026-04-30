from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(slots=True)
class GradCamResult:
    heatmap: torch.Tensor


def generate_gradcam_plus_vit(
    model: torch.nn.Module,
    inputs: torch.Tensor,
    class_index: int | None = None,
) -> GradCamResult:
    """Grad-CAM++ for ViT-based models (e.g. RETFound).

    Hooks the last transformer block output [B, N+1, D], removes the CLS token,
    reshapes the patch token sequence into a spatial grid [B, D, H, W], then
    applies the Grad-CAM++ formula (Chattopadhay et al., 2018).

    Args:
        model: ViT model with model.blocks[-1] as the last transformer block.
        inputs: Image tensor [B, C, H, W] on the same device as model.
        class_index: Target class index. If None, uses the argmax of the output
            (or the single logit for binary classification).

    Returns:
        GradCamResult with heatmap [B, H_in, W_in] normalized to [0, 1].
    """
    layer = model.blocks[-1]

    activations: dict[str, torch.Tensor] = {}
    gradients: dict[str, torch.Tensor] = {}

    def forward_hook(_module, _input, output):
        activations["value"] = output.detach()

    def backward_hook(_module, _grad_input, grad_output):
        gradients["value"] = grad_output[0].detach()

    fh = layer.register_forward_hook(forward_hook)
    bh = layer.register_full_backward_hook(backward_hook)

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

        # act/grad: [B, N+1, D] — index 0 is CLS token, 1: are patch tokens
        act = activations["value"][:, 1:, :]   # [B, N, D]
        grad = gradients["value"][:, 1:, :]    # [B, N, D]

        # Reshape patch sequence to spatial grid [B, D, H, W]
        n_patches = act.shape[1]
        h = w = int(math.isqrt(n_patches))
        if h * w != n_patches:
            raise ValueError(
                f"Patch sequence length {n_patches} is not a perfect square. "
                "Only square grids are supported."
            )
        B, _, D = act.shape
        act = act.permute(0, 2, 1).reshape(B, D, h, w)   # [B, D, h, w]
        grad = grad.permute(0, 2, 1).reshape(B, D, h, w)  # [B, D, h, w]

        # Grad-CAM++ weights (Chattopadhay et al., 2018)
        # alpha^k = grad^2 / (2*grad^2 + sum_{spatial}(act * grad^3) + eps)
        grad2 = grad ** 2
        grad3 = grad ** 3
        spatial_sum = (act * grad3).sum(dim=(2, 3), keepdim=True)  # [B, D, 1, 1]
        alpha = grad2 / (2.0 * grad2 + spatial_sum + 1e-8)         # [B, D, h, w]

        weights = (alpha * F.relu(grad)).sum(dim=(2, 3), keepdim=True)  # [B, D, 1, 1]
        cam = F.relu((weights * act).sum(dim=1, keepdim=True))          # [B, 1, h, w]

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
        fh.remove()
        bh.remove()
