from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from drscreen.xai.perturbation import _infer_class_index, _target_scores


def _resize_attribution(
    attribution: np.ndarray,
    size: tuple[int, int],
    device: torch.device,
) -> torch.Tensor:
    attr = torch.as_tensor(attribution, dtype=torch.float32, device=device)
    if attr.ndim != 2:
        raise ValueError("attribution must have shape [H,W]")
    attr = attr.unsqueeze(0).unsqueeze(0)
    if attr.shape[-2:] != size:
        attr = F.interpolate(attr, size=size, mode="bilinear", align_corners=False)
    return attr[0, 0]


def _score_curve(
    model: torch.nn.Module,
    inputs: torch.Tensor,
    attribution: np.ndarray,
    n_steps: int,
    class_index: int | None,
    mode: str,
    batch_size: int,
) -> list[float]:
    if inputs.ndim != 4 or inputs.shape[0] != 1:
        raise ValueError("faithfulness expects inputs with shape [1,C,H,W]")
    if n_steps <= 0:
        raise ValueError("n_steps must be positive")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if mode not in {"deletion", "insertion"}:
        raise ValueError(f"unknown faithfulness mode: {mode!r}")

    _b, _c, h, w = inputs.shape
    attr = _resize_attribution(attribution, (h, w), inputs.device)
    order = torch.argsort(attr.flatten(), descending=True)
    total = order.numel()
    baseline = torch.zeros_like(inputs)

    with torch.no_grad():
        base_outputs = model(inputs)
        target = _infer_class_index(base_outputs, class_index)

    samples: list[torch.Tensor] = []
    for step in range(n_steps + 1):
        k = int(round(total * step / n_steps))
        if mode == "deletion":
            sample = inputs.clone()
            if k > 0:
                flat = sample.view(1, sample.shape[1], -1)
                base_flat = baseline.view(1, baseline.shape[1], -1)
                flat[:, :, order[:k]] = base_flat[:, :, order[:k]]
        else:
            sample = baseline.clone()
            if k > 0:
                flat = sample.view(1, sample.shape[1], -1)
                src_flat = inputs.view(1, inputs.shape[1], -1)
                flat[:, :, order[:k]] = src_flat[:, :, order[:k]]
        samples.append(sample)

    scores: list[torch.Tensor] = []
    with torch.no_grad():
        for start in range(0, len(samples), batch_size):
            batch = torch.cat(samples[start:start + batch_size], dim=0)
            out = model(batch)
            scores.append(_target_scores(out, target).detach().cpu())
    return [float(v) for v in torch.cat(scores, dim=0)]


def deletion_auc(
    model: torch.nn.Module,
    inputs: torch.Tensor,
    attribution: np.ndarray,
    n_steps: int = 100,
    class_index: int | None = None,
    batch_size: int = 16,
) -> float:
    """AUC of probability while removing most-attributed pixels first.

    Lower is better: confidence should drop quickly when important pixels are
    removed.
    """
    curve = _score_curve(
        model, inputs, attribution, n_steps, class_index, "deletion", batch_size
    )
    xs = np.linspace(0.0, 1.0, len(curve), dtype=np.float32)
    return float(np.trapz(np.asarray(curve, dtype=np.float32), xs))


def insertion_auc(
    model: torch.nn.Module,
    inputs: torch.Tensor,
    attribution: np.ndarray,
    n_steps: int = 100,
    class_index: int | None = None,
    batch_size: int = 16,
) -> float:
    """AUC of probability while inserting most-attributed pixels first.

    Higher is better: confidence should recover quickly as important pixels are
    inserted into a blank baseline image.
    """
    curve = _score_curve(
        model, inputs, attribution, n_steps, class_index, "insertion", batch_size
    )
    xs = np.linspace(0.0, 1.0, len(curve), dtype=np.float32)
    return float(np.trapz(np.asarray(curve, dtype=np.float32), xs))


def faithfulness_auc(
    model: torch.nn.Module,
    inputs: torch.Tensor,
    attribution: np.ndarray,
    n_steps: int = 100,
    class_index: int | None = None,
    batch_size: int = 16,
) -> dict[str, float]:
    deletion = deletion_auc(
        model, inputs, attribution, n_steps=n_steps,
        class_index=class_index, batch_size=batch_size,
    )
    insertion = insertion_auc(
        model, inputs, attribution, n_steps=n_steps,
        class_index=class_index, batch_size=batch_size,
    )
    return {
        "deletion_auc": deletion,
        "insertion_auc": insertion,
        "insertion_minus_deletion": insertion - deletion,
    }
