from __future__ import annotations

import argparse

import numpy as np
import torch
import torch.nn.functional as F

PERTURBATION_METHODS = {"occlusion", "rise"}


def _normalize_map(arr: np.ndarray) -> np.ndarray:
    arr = arr.astype(np.float32, copy=False)
    mn = float(np.min(arr))
    mx = float(np.max(arr))
    if mx <= mn + 1e-8:
        return np.zeros_like(arr, dtype=np.float32)
    return ((arr - mn) / (mx - mn)).astype(np.float32)


def _infer_class_index(outputs: torch.Tensor, class_index: int | None) -> int:
    if class_index is not None:
        return int(class_index)
    if outputs.ndim != 2:
        raise ValueError("Expected logits shape [batch, classes or 1].")
    if outputs.shape[1] == 1:
        return 0
    return int(outputs.argmax(dim=1).item())


def _target_scores(
    outputs: torch.Tensor,
    class_index: int,
) -> torch.Tensor:
    if outputs.ndim != 2:
        raise ValueError("Expected logits shape [batch, classes or 1].")
    if outputs.shape[1] == 1:
        return torch.sigmoid(outputs[:, 0])
    return torch.softmax(outputs, dim=1)[:, class_index]


def _batched_scores(
    model: torch.nn.Module,
    inputs: torch.Tensor,
    class_index: int,
    batch_size: int,
) -> torch.Tensor:
    scores: list[torch.Tensor] = []
    with torch.no_grad():
        for start in range(0, inputs.shape[0], batch_size):
            out = model(inputs[start:start + batch_size])
            scores.append(_target_scores(out, class_index).detach())
    return torch.cat(scores, dim=0)


def occlusion_attribution(
    model: torch.nn.Module,
    inputs: torch.Tensor,
    grid_size: int = 16,
    patch_value: float = 0.0,
    class_index: int | None = None,
    batch_size: int = 16,
) -> np.ndarray:
    """Return deterministic grid occlusion attribution for one image.

    ``grid_size=16`` means 16x16 cells, so the method costs 256 additional
    forward passes per image before batching. The score assigned to each cell is
    the positive drop in target-class probability when that cell is replaced by
    ``patch_value`` in normalized input space.
    """
    if inputs.ndim != 4 or inputs.shape[0] != 1:
        raise ValueError("occlusion_attribution expects inputs with shape [1,C,H,W]")
    if grid_size <= 0:
        raise ValueError("grid_size must be positive")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            base_outputs = model(inputs)
            target = _infer_class_index(base_outputs, class_index)
            base_score = _target_scores(base_outputs, target)[0]

        _b, _c, h, w = inputs.shape
        y_edges = torch.linspace(0, h, grid_size + 1, dtype=torch.int64)
        x_edges = torch.linspace(0, w, grid_size + 1, dtype=torch.int64)

        occluded: list[torch.Tensor] = []
        coords: list[tuple[int, int, int, int]] = []
        for yi in range(grid_size):
            y0, y1 = int(y_edges[yi].item()), int(y_edges[yi + 1].item())
            for xi in range(grid_size):
                x0, x1 = int(x_edges[xi].item()), int(x_edges[xi + 1].item())
                sample = inputs.clone()
                sample[..., y0:y1, x0:x1] = patch_value
                occluded.append(sample)
                coords.append((y0, y1, x0, x1))

        batch = torch.cat(occluded, dim=0)
        scores = _batched_scores(model, batch, target, batch_size=batch_size)
        drops = torch.relu(base_score - scores).detach().cpu().numpy()

        attr = np.zeros((h, w), dtype=np.float32)
        for drop, (y0, y1, x0, x1) in zip(drops, coords):
            attr[y0:y1, x0:x1] = float(drop)
        return _normalize_map(attr)
    finally:
        if was_training:
            model.train()


def rise_attribution(
    model: torch.nn.Module,
    inputs: torch.Tensor,
    num_masks: int = 4000,
    mask_resolution: int = 7,
    keep_prob: float = 0.5,
    class_index: int | None = None,
    batch_size: int = 32,
    seed: int = 0,
) -> np.ndarray:
    """Return RISE attribution for one image.

    Random low-resolution masks are upsampled to input resolution and weighted
    by the target-class probability of the masked image. This is suitable for
    evaluation and faithfulness diagnostics, not production latency.
    """
    if inputs.ndim != 4 or inputs.shape[0] != 1:
        raise ValueError("rise_attribution expects inputs with shape [1,C,H,W]")
    if num_masks <= 0:
        raise ValueError("num_masks must be positive")
    if mask_resolution <= 0:
        raise ValueError("mask_resolution must be positive")
    if not (0.0 < keep_prob < 1.0):
        raise ValueError("keep_prob must be in (0, 1)")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            base_outputs = model(inputs)
            target = _infer_class_index(base_outputs, class_index)

        _b, _c, h, w = inputs.shape
        generator = torch.Generator(device=inputs.device)
        generator.manual_seed(seed)
        low_res = torch.rand(
            num_masks, 1, mask_resolution, mask_resolution,
            generator=generator,
            device=inputs.device,
        )
        low_res = (low_res < keep_prob).float()
        masks = F.interpolate(low_res, size=(h, w), mode="bilinear", align_corners=False)

        scores: list[torch.Tensor] = []
        weighted = torch.zeros(h, w, device=inputs.device, dtype=torch.float32)
        with torch.no_grad():
            for start in range(0, num_masks, batch_size):
                batch_masks = masks[start:start + batch_size]
                batch_inputs = inputs * batch_masks
                out = model(batch_inputs)
                batch_scores = _target_scores(out, target).detach()
                scores.append(batch_scores)
                weighted += (batch_scores.view(-1, 1, 1) * batch_masks[:, 0]).sum(dim=0)

        attr = weighted / (num_masks * keep_prob)
        return _normalize_map(attr.detach().cpu().numpy())
    finally:
        if was_training:
            model.train()


def _self_test() -> None:
    class Dummy(torch.nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x.mean(dim=(1, 2, 3), keepdim=True).flatten(1)

    model = Dummy()
    x = torch.ones(1, 3, 64, 64)
    occ = occlusion_attribution(model, x, grid_size=4, batch_size=4)
    rise = rise_attribution(model, x, num_masks=8, mask_resolution=4, batch_size=4)
    assert occ.shape == (64, 64)
    assert rise.shape == (64, 64)
    assert np.isfinite(occ).all()
    assert np.isfinite(rise).all()
    print("perturbation self-test passed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        _self_test()
