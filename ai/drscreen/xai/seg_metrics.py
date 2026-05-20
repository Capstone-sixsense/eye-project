from __future__ import annotations

import numpy as np
import torch


def dice_iou_from_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    threshold: float = 0.5,
    eps: float = 1e-6,
) -> dict[str, float | None]:
    """Compute positive-channel Dice/IoU and union Dice/IoU for segmentation."""
    probs = torch.sigmoid(logits)
    pred = (probs >= threshold).float()
    targets = targets.float()

    per_channel_dice: list[float] = []
    per_channel_iou: list[float] = []
    for b in range(targets.shape[0]):
        for c in range(targets.shape[1]):
            gt = targets[b, c]
            if gt.sum() <= 0:
                continue
            pr = pred[b, c]
            inter = float((pr * gt).sum().item())
            pred_sum = float(pr.sum().item())
            gt_sum = float(gt.sum().item())
            union = pred_sum + gt_sum - inter
            per_channel_dice.append((2.0 * inter + eps) / (pred_sum + gt_sum + eps))
            per_channel_iou.append((inter + eps) / (union + eps))

    pred_union = pred.amax(dim=1)
    target_union = targets.amax(dim=1)
    union_dice: list[float] = []
    union_iou: list[float] = []
    for b in range(target_union.shape[0]):
        gt = target_union[b]
        if gt.sum() <= 0:
            continue
        pr = pred_union[b]
        inter = float((pr * gt).sum().item())
        pred_sum = float(pr.sum().item())
        gt_sum = float(gt.sum().item())
        union = pred_sum + gt_sum - inter
        union_dice.append((2.0 * inter + eps) / (pred_sum + gt_sum + eps))
        union_iou.append((inter + eps) / (union + eps))

    return {
        "mdice": float(np.mean(per_channel_dice)) if per_channel_dice else None,
        "miou": float(np.mean(per_channel_iou)) if per_channel_iou else None,
        "union_dice": float(np.mean(union_dice)) if union_dice else None,
        "union_iou": float(np.mean(union_iou)) if union_iou else None,
        "n_positive_channels": len(per_channel_dice),
        "n_positive_images": len(union_dice),
    }
