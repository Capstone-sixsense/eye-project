from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class BinaryFocalLoss(nn.Module):
    """Binary focal loss (Lin et al., 2017, arXiv:1708.02002).

    Down-weights easy, well-classified examples and concentrates training
    on hard, misclassified ones. Useful when the model is over-confident
    on in-domain examples while under-confident on shifted domains.

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Args:
        gamma: Focusing parameter. gamma=0 reduces to weighted BCE.
               Typical range: 0.5 -- 5.0. Default: 2.0.
        alpha: Weight for the positive (abnormal) class, in [0, 1].
               The negative class receives weight (1 - alpha).
               None disables per-class weighting.
    """

    def __init__(self, gamma: float = 2.0, alpha: float | None = None) -> None:
        super().__init__()
        if gamma < 0:
            raise ValueError(f"gamma must be >= 0, got {gamma}")
        if alpha is not None and not (0.0 <= alpha <= 1.0):
            raise ValueError(f"alpha must be in [0, 1], got {alpha}")
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        probs = torch.sigmoid(logits)
        p_t = probs * targets + (1.0 - probs) * (1.0 - targets)
        focal_weight = (1.0 - p_t) ** self.gamma

        if self.alpha is not None:
            alpha_t = self.alpha * targets + (1.0 - self.alpha) * (1.0 - targets)
            focal_weight = alpha_t * focal_weight

        return (focal_weight * bce).mean()


class DiceBCELoss(nn.Module):
    """Dice + BCE combined loss for binary segmentation.

    Dice loss handles severe class imbalance (lesion pixels << background).
    BCE provides stable per-pixel gradients early in training.

    L = (1 - w) * BCE + w * Dice

    Args:
        dice_weight: Weight for Dice term; BCE weight = 1 - dice_weight. Default 0.5.
        smooth: Laplace smoothing to avoid division by zero. Default 1.0.
    """

    def __init__(self, dice_weight: float = 0.5, smooth: float = 1.0) -> None:
        super().__init__()
        if not (0.0 <= dice_weight <= 1.0):
            raise ValueError(f"dice_weight must be in [0, 1], got {dice_weight}")
        self.dice_weight = dice_weight
        self.bce_weight = 1.0 - dice_weight
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(logits, targets)

        probs = torch.sigmoid(logits).view(-1)
        flat_targets = targets.view(-1)
        intersection = (probs * flat_targets).sum()
        dice = 1.0 - (2.0 * intersection + self.smooth) / (
            probs.sum() + flat_targets.sum() + self.smooth
        )

        return self.bce_weight * bce + self.dice_weight * dice


class FocalTverskyBCELoss(nn.Module):
    """Focal Tversky + BCE loss for sparse lesion segmentation.

    The Tversky term can penalize false-positive whole-image masks more
    aggressively than Dice while still keeping gradients on tiny lesion pixels.
    BCE is retained as a small stabilizer for early training.
    """

    def __init__(
        self,
        alpha: float = 0.7,
        beta: float = 0.3,
        gamma: float = 4.0 / 3.0,
        bce_weight: float = 0.2,
        smooth: float = 1.0,
    ) -> None:
        super().__init__()
        if alpha < 0 or beta < 0:
            raise ValueError("alpha and beta must be non-negative")
        if gamma <= 0:
            raise ValueError(f"gamma must be > 0, got {gamma}")
        if not (0.0 <= bce_weight <= 1.0):
            raise ValueError(f"bce_weight must be in [0, 1], got {bce_weight}")
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.bce_weight = bce_weight
        self.tversky_weight = 1.0 - bce_weight
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        dims = (0, 2, 3)
        targets = targets.float()
        tp = (probs * targets).sum(dim=dims)
        fp = (probs * (1.0 - targets)).sum(dim=dims)
        fn = ((1.0 - probs) * targets).sum(dim=dims)
        tversky = (tp + self.smooth) / (
            tp + self.alpha * fp + self.beta * fn + self.smooth
        )
        focal_tversky = torch.pow(1.0 - tversky, self.gamma).mean()
        bce = F.binary_cross_entropy_with_logits(logits, targets)
        return self.tversky_weight * focal_tversky + self.bce_weight * bce


class CamAlignmentLoss(nn.Module):
    """Cosine-similarity loss between a Layer-CAM map and a lesion mask.

    Goal (Phase 4 plan §3 B2): make the classifier's attribution on the same
    intermediate feature used for Grad-CAM/Layer-CAM agree spatially with the
    lesion mask, without forcing the classifier to use that path for pooling
    (gated_pooling-style routing produced DDR regressions in v33–v35).

    Inputs are an activation tensor [B, C, H, W] (block4 feature) and the
    score gradient w.r.t. that activation (same shape, ``create_graph=True``
    when computed by the caller). The CAM is constructed as
    ``ReLU(sum_c ReLU(grad_c) * act_c)`` (Layer-CAM) and compared against the
    union mask via cosine similarity per sample. Cosine is preferred over MSE
    because lesion masks are sparse: MSE on the background dominates while
    cosine measures shape alignment regardless of magnitude.

    Args:
        eps: numerical floor for norms.
    """

    def __init__(self, eps: float = 1e-8) -> None:
        super().__init__()
        self.eps = eps

    def forward(
        self,
        activation: torch.Tensor,
        gradient: torch.Tensor,
        mask: torch.Tensor,
        valid: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if activation.shape != gradient.shape:
            raise ValueError(
                f"activation/gradient shape mismatch: {activation.shape} vs {gradient.shape}"
            )

        cam = F.relu((F.relu(gradient) * activation).sum(dim=1, keepdim=True))
        if cam.shape[-2:] != mask.shape[-2:]:
            cam = F.interpolate(
                cam, size=mask.shape[-2:], mode="bilinear", align_corners=False
            )

        if mask.shape[1] > 1:
            mask_union = mask.amax(dim=1, keepdim=True)
        else:
            mask_union = mask

        if valid is not None:
            valid_idx = valid.view(-1).bool()
            if not valid_idx.any():
                return cam.new_zeros(())
            cam = cam[valid_idx]
            mask_union = mask_union[valid_idx]

        b = cam.shape[0]
        cam_flat = cam.reshape(b, -1)
        mask_flat = mask_union.reshape(b, -1).float()

        cam_norm = cam_flat.norm(dim=1).clamp_min(self.eps)
        mask_norm = mask_flat.norm(dim=1).clamp_min(self.eps)
        per_sample_keep = mask_flat.sum(dim=1) > 0
        if not per_sample_keep.any():
            return cam.new_zeros(())

        cos = (cam_flat * mask_flat).sum(dim=1) / (cam_norm * mask_norm)
        cos = cos[per_sample_keep]
        return (1.0 - cos).mean()


class CoralLoss(nn.Module):
    """Deep CORAL: Correlation Alignment (Sun & Saenko, 2016, arXiv:1607.01719).

    Minimises the squared Frobenius-norm difference between the unbiased sample
    covariance matrices of two domain feature sets.  Applied on pooled
    pre-classifier features so it acts directly on the representation space.

    L_CORAL = (1 / 4d²) * ||C_source − C_target||²_F

    Args:
        n < 2 per domain: returns zero loss (no covariance defined).
    """

    @staticmethod
    def _covariance(x: torch.Tensor) -> torch.Tensor:
        n, d = x.shape
        if n < 2:
            return x.new_zeros(d, d)
        xc = x - x.mean(0, keepdim=True)
        return (xc.T @ xc) / (n - 1)

    def forward(self, source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        d = source.size(1)
        cs = self._covariance(source.float())
        ct = self._covariance(target.float())
        return torch.norm(cs - ct, p="fro") ** 2 / (4 * d * d)
