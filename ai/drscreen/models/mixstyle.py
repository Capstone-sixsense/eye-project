from __future__ import annotations

import torch
import torch.nn as nn


class MixStyle(nn.Module):
    """MixStyle (Zhou et al., ICLR 2021) for domain generalization.

    Probabilistically mixes instance-level feature statistics (mean, std)
    across samples within a batch during training. This simulates domain-level
    style variation and encourages domain-invariant representations.

    Active only during training; identity during eval.

    Statistics are computed in FP32 regardless of input dtype to prevent
    precision issues under AMP (BF16/FP16).

    Args:
        p: Probability of applying per forward call.
        alpha: Beta distribution concentration parameter.
        eps: Small constant for numerical stability in variance computation.
    """

    def __init__(self, p: float = 0.5, alpha: float = 0.1, eps: float = 1e-6) -> None:
        super().__init__()
        self.p = p
        self.alpha = alpha
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or x.size(0) <= 1:
            return x

        if torch.rand(1).item() > self.p:
            return x

        orig_dtype = x.dtype
        x_fp32 = x.float()

        B = x_fp32.size(0)
        spatial_dims = tuple(range(2, x_fp32.dim()))

        mu = x_fp32.mean(dim=spatial_dims, keepdim=True)
        sig = (x_fp32.var(dim=spatial_dims, keepdim=True) + self.eps).sqrt()

        x_normed = (x_fp32 - mu) / sig

        perm = torch.randperm(B, device=x.device)
        lam = torch.distributions.Beta(self.alpha, self.alpha).sample(
            (B,) + (1,) * (x_fp32.dim() - 1),
        ).to(device=x.device)

        mu_mix = lam * mu + (1 - lam) * mu[perm]
        sig_mix = lam * sig + (1 - lam) * sig[perm]

        return (x_normed * sig_mix + mu_mix).to(orig_dtype)
