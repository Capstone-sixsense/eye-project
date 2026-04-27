from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class _ProjectionHead(nn.Module):
    """2-layer MLP projection head with BN and ReLU (SimCLR-v2 style)."""

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim, bias=False),
            nn.BatchNorm1d(out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SimCLRModel(nn.Module):
    """RETFound encoder wrapped with a SimCLR projection head.

    The encoder is expected to be a timm ViT-Large/16 instance.
    forward_features() is used to extract the CLS token, bypassing the
    classification head so state_dict keys remain identical to the base
    ViT-Large model (no extra prefix). This makes saving and reloading
    the encoder straightforward.

    Args:
        encoder: timm ViT-Large/16 model (from build_model("retfound")).
        feature_dim: CLS token dimension (1024 for ViT-Large).
        proj_hidden: Projection head hidden dimension.
        proj_out: Projection head output dimension (embedding space).
    """

    def __init__(
        self,
        encoder: nn.Module,
        feature_dim: int = 1024,
        proj_hidden: int = 2048,
        proj_out: int = 128,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.projector = _ProjectionHead(feature_dim, proj_hidden, proj_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # CLS token: forward_features returns (B, seq_len, D); index 0 is CLS.
        features = self.encoder.forward_features(x)[:, 0]
        return self.projector(features)

    def forward_pair(
        self, x1: torch.Tensor, x2: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Process two augmented views in a single encoder forward pass.

        Concatenates x1 and x2 along the batch dimension so the encoder runs
        once instead of twice per training step. Halves the number of ViT
        kernel launches and improves GPU utilization.
        """
        x = torch.cat([x1, x2], dim=0)
        features = self.encoder.forward_features(x)[:, 0]
        z = self.projector(features)
        z1, z2 = z.chunk(2, dim=0)
        return z1, z2

    def encoder_state_dict(self) -> dict[str, torch.Tensor]:
        """Return encoder weights only, keyed without 'encoder.' prefix.

        Saved as {"model": encoder_state_dict()} so load_retfound_backbone
        can load this checkpoint without modification.
        """
        prefix = "encoder."
        return {
            k[len(prefix):]: v
            for k, v in self.state_dict().items()
            if k.startswith(prefix)
        }


class NTXentLoss(nn.Module):
    """Normalized Temperature-scaled Cross Entropy loss (Chen et al., 2020).

    For a batch of N image pairs, constructs 2N embeddings and treats the
    two views of the same image as the positive pair. All other 2N-2
    embeddings in the batch are negatives.

    Args:
        temperature: Softmax temperature tau. Lower values sharpen the
                     distribution and produce harder negatives. Default: 0.07.
    """

    def __init__(self, temperature: float = 0.07) -> None:
        super().__init__()
        if temperature <= 0.0:
            raise ValueError(f"temperature must be > 0, got {temperature}")
        self.temperature = temperature

    def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        # Cast to float32 before similarity computation: with temperature=0.07,
        # exp(sim / 0.07) can reach exp(14.3) ≈ 1.6M which overflows FP16 (max 65504).
        z1 = F.normalize(z1.float(), dim=1)
        z2 = F.normalize(z2.float(), dim=1)
        batch_size = z1.shape[0]

        z = torch.cat([z1, z2], dim=0)  # (2N, D)
        sim = torch.mm(z, z.T) / self.temperature  # (2N, 2N)

        # Mask diagonal (self-similarity) to -inf so it never wins softmax.
        mask = torch.eye(2 * batch_size, device=z.device, dtype=torch.bool)
        sim = sim.masked_fill(mask, float("-inf"))

        # Positive index for row i is i+N, and for row i+N is i.
        labels = torch.cat([
            torch.arange(batch_size, 2 * batch_size, device=z.device),
            torch.arange(batch_size, device=z.device),
        ])
        return F.cross_entropy(sim, labels)
