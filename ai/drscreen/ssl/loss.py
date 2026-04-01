from __future__ import annotations

import torch
import torch.nn.functional as F


def nt_xent_loss(z1: torch.Tensor, z2: torch.Tensor, temperature: float = 0.07) -> torch.Tensor:
    """NT-Xent contrastive loss from SimCLR (Chen et al., 2020).

    z1, z2: (N, D) projected representations of two views of the same image.
    Positive pairs are (z1[i], z2[i]). All other pairs in the batch are negatives.
    """
    # Cast to float32 unconditionally: the similarity matrix divided by a small
    # temperature (e.g. 0.07) can overflow FP16 (max ~65504) before softmax.
    batch_size = z1.shape[0]
    z = F.normalize(torch.cat([z1.float(), z2.float()], dim=0), dim=1)  # (2N, D)

    sim = torch.mm(z, z.T) / temperature  # (2N, 2N)

    # Exclude self-similarity from softmax denominator.
    sim.fill_diagonal_(float("-inf"))

    # Positive pair indices: row i pairs with i+N, row i+N pairs with i.
    labels = torch.cat(
        [torch.arange(batch_size, 2 * batch_size), torch.arange(batch_size)],
        dim=0,
    ).to(z.device)

    return F.cross_entropy(sim, labels)
