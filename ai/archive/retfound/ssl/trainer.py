from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from drscreen.models.build import build_model, load_retfound_backbone
from drscreen.settings import resolve_project_path
from drscreen.ssl.dataset import SSLDataset
from drscreen.ssl.simclr import NTXentLoss, SimCLRModel
from drscreen.utils.logging import get_logger
from drscreen.utils.seed import set_seed


LOGGER = get_logger(__name__)


def _freeze_vit_lower_blocks(encoder: torch.nn.Module, n: int) -> None:
    """Freeze the first n transformer blocks of a timm ViT encoder plus the
    patch embedding and positional embedding.

    Frozen parameters are excluded from the backward graph, which reduces
    per-step compute and memory proportionally to the frozen fraction.

    Args:
        encoder: timm ViT model (e.g. vit_large_patch16_224).
        n: Number of transformer blocks to freeze (from the bottom).
    """
    for param in encoder.patch_embed.parameters():
        param.requires_grad = False
    if hasattr(encoder, "cls_token"):
        encoder.cls_token.requires_grad = False
    if hasattr(encoder, "pos_embed"):
        encoder.pos_embed.requires_grad = False

    blocks = list(encoder.blocks)
    for block in blocks[:n]:
        for param in block.parameters():
            param.requires_grad = False


def _build_ssl_transform(image_size: int) -> transforms.Compose:
    """SimCLR augmentation pipeline for fundus images.

    Aggressive spatial and photometric augmentation encourages the model
    to learn domain-invariant representations. Grayscale is excluded
    because color is diagnostically relevant in fundus images (hemorrhages,
    hard exudates). Gaussian blur probability is kept at 0.5 following the
    original SimCLR paper.
    """
    blur_kernel = image_size // 10 * 2 + 1  # odd kernel, ~10% of image size
    return transforms.Compose([
        transforms.RandomResizedCrop(
            image_size,
            scale=(0.2, 1.0),
            interpolation=transforms.InterpolationMode.BICUBIC,
        ),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomApply([
            transforms.ColorJitter(
                brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1
            ),
        ], p=0.8),
        transforms.RandomApply([
            transforms.GaussianBlur(kernel_size=blur_kernel, sigma=(0.1, 2.0)),
        ], p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def run_ssl_pretraining(
    config: dict[str, Any],
    *,
    config_path: Path,
    project_root: Path,
) -> dict[str, Any]:
    """SimCLR pretraining on all manifest images (all splits, all domains).

    Uses the full manifest (train + val + test + external_test) so the
    encoder is exposed to the Messidor domain before supervised fine-tuning.
    Saves the encoder in RETFound MAE checkpoint format so it can be loaded
    by load_retfound_backbone() without modification.
    """
    ssl_cfg = config["ssl"]
    train_cfg = config["train"]

    set_seed(int(train_cfg.get("seed", 42)))
    device_str = str(train_cfg.get("device", "cuda"))
    device = torch.device(
        "cuda" if device_str.startswith("cuda") and torch.cuda.is_available() else "cpu"
    )

    image_size = int(config["data"]["image_size"])
    ssl_transform = _build_ssl_transform(image_size)

    manifest_path = resolve_project_path(project_root, config["data"]["manifest_path"])
    image_root = resolve_project_path(project_root, config["data"]["image_root"])
    dataset = SSLDataset(manifest_path, image_root, ssl_transform)
    LOGGER.info("SSL dataset: %d images (all splits, all domains)", len(dataset))

    batch_size = int(ssl_cfg.get("batch_size", 32))
    num_workers = int(config["data"].get("num_workers", 4))
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        drop_last=True,
    )

    # Optionally subsample the dataset to cap total SSL compute.
    max_samples = int(ssl_cfg.get("max_samples", 0))
    if max_samples and max_samples < len(dataset):
        import random as _random
        _random.seed(int(train_cfg.get("seed", 42)))
        indices = _random.sample(range(len(dataset)), max_samples)
        dataset = torch.utils.data.Subset(dataset, indices)
        LOGGER.info("SSL dataset subsampled to %d images (max_samples=%d)", len(dataset), max_samples)

    # Build encoder from RETFound MAE weights.
    encoder = build_model("retfound", pretrained=False, num_outputs=1).to(device)
    retfound_path_str = str(train_cfg.get("pretrained_backbone_path", "")).strip()
    if retfound_path_str:
        retfound_path = resolve_project_path(project_root, retfound_path_str)
        if retfound_path.exists():
            missing, unexpected = load_retfound_backbone(encoder, retfound_path)
            LOGGER.info(
                "Loaded RETFound MAE backbone from %s (missing=%d, unexpected=%d)",
                retfound_path,
                len(missing),
                len(unexpected),
            )
        else:
            LOGGER.warning("pretrained_backbone_path not found: %s", retfound_path)

    # Freeze lower ViT blocks to speed up SSL.
    # RETFound already has strong retinal representations; only the top blocks
    # need to adapt to the multi-domain mixture. Freezing 20/24 blocks reduces
    # per-step compute ~5x and allows convergence in far fewer epochs.
    frozen_blocks = int(ssl_cfg.get("frozen_blocks", 0))
    if frozen_blocks > 0:
        _freeze_vit_lower_blocks(encoder, frozen_blocks)
        trainable = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
        total = sum(p.numel() for p in encoder.parameters())
        LOGGER.info(
            "Froze first %d ViT blocks — trainable encoder params: %d / %d (%.1f%%)",
            frozen_blocks, trainable, total, 100.0 * trainable / max(total, 1),
        )

    feature_dim = int(ssl_cfg.get("feature_dim", 1024))
    proj_hidden = int(ssl_cfg.get("proj_hidden_dim", 2048))
    proj_out = int(ssl_cfg.get("proj_out_dim", 128))
    model = SimCLRModel(
        encoder, feature_dim=feature_dim, proj_hidden=proj_hidden, proj_out=proj_out
    ).to(device)

    temperature = float(ssl_cfg.get("temperature", 0.07))
    criterion = NTXentLoss(temperature=temperature)

    lr = float(ssl_cfg.get("learning_rate", 3e-4))
    weight_decay = float(ssl_cfg.get("weight_decay", 1e-4))
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=weight_decay)

    epochs = int(ssl_cfg.get("epochs", 30))
    amp_enabled = bool(train_cfg.get("amp", True)) and device.type == "cuda"

    # Cosine LR decay over the full SSL run prevents large weight updates in
    # later epochs, which is the main cause of feature drift / SSL overfitting.
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # Early stopping on training loss plateau.
    # Contrastive loss still decreasing does not guarantee downstream improvement;
    # stopping when loss plateaus avoids overfitting the SSL objective.
    es_patience = int(ssl_cfg.get("early_stopping_patience", 0))
    es_min_delta = float(ssl_cfg.get("early_stopping_min_delta", 0.005))
    best_loss = float("inf")
    no_improve = 0

    history: list[dict[str, Any]] = []
    stopped_epoch = epochs
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        n_batches = 0
        for view1, view2 in loader:
            view1 = view1.to(device)
            view2 = view2.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=amp_enabled):
                z1, z2 = model.forward_pair(view1, view2)
                loss = criterion(z1, z2)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach().item())
            n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)
        scheduler.step()
        history.append({"epoch": epoch, "loss": avg_loss})
        LOGGER.info("SSL epoch=%d/%d  avg_loss=%.4f", epoch, epochs, avg_loss)

        if avg_loss < best_loss - es_min_delta:
            best_loss = avg_loss
            no_improve = 0
        else:
            no_improve += 1

        if es_patience > 0 and no_improve >= es_patience:
            LOGGER.info(
                "SSL early stopping at epoch %d — no improvement > %.4f for %d epochs",
                epoch, es_min_delta, es_patience,
            )
            stopped_epoch = epoch
            break

    # Save encoder weights in {"model": state_dict} format so
    # load_retfound_backbone() can load this checkpoint directly.
    output_path = resolve_project_path(
        project_root, str(ssl_cfg.get("output_path", "artifacts/checkpoints/ssl/retfound_simclr.pth"))
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.encoder_state_dict()}, output_path)
    LOGGER.info("SSL encoder saved to %s", output_path)

    summary: dict[str, Any] = {
        "config_path": str(config_path),
        "manifest_path": str(manifest_path),
        "num_ssl_images": len(dataset),
        "epochs": stopped_epoch,
        "batch_size": batch_size,
        "temperature": temperature,
        "output_path": str(output_path),
        "history": history,
    }
    summary_path = output_path.parent / "ssl_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    summary["summary_path"] = str(summary_path)
    return summary
