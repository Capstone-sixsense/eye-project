from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import timm
import torch
import torch.nn as nn
from timm.layers import EcaModule
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from drscreen.models.build import _inject_spatial_attention
from drscreen.settings import resolve_project_path
from drscreen.ssl.augmentations import SSLAugmentationPair
from drscreen.ssl.dataset import SSLManifestDataset
from drscreen.ssl.loss import nt_xent_loss
from drscreen.utils.logging import get_logger
from drscreen.utils.seed import set_seed


LOGGER = get_logger(__name__)

_EFFICIENTNET_B5_FEATURE_DIM = 2048


class _SimCLRModel(nn.Module):
    """EfficientNet-B5 backbone with a 2-layer MLP projection head for SimCLR."""

    def __init__(self, backbone: nn.Module, feature_dim: int, projection_dim: int) -> None:
        super().__init__()
        self.backbone = backbone
        self.projector = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.BatchNorm1d(feature_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feature_dim, projection_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.projector(self.backbone(x))


def _save_backbone(backbone: nn.Module, path: Path) -> None:
    torch.save({"model_state_dict": backbone.state_dict()}, path)


def run_ssl_pretraining(
    config: dict[str, Any],
    *,
    config_path: Path,
    project_root: Path,
) -> dict[str, Any]:
    """Run SimCLR-style contrastive SSL pretraining on all manifest images.

    Uses all rows from the manifest regardless of split so that Messidor
    external_test images participate in pretraining. Labels are not used.
    After pretraining, the backbone state dict is saved and can be loaded
    by the supervised training runner via train.pretrained_backbone_path.
    """
    ssl_cfg = config["ssl"]
    data_cfg = config["data"]
    train_cfg = config["train"]

    set_seed(int(train_cfg.get("seed", 42)))

    device_name = str(train_cfg.get("device", "cpu"))
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        LOGGER.warning("CUDA requested but not available. Falling back to CPU.")
        device_name = "cpu"
    device = torch.device(device_name)

    image_size = int(ssl_cfg.get("image_size", 224))
    mean: tuple[float, float, float] = (0.485, 0.456, 0.406)
    std: tuple[float, float, float] = (0.229, 0.224, 0.225)
    use_preprocessing = bool(ssl_cfg.get("use_preprocessing", True))
    transform = SSLAugmentationPair(image_size=image_size, mean=mean, std=std, use_preprocessing=use_preprocessing)

    manifest_path = resolve_project_path(project_root, data_cfg["manifest_path"])
    image_root = resolve_project_path(project_root, data_cfg["image_root"])
    dataset = SSLManifestDataset(
        manifest_path=manifest_path,
        image_root=image_root,
        transform=transform,
    )
    if len(dataset) == 0:
        raise ValueError("SSL dataset is empty.")

    batch_size = int(ssl_cfg.get("batch_size", 64))
    num_workers = int(data_cfg.get("num_workers", 0))
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        drop_last=True,
    )

    use_attention = bool(config.get("model", {}).get("use_attention", True))
    projection_dim = int(ssl_cfg.get("projection_dim", 128))

    backbone = timm.create_model(
        "efficientnet_b5",
        pretrained=True,
        se_layer=EcaModule,
        num_classes=0,
    )
    if use_attention:
        _inject_spatial_attention(backbone)

    model = _SimCLRModel(
        backbone,
        feature_dim=_EFFICIENTNET_B5_FEATURE_DIM,
        projection_dim=projection_dim,
    ).to(device)

    epochs = int(ssl_cfg.get("epochs", 100))
    lr = float(ssl_cfg.get("learning_rate", 3e-4))
    weight_decay = float(ssl_cfg.get("weight_decay", 1e-4))
    temperature = float(ssl_cfg.get("temperature", 0.07))
    checkpoint_interval = int(ssl_cfg.get("checkpoint_interval", 10))
    amp_enabled = bool(train_cfg.get("amp", False)) and device.type == "cuda"

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    amp_dtype = (
        torch.bfloat16
        if device.type == "cuda" and torch.cuda.is_bf16_supported()
        else torch.float16
    )

    output_dir = resolve_project_path(project_root, ssl_cfg.get("output_dir", "artifacts/ssl"))
    output_dir.mkdir(parents=True, exist_ok=True)

    total_batches_per_epoch = len(loader)
    log_interval = max(1, total_batches_per_epoch // 5)
    history: list[dict[str, Any]] = []
    best_loss = float("inf")

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total_batches = 0

        for batch_idx, batch in enumerate(loader, 1):
            view1 = batch["view1"].to(device)
            view2 = batch["view2"].to(device)

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
                z1 = model(view1)
                z2 = model(view2)
                loss = nt_xent_loss(z1, z2, temperature=temperature)

            loss.backward()
            optimizer.step()

            total_loss += float(loss.detach().item())
            total_batches += 1

            if batch_idx % log_interval == 0 or batch_idx == total_batches_per_epoch:
                LOGGER.info(
                    "ssl epoch=%s/%s batch=%s/%s loss=%.4f",
                    epoch, epochs, batch_idx, total_batches_per_epoch,
                    total_loss / total_batches,
                )

        scheduler.step()
        avg_loss = total_loss / max(total_batches, 1)
        history.append({"epoch": epoch, "loss": avg_loss})
        LOGGER.info("ssl epoch=%s/%s avg_loss=%.4f", epoch, epochs, avg_loss)

        if avg_loss < best_loss:
            best_loss = avg_loss
            _save_backbone(model.backbone, output_dir / "backbone_best.pt")

        if epoch % checkpoint_interval == 0:
            _save_backbone(model.backbone, output_dir / f"backbone_epoch{epoch:04d}.pt")

    _save_backbone(model.backbone, output_dir / "backbone_last.pt")

    summary: dict[str, Any] = {
        "project_root": str(project_root),
        "config_path": str(config_path),
        "manifest_path": str(manifest_path),
        "dataset_size": len(dataset),
        "epochs": epochs,
        "best_loss": best_loss,
        "backbone_best_path": str(output_dir / "backbone_best.pt"),
        "backbone_last_path": str(output_dir / "backbone_last.pt"),
        "history": history,
    }
    summary_path = output_dir / "ssl_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    summary["summary_path"] = str(summary_path)
    return summary
