from __future__ import annotations

from dataclasses import asdict, dataclass

from torchvision import models
from torchvision.transforms import InterpolationMode


@dataclass(frozen=True, slots=True)
class ModelProfile:
    architecture: str
    resize_size: int
    crop_size: int
    interpolation: str
    mean: tuple[float, float, float]
    std: tuple[float, float, float]
    num_params: int
    gflops: float
    optimizer: str
    scheduler: str
    batch_size: int
    head_learning_rate: float
    backbone_learning_rate: float
    weight_decay: float
    head_epochs: int
    finetune_epochs: int
    warmup_epochs: int
    gradient_clip_norm: float
    use_attention: bool
    gradcam_target_layer: str
    rationale: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def get_weights_enum(architecture: str):
    mapping = {
        "resnet50": models.ResNet50_Weights.DEFAULT,
        "convnext_tiny": models.ConvNeXt_Tiny_Weights.DEFAULT,
    }
    if architecture not in mapping:
        raise ValueError(f"Unsupported model architecture: {architecture}")
    return mapping[architecture]


def resolve_interpolation_mode(name: str) -> InterpolationMode:
    return InterpolationMode(name.lower())


def get_model_profile(architecture: str) -> ModelProfile:
    if architecture == "efficientnet_b5":
        # timm EfficientNet-B5 with ECA replacing SE blocks.
        # Input config derived from timm data_config: 448x448, bicubic, ImageNet stats.
        # Parameter count: 25.2M (ECA uses 1D conv instead of FC, ~5M fewer than torchvision SE variant).
        return ModelProfile(
            architecture=architecture,
            resize_size=448,
            crop_size=448,
            interpolation="bicubic",
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
            num_params=23_155_054,
            gflops=9.9,
            optimizer="adamw",
            scheduler="cosine",
            batch_size=4,
            head_learning_rate=2e-4,
            backbone_learning_rate=8e-5,
            weight_decay=1e-4,
            head_epochs=3,
            finetune_epochs=15,
            warmup_epochs=2,
            gradient_clip_norm=1.0,
            use_attention=True,
            gradcam_target_layer="blocks.6",
            rationale=(
                "timm EfficientNet-B5 with ECA (replaces SE, no dimensionality reduction) and "
                "CBAM Spatial Attention injected after every MBConv block. Input 448x448 bicubic. "
                "ECA reduces parameters by ~5M vs the torchvision SE variant while strengthening "
                "inter-channel relationships via 1D convolution."
            ),
        )

    weights = get_weights_enum(architecture)
    transforms = weights.transforms()
    resize_size = int(transforms.resize_size[0])
    crop_size = int(transforms.crop_size[0])
    interpolation = transforms.interpolation.value
    mean = tuple(float(v) for v in transforms.mean)
    std = tuple(float(v) for v in transforms.std)
    meta = weights.meta
    gflops = float(meta["_ops"])

    if architecture == "resnet50":
        return ModelProfile(
            architecture=architecture,
            resize_size=resize_size,
            crop_size=crop_size,
            interpolation=interpolation,
            mean=mean,
            std=std,
            num_params=int(meta["num_params"]),
            gflops=gflops,
            optimizer="adamw",
            scheduler="cosine",
            batch_size=16,
            head_learning_rate=5e-4,
            backbone_learning_rate=1.5e-4,
            weight_decay=2e-5,
            head_epochs=2,
            finetune_epochs=10,
            warmup_epochs=1,
            gradient_clip_norm=1.0,
            use_attention=False,
            gradcam_target_layer="",
            rationale=(
                "Stable baseline with smaller crop size and lower preprocessing cost than "
                "EfficientNet-B3. The batch size can be increased, which makes it a good fallback "
                "and comparison model."
            ),
        )

    if architecture == "convnext_tiny":
        return ModelProfile(
            architecture=architecture,
            resize_size=resize_size,
            crop_size=crop_size,
            interpolation=interpolation,
            mean=mean,
            std=std,
            num_params=int(meta["num_params"]),
            gflops=gflops,
            optimizer="adamw",
            scheduler="cosine",
            batch_size=8,
            head_learning_rate=2e-4,
            backbone_learning_rate=5e-5,
            weight_decay=1e-2,
            head_epochs=3,
            finetune_epochs=15,
            warmup_epochs=2,
            gradient_clip_norm=1.0,
            use_attention=False,
            gradcam_target_layer="",
            rationale=(
                "Largest model here. It benefits from a lower fine-tuning LR and stronger weight "
                "decay, so it is better treated as a challenger than the first implementation target."
            ),
        )

    raise ValueError(f"Unsupported model architecture: {architecture}")
