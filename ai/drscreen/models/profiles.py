"""아키텍처별 입력 정규화 + 학습/추론 하이퍼파라미터 프로필 레지스트리.

각 모델이 기대하는 입력 크기/보간법/정규화 통계(mean/std)와 권장 학습 설정을
한곳에 모은다. service.py는 추론 transform을 만들 때 이 프로필의 mean/std/
interpolation을 사용한다. rationale 필드는 그 설정을 고른 이유를 영어로 남긴
메모이며, 동작에는 영향을 주지 않는다.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace

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
    # 융합 모델은 입력 정규화를 v31 분류기(EfficientNet-B5) 프로필과 동일하게 쓰므로
    # 그 프로필을 복제하고 architecture/rationale만 교체한다.
    if architecture == "v31_v8b_fusion":
        return replace(
            get_model_profile("efficientnet_b5"),
            architecture=architecture,
            rationale=(
                "Deployment wrapper: v31 EfficientNet-B5 classifier plus v8b "
                "ResNet50 lesion evidence segmenter. Input normalization follows "
                "the v31 EfficientNet/ImageNet profile."
            ),
        )

    if architecture == "concept_bottleneck":
        return ModelProfile(
            architecture=architecture,
            resize_size=512,
            crop_size=512,
            interpolation="bicubic",
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
            num_params=24_000_000,
            gflops=10.0,
            optimizer="adamw",
            scheduler="cosine",
            batch_size=4,
            head_learning_rate=2e-4,
            backbone_learning_rate=4e-5,
            weight_decay=1e-4,
            head_epochs=3,
            finetune_epochs=8,
            warmup_epochs=1,
            gradient_clip_norm=1.0,
            use_attention=False,
            gradcam_target_layer="concept_map",
            rationale=(
                "EfficientNet-B5 backbone with a spatial MA/HE/EX/SE concept head. "
                "The abnormal logit is a linear function of pooled concept logits, "
                "so the evidence map is in the classifier forward path."
            ),
        )

    if architecture == "sparse_bagnet":
        return ModelProfile(
            architecture=architecture,
            resize_size=512,
            crop_size=512,
            interpolation="bilinear",
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
            num_params=2_000_000,
            gflops=2.0,
            optimizer="adamw",
            scheduler="cosine",
            batch_size=16,
            head_learning_rate=5e-4,
            backbone_learning_rate=1e-4,
            weight_decay=1e-4,
            head_epochs=0,
            finetune_epochs=12,
            warmup_epochs=1,
            gradient_clip_norm=1.0,
            use_attention=False,
            gradcam_target_layer="patch_logits",
            rationale=(
                "Sparse BagNet-style local-evidence classifier. The image logit is "
                "computed from local patch logits, so the evidence map is part of "
                "the forward path instead of a post-hoc attribution."
            ),
        )

    if architecture == "deeplabv3_resnet50":
        return ModelProfile(
            architecture=architecture,
            resize_size=512,
            crop_size=512,
            interpolation="bilinear",
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
            num_params=39_000_000,
            gflops=178.7,
            optimizer="adamw",
            scheduler="cosine",
            batch_size=2,
            head_learning_rate=1e-4,
            backbone_learning_rate=1e-5,
            weight_decay=1e-4,
            head_epochs=0,
            finetune_epochs=40,
            warmup_epochs=0,
            gradient_clip_norm=1.0,
            use_attention=False,
            gradcam_target_layer="",
            rationale=(
                "Phase 4-G stronger segmentation baseline. DeepLabV3 adds atrous "
                "context aggregation on an ImageNet-pretrained ResNet50 backbone while "
                "keeping the evidence path classifier-independent."
            ),
        )

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
                "timm EfficientNet-B5 with _EcaSpatialAttn as se_layer (ECA channel attention + "
                "CBAM spatial attention integrated inside each MBConv at the SE position). "
                "Input 448x448 bicubic. Attention lives inside the block so Grad-CAM target "
                "blocks.6 reflects clean residual output, not an attention-modulated surface."
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
