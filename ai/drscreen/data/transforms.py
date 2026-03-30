from __future__ import annotations

import cv2
import numpy as np
import torch
from PIL import Image as PILImage
from torchvision import transforms

from drscreen.models.profiles import get_model_profile, resolve_interpolation_mode



class FundusPreprocess:
    """Fundus-specific adaptive preprocessing pipeline.

    Two-stage pipeline:
    1. Circular crop — removes black border padding introduced by fundus
       cameras. Black borders distort CLAHE histograms and waste model
       capacity on uninformative pixels.
    2. Ben Graham normalization — subtracts the local mean illumination
       (Gaussian-blurred version of the image) to remove uneven lighting.
       sigmaX scales with the image's longest dimension so the operation
       is resolution-adaptive.

    Replaces the previous CLAHE + GaussianBlur pipeline, which applied
    CLAHE to all channels before removing black borders, causing
    histogram distortion and uniform illumination artefacts.

    Reference: Graham B., "Kaggle Diabetic Retinopathy Detection", 2015
               (1st place, ~0.84 QWK).
    """

    def __init__(
        self,
        crop_tol: int = 7,
        ben_graham_weight: float = 4.0,
        ben_graham_offset: float = 128.0,
        output_size: int | None = None,
    ) -> None:
        self._crop_tol = crop_tol
        self._weight = ben_graham_weight
        self._offset = ben_graham_offset
        self._output_size = output_size

    def __call__(self, img: PILImage.Image) -> PILImage.Image:
        arr = np.asarray(img.convert("RGB")).copy()
        arr = self._circular_crop(arr)
        arr = self._ben_graham(arr)
        result = PILImage.fromarray(arr)
        if self._output_size is not None:
            result = result.resize((self._output_size, self._output_size), PILImage.BICUBIC)
        return result

    def _circular_crop(self, image: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        _, mask = cv2.threshold(gray, self._crop_tol, 255, cv2.THRESH_BINARY)
        coords = cv2.findNonZero(mask)
        if coords is None:
            return image
        x, y, w, h = cv2.boundingRect(coords)
        cropped = image[y : y + h, x : x + w]
        side = max(w, h)
        pad_top = (side - h) // 2
        pad_bottom = side - h - pad_top
        pad_left = (side - w) // 2
        pad_right = side - w - pad_left
        return cv2.copyMakeBorder(
            cropped, pad_top, pad_bottom, pad_left, pad_right,
            cv2.BORDER_CONSTANT, value=(0, 0, 0),
        )

    def _ben_graham(self, image: np.ndarray) -> np.ndarray:
        sigma_x = max(image.shape[:2]) / 30.0
        blurred = cv2.GaussianBlur(image, (0, 0), sigma_x)
        result = cv2.addWeighted(image, self._weight, blurred, -self._weight, self._offset)
        return np.clip(result, 0, 255).astype(np.uint8)


def build_train_transform(
    crop_size: int,
    resize_size: int | None = None,
    interpolation: str = "bilinear",
    mean: tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: tuple[float, float, float] = (0.229, 0.224, 0.225),
    use_preprocessing: bool = False,
) -> transforms.Compose:
    resize = resize_size or crop_size
    interpolation_mode = resolve_interpolation_mode(interpolation)
    steps = []
    if use_preprocessing:
        steps.append(FundusPreprocess())
    steps.extend(
        [
            transforms.Resize((resize, resize), interpolation=interpolation_mode),
            transforms.RandomResizedCrop(
                crop_size,
                scale=(0.85, 1.0),
                interpolation=interpolation_mode,
            ),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )
    return transforms.Compose(steps)


def build_eval_transform(
    crop_size: int,
    resize_size: int | None = None,
    interpolation: str = "bilinear",
    mean: tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: tuple[float, float, float] = (0.229, 0.224, 0.225),
    use_preprocessing: bool = False,
) -> transforms.Compose:
    resize = resize_size or crop_size
    interpolation_mode = resolve_interpolation_mode(interpolation)
    steps = []
    if use_preprocessing:
        steps.append(FundusPreprocess())
    steps.extend(
        [
            transforms.Resize((resize, resize), interpolation=interpolation_mode),
            transforms.CenterCrop(crop_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )
    return transforms.Compose(steps)


def build_transforms_for_model(model_name: str) -> tuple[transforms.Compose, transforms.Compose]:
    profile = get_model_profile(model_name)
    train_transform = build_train_transform(
        crop_size=profile.crop_size,
        resize_size=profile.resize_size,
        interpolation=profile.interpolation,
        mean=profile.mean,
        std=profile.std,
    )
    eval_transform = build_eval_transform(
        crop_size=profile.crop_size,
        resize_size=profile.resize_size,
        interpolation=profile.interpolation,
        mean=profile.mean,
        std=profile.std,
    )
    return train_transform, eval_transform


def denormalize_tensor(tensor: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor((0.485, 0.456, 0.406), device=tensor.device).view(-1, 1, 1)
    std = torch.tensor((0.229, 0.224, 0.225), device=tensor.device).view(-1, 1, 1)
    return (tensor * std) + mean
