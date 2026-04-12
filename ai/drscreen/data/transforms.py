from __future__ import annotations

import cv2
import numpy as np
import torch
from PIL import Image as PILImage
from torchvision import transforms

import albumentations as A
from albumentations.pytorch import ToTensorV2

from drscreen.models.profiles import resolve_interpolation_mode


class FundusPreprocess:
    """Fundus-specific adaptive preprocessing pipeline.

    Two-stage pipeline:
    1. Circular crop -- removes black border padding introduced by fundus
       cameras. Black borders distort CLAHE histograms and waste model
       capacity on uninformative pixels.
    2. Ben Graham normalization -- subtracts the local mean illumination
       (Gaussian-blurred version of the image) to remove uneven lighting.
       A circular mask fills the black background with the per-channel
       fundus mean before blurring, preventing border bleed (dark halo
       artifact at the retinal boundary). sigmaX scales with the image's
       longest dimension so the operation is resolution-adaptive.

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

    def _get_circle_mask(self, h: int, w: int) -> np.ndarray:
        cy, cx = h / 2.0, w / 2.0
        radius = min(h, w) / 2.0
        Y, X = np.ogrid[:h, :w]
        return ((X - cx) ** 2 + (Y - cy) ** 2) <= radius ** 2

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
        h, w = image.shape[:2]
        mask = self._get_circle_mask(h, w)

        # Fill non-fundus pixels with per-channel mean of the fundus region
        # before blurring. Without this, the black background (value=0)
        # bleeds inward via Gaussian blur, creating a dark halo at the
        # retinal boundary that suppresses peripheral lesion signals.
        work = image.copy()
        for c in range(3):
            channel = image[:, :, c]
            fill_val = int(channel[mask].mean()) if mask.any() else 128
            work[:, :, c][~mask] = fill_val

        sigma_x = max(h, w) / 30.0
        blurred = cv2.GaussianBlur(work, (0, 0), sigma_x)
        result = cv2.addWeighted(work, self._weight, blurred, -self._weight, self._offset)
        result = np.clip(result, 0, 255).astype(np.uint8)

        # Restore circular mask: zero out non-fundus region.
        result[~mask] = 0
        return result


class _TrainTransform:
    """PIL-compatible wrapper for the albumentations-based training pipeline."""

    def __init__(self, pil_steps: list, aug: A.Compose) -> None:
        self._pil_steps = pil_steps
        self._aug = aug

    def __call__(self, img: PILImage.Image) -> torch.Tensor:
        for step in self._pil_steps:
            img = step(img)
        return self._aug(image=np.asarray(img))["image"]


def build_train_transform(
    crop_size: int,
    resize_size: int | None = None,
    interpolation: str = "bilinear",
    mean: tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: tuple[float, float, float] = (0.229, 0.224, 0.225),
    use_preprocessing: bool = False,
) -> _TrainTransform:
    resize = resize_size or crop_size

    pil_steps: list = []
    if use_preprocessing:
        pil_steps.append(FundusPreprocess())

    aug = A.Compose([
        # Resize to intermediate resolution before crop so that scale=(0.8, 1.0)
        # crops from a higher-res image, avoiding quality degradation.
        # Lower bound 0.8 (not 0.7) to avoid cropping out peripheral lesions.
        A.Resize(resize, resize),
        A.RandomResizedCrop(size=(crop_size, crop_size), scale=(0.8, 1.0)),

        # Geometric: fundus lesions are rotation-invariant; all orientations valid.
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Rotate(limit=180, border_mode=cv2.BORDER_CONSTANT, fill=0, p=0.8),

        # Photometric: simulate camera/exposure variation across domains.
        # hue is capped at 0.03 -- hemorrhages (red) and exudates (yellow-white)
        # carry diagnostic color information that must not be distorted.
        A.ColorJitter(brightness=0.35, contrast=0.35, saturation=0.25, hue=0.03, p=0.8),
        A.RandomGamma(gamma_limit=(75, 130), p=0.5),

        # Sensor and optics variation: low probability / low magnitude to
        # preserve fine lesion morphology.
        A.GaussianBlur(blur_limit=(3, 5), p=0.2),
        A.GaussNoise(std_range=(0.02, 0.07), p=0.3),

        A.Normalize(mean=mean, std=std),
        ToTensorV2(),
    ])

    return _TrainTransform(pil_steps, aug)


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


