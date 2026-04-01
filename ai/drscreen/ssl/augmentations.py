from __future__ import annotations

import cv2
import numpy as np
import torch
from PIL import Image as PILImage

import albumentations as A
from albumentations.pytorch import ToTensorV2

from drscreen.data.transforms import FundusPreprocess


class SSLAugmentationPair:
    """Returns two independently augmented views of the same fundus image.

    Ben Graham preprocessing is applied once before augmentation. The SSL
    augmentations are stronger than the supervised training transforms so that
    the model is forced to learn domain-invariant structural representations
    rather than domain-specific color or brightness statistics.
    """

    def __init__(
        self,
        image_size: int = 224,
        mean: tuple[float, float, float] = (0.485, 0.456, 0.406),
        std: tuple[float, float, float] = (0.229, 0.224, 0.225),
        use_preprocessing: bool = True,
    ) -> None:
        self._preprocess = FundusPreprocess() if use_preprocessing else None
        self._aug = A.Compose([
            A.Resize(image_size, image_size),
            A.RandomResizedCrop(size=(image_size, image_size), scale=(0.5, 1.0)),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Rotate(limit=180, border_mode=cv2.BORDER_CONSTANT, fill=0, p=0.8),
            # Stronger photometric augmentation than supervised training:
            # forces the encoder to learn structure rather than color/brightness cues.
            A.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.05, p=0.8),
            A.RandomGamma(gamma_limit=(60, 140), p=0.5),
            A.GaussianBlur(blur_limit=(3, 7), p=0.3),
            A.GaussNoise(std_range=(0.02, 0.1), p=0.3),
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ])

    def __call__(self, img: PILImage.Image) -> tuple[torch.Tensor, torch.Tensor]:
        if self._preprocess is not None:
            img = self._preprocess(img)
        arr = np.asarray(img)
        view1 = self._aug(image=arr.copy())["image"]
        view2 = self._aug(image=arr.copy())["image"]
        return view1, view2
