from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np


@dataclass(slots=True)
class ImageQualityResult:
    blur_score: float
    mean_brightness: float
    is_low_quality: bool
    recommended_action: str
    reasons: list[str] = field(default_factory=list)


def to_grayscale(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Expected grayscale or RGB image.")
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)


def compute_blur_score(image: np.ndarray) -> float:
    gray = to_grayscale(image)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def compute_mean_brightness(image: np.ndarray) -> float:
    gray = to_grayscale(image)
    return float(gray.mean())


def assess_image_quality(
    image: np.ndarray,
    blur_threshold: float = 100.0,
    brightness_threshold: float = 40.0,
    low_quality_action: str = "warn",
) -> ImageQualityResult:
    if low_quality_action not in {"warn", "block"}:
        raise ValueError("low_quality_action must be 'warn' or 'block'.")

    blur_score = compute_blur_score(image)
    mean_brightness = compute_mean_brightness(image)

    reasons: list[str] = []
    if blur_score < blur_threshold:
        reasons.append("blur_score_below_threshold")
    if mean_brightness < brightness_threshold:
        reasons.append("brightness_below_threshold")

    return ImageQualityResult(
        blur_score=blur_score,
        mean_brightness=mean_brightness,
        is_low_quality=bool(reasons),
        recommended_action=low_quality_action if reasons else "pass",
        reasons=reasons,
    )


def build_quality_warning_message(result: ImageQualityResult) -> str | None:
    if not result.is_low_quality:
        return None

    reasons: list[str] = []
    if "blur_score_below_threshold" in result.reasons:
        reasons.append("blur")
    if "brightness_below_threshold" in result.reasons:
        reasons.append("darkness")
    if not reasons:
        reasons.append("quality")

    reason_text = " and ".join(reasons)
    if result.recommended_action == "block":
        return f"Low image quality detected ({reason_text}). Review is required before inference."
    return f"Low image quality detected ({reason_text}). Inference continues with a warning."
