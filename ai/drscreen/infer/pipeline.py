from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Sequence

import numpy as np
import torch
from PIL import Image

from drscreen.quality.checks import (
    ImageQualityResult,
    assess_image_quality,
    build_quality_warning_message,
)
from drscreen.quality.quickqual import QuickQualGrade

if TYPE_CHECKING:
    from drscreen.quality.quickqual import QuickQualAssessor


@dataclass(slots=True)
class InferenceResult:
    predicted_index: int
    predicted_label: str
    abnormal_probability: float
    should_block: bool
    quality_warning: str | None
    quality: ImageQualityResult
    quality_grade: QuickQualGrade | None
    quality_grade_confidence: float | None

    def to_dict(self) -> dict[str, object]:
        data = asdict(self)
        data["quality"] = asdict(self.quality)
        return data


def load_rgb_image(image_path: str | Path) -> np.ndarray:
    return np.asarray(Image.open(image_path).convert("RGB"))


def run_single_image_inference(
    model: torch.nn.Module,
    image_tensor: torch.Tensor,
    raw_image: np.ndarray,
    label_names: Sequence[str] = ("normal", "abnormal"),
    blur_threshold: float = 100.0,
    brightness_threshold: float = 40.0,
    low_quality_action: str = "warn",
    quality_assessor: QuickQualAssessor | None = None,
) -> InferenceResult:
    quality = assess_image_quality(
        raw_image,
        blur_threshold=blur_threshold,
        brightness_threshold=brightness_threshold,
        low_quality_action=low_quality_action,
    )

    quality_grade: QuickQualGrade | None = None
    quality_grade_confidence: float | None = None
    if quality_assessor is not None:
        quality_result = quality_assessor.assess(raw_image)
        quality_grade = quality_result.grade
        quality_grade_confidence = quality_result.confidence

    model.eval()
    with torch.inference_mode():
        logits = model(image_tensor.unsqueeze(0))
        if logits.shape[-1] == 1:
            abnormal_probability = torch.sigmoid(logits[0, 0]).item()
            predicted_index = int(abnormal_probability >= 0.5)
        else:
            probabilities = torch.softmax(logits[0], dim=0)
            predicted_index = int(torch.argmax(probabilities).item())
            abnormal_probability = float(probabilities[min(1, len(probabilities) - 1)].item())

    return InferenceResult(
        predicted_index=predicted_index,
        predicted_label=label_names[predicted_index],
        abnormal_probability=abnormal_probability,
        should_block=quality.is_low_quality and quality.recommended_action == "block",
        quality_warning=build_quality_warning_message(quality),
        quality=quality,
        quality_grade=quality_grade,
        quality_grade_confidence=quality_grade_confidence,
    )
