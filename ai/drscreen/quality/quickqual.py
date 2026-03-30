from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np
import torch
from PIL import Image as PILImage
from torchvision import models, transforms

if TYPE_CHECKING:
    from typing import Any

QuickQualGrade = Literal["good", "usable", "reject"]
_GRADE_NAMES: tuple[str, ...] = ("good", "usable", "reject")

_TRANSFORM = transforms.Compose(
    [
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


@dataclass(slots=True)
class QuickQualResult:
    grade: QuickQualGrade
    confidence: float
    probabilities: dict[str, float]

    @property
    def is_reject(self) -> bool:
        return self.grade == "reject"


class QuickQualAssessor:
    """QuickQual fundus image quality assessor.

    DenseNet121 (ImageNet pretrained) backbone + SVM classifier.
    Classifies fundus image quality as Good / Usable / Reject.

    Weights: quickqual_dn121_512.pkl — download from:
    https://github.com/justinengelmann/QuickQual/releases/download/1.0/quickqual_dn121_512.pkl
    Place at artifacts/quickqual/quickqual_dn121_512.pkl.

    Reference: Engelmann et al., "QuickQual" (2023) https://arxiv.org/abs/2307.13646
    Performance: EyeQ accuracy 88.50%, AUC 0.9687 (vs MCF-Net 88.00%, 0.9588).
    """

    def __init__(self, weights_path: str | Path, device: str | torch.device = "cpu") -> None:
        self._device = torch.device(device) if isinstance(device, str) else device
        backbone = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
        backbone.classifier = torch.nn.Identity()
        self._backbone = backbone.to(self._device).eval()
        with open(weights_path, "rb") as f:
            self._svm = pickle.load(f)

    def assess(self, image: np.ndarray) -> QuickQualResult:
        """Assess quality of an RGB numpy image (H x W x 3, uint8)."""
        tensor = _TRANSFORM(PILImage.fromarray(image)).unsqueeze(0).to(self._device)
        with torch.inference_mode():
            features = self._backbone(tensor).cpu().numpy()

        pred_idx = int(self._svm.predict(features)[0])
        grade: QuickQualGrade = _GRADE_NAMES[pred_idx]  # type: ignore[assignment]

        if hasattr(self._svm, "predict_proba"):
            raw_probs = self._svm.predict_proba(features)[0]
            probs = {name: float(raw_probs[i]) for i, name in enumerate(_GRADE_NAMES)}
            confidence = float(raw_probs[pred_idx])
        else:
            probs = {name: float(i == pred_idx) for i, name in enumerate(_GRADE_NAMES)}
            confidence = 1.0

        return QuickQualResult(grade=grade, confidence=confidence, probabilities=probs)

    @classmethod
    def from_config(
        cls,
        config: dict[str, Any],
        project_root: Path,
        device: str | torch.device = "cpu",
    ) -> QuickQualAssessor | None:
        """Load assessor from config. Returns None if weights_path absent or file missing."""
        cfg = config.get("quickqual") or {}
        weights_str = cfg.get("weights_path")
        if not weights_str:
            return None
        weights_path = project_root / weights_str
        if not weights_path.exists():
            return None
        return cls(weights_path, device=device)
