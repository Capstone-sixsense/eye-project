from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(slots=True)
class InferencePayload:
    # Core prediction (mirrors InferenceResult)
    predicted_index: int
    predicted_label: str
    abnormal_probability: float
    # Classification metadata
    decision_threshold: float
    checkpoint_path: str
    # Artifact paths
    prediction_path: str | None
    heatmap_path: str | None
    # XAI
    xai_error_code: str | None
    xai_no_region: bool
    evidence_type: str | None = None
    lesion_map_path: str | None = None
    lesion_summary: dict[str, Any] | None = None
    evidence_warning: str | None = None
    # Quality fields — reserved, always None (populated by quality model when added)
    quality: None = None
    quality_warning: None = None
    quality_grade: None = None
    quality_grade_confidence: None = None
    # Backend gating
    should_block: bool = False
    # Eval metrics from external_test_{version}_best_metrics.json (None if file absent)
    eval_metrics: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
