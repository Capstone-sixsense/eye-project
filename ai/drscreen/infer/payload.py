"""백엔드로 전달되는 추론 결과 페이로드 계약(contract).

이 dataclass의 필드 = `eye-project` 백엔드가 소비하는 JSON 스키마다
(`docs/AI_HANDOFF.md` 5절). 필드 추가/이름 변경은 백엔드와의 계약 변경이므로
tests/regression/test_payload_contract.py가 이 구조를 고정 검증한다.

quality_* 필드는 QuickQual이 별도 백엔드 태스크로 분리된 뒤 항상 None으로
채워지는 호환용 placeholder다(AI는 더 이상 품질 필터링을 하지 않는다).
"""

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
