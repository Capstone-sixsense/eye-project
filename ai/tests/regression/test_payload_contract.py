from __future__ import annotations

from dataclasses import fields

from drscreen.infer.payload import InferencePayload


EXPECTED_PAYLOAD_KEYS = [
    "predicted_index",
    "predicted_label",
    "abnormal_probability",
    "decision_threshold",
    "checkpoint_path",
    "prediction_path",
    "heatmap_path",
    "xai_error_code",
    "xai_no_region",
    "evidence_type",
    "lesion_map_path",
    "lesion_summary",
    "evidence_warning",
    "quality",
    "quality_warning",
    "quality_grade",
    "quality_grade_confidence",
    "should_block",
    "eval_metrics",
]


def test_inference_payload_contract_keys_are_stable() -> None:
    assert [field.name for field in fields(InferencePayload)] == EXPECTED_PAYLOAD_KEYS
