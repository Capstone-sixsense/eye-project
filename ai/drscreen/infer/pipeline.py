"""단일 이미지 분류 추론의 최소 단위.

메타 융합(fusion)을 쓰지 않는 일반 분류기를 위한 경로다. service.py는
`use_meta_classifier=False`일 때 이 함수를 호출하고, 융합 모델일 때는
V31V8bFusion.predict_fusion_score를 직접 쓴다.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from collections.abc import Sequence

import torch


@dataclass(slots=True)
class InferenceResult:
    predicted_index: int
    predicted_label: str
    abnormal_probability: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def run_single_image_inference(
    model: torch.nn.Module,
    image_tensor: torch.Tensor,
    label_names: Sequence[str] = ("normal", "abnormal"),
    threshold: float = 0.5,
) -> InferenceResult:
    model.eval()
    with torch.inference_mode():
        # image_tensor [C,H,W] -> [1,C,H,W] 배치 차원 추가 후 forward.
        logits = model(image_tensor.unsqueeze(0))
        if logits.shape[-1] == 1:
            # 단일 출력 헤드: sigmoid 확률이 임계값 이상이면 abnormal(1).
            abnormal_probability = torch.sigmoid(logits[0, 0]).item()
            predicted_index = int(abnormal_probability >= threshold)
        else:
            # 다중 출력 헤드: argmax로 예측 클래스, 확률은 '비정상' 클래스(index 1) 기준.
            probabilities = torch.softmax(logits[0], dim=0)
            predicted_index = int(torch.argmax(probabilities).item())
            abnormal_probability = float(probabilities[min(1, len(probabilities) - 1)].item())

    return InferenceResult(
        predicted_index=predicted_index,
        predicted_label=label_names[predicted_index],
        abnormal_probability=abnormal_probability,
    )
