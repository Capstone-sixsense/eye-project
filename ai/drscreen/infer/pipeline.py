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
        logits = model(image_tensor.unsqueeze(0))
        if logits.shape[-1] == 1:
            abnormal_probability = torch.sigmoid(logits[0, 0]).item()
            predicted_index = int(abnormal_probability >= threshold)
        else:
            probabilities = torch.softmax(logits[0], dim=0)
            predicted_index = int(torch.argmax(probabilities).item())
            abnormal_probability = float(probabilities[min(1, len(probabilities) - 1)].item())

    return InferenceResult(
        predicted_index=predicted_index,
        predicted_label=label_names[predicted_index],
        abnormal_probability=abnormal_probability,
    )
