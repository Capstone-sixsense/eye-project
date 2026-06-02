from __future__ import annotations

import torch

from drscreen.models.fusion import V31V8bFusion


class _ConstantClassifier(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.full((x.shape[0], 1), 0.25, dtype=x.dtype, device=x.device)


class _CountingSegmenter(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.calls += 1
        return torch.zeros((x.shape[0], 4, x.shape[2], x.shape[3]), dtype=x.dtype, device=x.device)


def test_predict_fusion_score_runs_segmenter_once() -> None:
    segmenter = _CountingSegmenter()
    model = V31V8bFusion(
        classifier=_ConstantClassifier(),
        segmenter=segmenter,
        feature_schema=["v31_probability", "v31_logit"],
    )
    output = model.predict_fusion_score(torch.ones(1, 3, 8, 8))

    assert segmenter.calls == 1
    assert output["seg_prob"].shape == (1, 4, 8, 8)
    assert output["meta_probability"] is None
