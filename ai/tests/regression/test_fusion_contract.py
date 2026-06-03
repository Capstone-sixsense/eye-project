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


def test_extractor_covers_v31_logit_only_schema() -> None:
    # Problem 1 collinearity fix: dropping v31_probability from the packaged
    # schema must stay safe because extraction is schema-driven and always
    # computes both v31 columns. A v31_logit-only schema must extract cleanly.
    from drscreen.infer.late_fusion_features import (
        base_lesion_feature_names,
        extended_lesion_feature_names,
        extract_late_fusion_features,
    )

    schema = ["v31_logit", *base_lesion_feature_names(), *extended_lesion_feature_names()]
    values = extract_late_fusion_features(
        v31_probability=0.7,
        v31_logit=0.8472978603872037,
        seg_prob=torch.rand(4, 16, 16),
        schema=schema,
    )
    assert len(values) == len(schema)


def test_od_anchored_features_are_anatomy_grounded() -> None:
    # Problem 3 Phase B: OD-anchored lesion features in seg_prob pixel space.
    from drscreen.infer.late_fusion_features import (
        extract_od_anchored_feature_dict,
        od_anchored_feature_names,
    )

    assert len(od_anchored_feature_names()) == 15  # (4 channels + union) x 3 features
    seg = torch.zeros(4, 32, 64)
    seg[0, 8:12, 38:42] = 0.9  # MA blob centred on the OD location
    d = extract_od_anchored_feature_dict(seg, od_xy=(40.0, 10.0), od_diameter=10.0)
    assert d["MA_peripapillary_ratio_ge_0.3"] == 1.0  # active MA within peripapillary radius
    assert d["MA_od_dist_mean_ge_0.3"] < 0.5  # blob sits near the OD
    assert d["EX_peripapillary_ratio_ge_0.3"] == 0.0  # no EX active -> zeroed
