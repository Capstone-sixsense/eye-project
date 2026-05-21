from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from drscreen.infer.late_fusion_features import (
    FUSION_AREA_THRESHOLDS,
    FUSION_TOPK_FRACS,
    extract_late_fusion_features,
)


class V31V8bFusion(nn.Module):
    """Deployment wrapper for v31 classification + v8b lesion evidence."""

    def __init__(
        self,
        classifier: nn.Module,
        segmenter: nn.Module,
        *,
        meta_params: dict[str, Any] | None = None,
        feature_schema: Sequence[str] | None = None,
        feature_extraction: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        self.classifier = classifier
        self.segmenter = segmenter
        self.meta_params = self._normalize_meta_params(meta_params)
        self.feature_schema = list(feature_schema or [])
        self.feature_extraction = dict(feature_extraction or {})

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(x)

    @torch.no_grad()
    def predict_seg(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.segmenter(x))

    @torch.no_grad()
    def predict_fusion_score(self, x: torch.Tensor) -> dict[str, Any]:
        logits = self.classifier(x)
        if isinstance(logits, tuple):
            logits = logits[0]
        logits = logits.reshape(logits.shape[0], -1)
        if logits.shape[0] != 1:
            raise ValueError("predict_fusion_score expects a single image batch.")
        if logits.shape[1] == 1:
            v31_logit = float(logits[0, 0].detach().cpu().item())
            v31_probability = float(torch.sigmoid(logits[0, 0]).detach().cpu().item())
        else:
            probs = torch.softmax(logits[0], dim=0)
            v31_probability = float(probs[min(1, len(probs) - 1)].detach().cpu().item())
            clipped = float(np.clip(v31_probability, 1e-6, 1.0 - 1e-6))
            v31_logit = float(np.log(clipped / (1.0 - clipped)))

        seg_prob = self.predict_seg(x)
        schema = self.feature_schema
        if not schema:
            raise ValueError("Fusion feature_schema is empty.")
        area_thresholds = self.feature_extraction.get("area_thresholds", FUSION_AREA_THRESHOLDS)
        topk_fracs = self.feature_extraction.get("topk_fracs", FUSION_TOPK_FRACS)
        features = extract_late_fusion_features(
            v31_probability=v31_probability,
            v31_logit=v31_logit,
            seg_prob=seg_prob[0],
            schema=schema,
            area_thresholds=area_thresholds,
            topk_fracs=topk_fracs,
        )
        meta_probability = None
        meta_logit = None
        if self.meta_params is not None:
            x_arr = np.asarray(features, dtype=np.float64)
            scaled = (x_arr - self.meta_params["scaler_mean"]) / self.meta_params["scaler_scale"]
            meta_logit = float(np.dot(scaled, self.meta_params["coef"]) + self.meta_params["intercept"])
            meta_probability = float(1.0 / (1.0 + np.exp(-meta_logit)))
        return {
            "v31_probability": v31_probability,
            "v31_logit": v31_logit,
            "seg_prob": seg_prob,
            "features": features,
            "meta_logit": meta_logit,
            "meta_probability": meta_probability,
        }

    @staticmethod
    def _normalize_meta_params(params: dict[str, Any] | None) -> dict[str, np.ndarray | float] | None:
        if params is None:
            return None
        coef = np.asarray(params["coef"], dtype=np.float64)
        if coef.ndim == 2:
            if coef.shape[0] != 1:
                raise ValueError(f"Expected binary coef shape [1,N], got {coef.shape}")
            coef = coef[0]
        intercept = np.asarray(params["intercept"], dtype=np.float64)
        if intercept.size != 1:
            raise ValueError(f"Expected one intercept, got {intercept.shape}")
        scaler_scale = np.asarray(params["scaler_scale"], dtype=np.float64)
        scaler_scale = np.where(scaler_scale == 0.0, 1.0, scaler_scale)
        return {
            "scaler_mean": np.asarray(params["scaler_mean"], dtype=np.float64),
            "scaler_scale": scaler_scale,
            "coef": coef,
            "intercept": float(intercept.reshape(-1)[0]),
        }
