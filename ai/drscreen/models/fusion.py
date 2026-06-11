"""v31 분류기 + v8b 병변 분할기를 결합하는 late-fusion(후기 융합) 배포 래퍼.

이 모듈은 현재 배포 모델 `v31_v8b_fusion_quickqual_v2`의 심장부다
(`docs/AI_HANDOFF.md` 1절 참조). 두 개의 독립 모델을 하나로 묶는다:

- classifier (v31): 안저 이미지를 보고 정상/비정상 logit을 출력한다.
- segmenter (v8b): 같은 이미지에서 4채널(MA/HE/EX/SE) 병변 확률맵을 만든다.

두 출력은 곧바로 최종 판정에 쓰이지 않는다. 대신 v31 log-odds 1개와
병변맵에서 뽑은 스칼라 특징(area/topk/std 등, 총 88개)을 모아
StandardScaler + LogisticRegression 메타 분류기에 통과시켜 최종
`meta_probability`(= abnormal_probability)를 만든다. 즉 "분류 점수"와
"병변 근거"를 선형 메타 모델이 결합하는 구조다.

핵심 메서드 흐름:
- forward()             : classifier 경로만 반환(학습/CAM 호환용). 최종 판정 아님.
- predict_seg()         : segmenter의 병변 확률맵(sigmoid) 반환.
- predict_fusion_score(): classifier+segmenter를 함께 돌려 메타 확률까지 산출(배포 경로).

메타 분류기 가중치(`meta_params`), 특징 순서(`feature_schema`), 특징 추출
설정(`feature_extraction`)은 체크포인트에 함께 저장되어 생성 시 주입된다.
"""

from __future__ import annotations

import math
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
        # 메타 분류기 파라미터(scaler 평균/스케일, LogReg 계수/절편)를 numpy로 정규화.
        # None이면 메타 융합 없이 v31/v8b 원시 출력만 사용 가능하다.
        self.meta_params = self._normalize_meta_params(meta_params)
        # 특징 벡터 순서(학습 때 fit한 순서와 반드시 일치해야 함).
        self.feature_schema = list(feature_schema or [])
        self.feature_extraction = dict(feature_extraction or {})

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 학습 루프 및 Grad-CAM 등 classifier 경로만 필요한 곳을 위한 통로.
        # 배포 최종 판정은 forward가 아니라 predict_fusion_score를 쓴다.
        return self.classifier(x)

    @torch.no_grad()
    def predict_seg(self, x: torch.Tensor, *, amp_enabled: bool = False) -> torch.Tensor:
        # segmenter logit -> sigmoid 확률맵. evidence 오버레이/특징 추출의 입력이 된다.
        with _bf16_autocast(x, enabled=amp_enabled):
            logits = self.segmenter(x)
        return torch.sigmoid(logits.float())

    @torch.no_grad()
    def predict_fusion_score(self, x: torch.Tensor, *, amp_enabled: bool = False) -> dict[str, Any]:
        # 배포 추론 경로: classifier와 segmenter를 한 번에 돌린 뒤 메타 확률까지 계산한다.
        with _bf16_autocast(x, enabled=amp_enabled):
            logits = self.classifier(x)
            seg_logits = self.segmenter(x)
        # MultiTaskModel은 (logits, seg) 튜플을 낼 수 있으므로 분류 logit만 취한다.
        if isinstance(logits, tuple):
            logits = logits[0]
        logits = logits.reshape(logits.shape[0], -1)
        # 이 메서드는 단일 이미지(batch=1) 추론 전용이다.
        if logits.shape[0] != 1:
            raise ValueError("predict_fusion_score expects a single image batch.")
        if logits.shape[1] == 1:
            # 단일 출력 헤드(num_outputs=1): logit과 그 sigmoid 확률을 그대로 사용.
            v31_logit = float(logits[0, 0].detach().cpu().item())
            v31_probability = float(torch.sigmoid(logits[0, 0]).detach().cpu().item())
        else:
            # 다중 출력 헤드: softmax로 '비정상' 클래스 확률을 얻고, 거기서 logit을 역산.
            probs = torch.softmax(logits[0], dim=0)
            v31_probability = float(probs[min(1, len(probs) - 1)].detach().cpu().item())
            clipped = float(np.clip(v31_probability, 1e-6, 1.0 - 1e-6))
            v31_logit = float(np.log(clipped / (1.0 - clipped)))

        seg_prob = torch.sigmoid(seg_logits.float())
        # 실제 특징 추출 + 메타 분류는 components 버전에 위임(TTA 재계산에서 재사용 가능).
        output = self.predict_fusion_from_components(
            v31_probability=v31_probability,
            v31_logit=v31_logit,
            seg_prob=seg_prob[0],
        )
        # 병변 확률맵은 evidence 오버레이/TTA에서 다시 쓰므로 함께 반환한다.
        output["seg_prob"] = seg_prob
        return output

    @torch.no_grad()
    def predict_fusion_from_components(
        self,
        *,
        v31_probability: float,
        v31_logit: float,
        seg_prob: torch.Tensor,
    ) -> dict[str, Any]:
        # 이미 계산된 v31 출력 + 병변맵으로부터 특징을 뽑아 메타 확률을 만든다.
        # (hflip feature-recalc TTA처럼 seg_prob만 바꿔 재계산할 때 직접 호출됨.)
        schema = self.feature_schema
        if not schema:
            raise ValueError("Fusion feature_schema is empty.")
        area_thresholds = self.feature_extraction.get("area_thresholds", FUSION_AREA_THRESHOLDS)
        topk_fracs = self.feature_extraction.get("topk_fracs", FUSION_TOPK_FRACS)
        # schema 순서대로 88개 특징 벡터를 구성(학습 시 fit한 순서와 동일해야 함).
        features = extract_late_fusion_features(
            v31_probability=v31_probability,
            v31_logit=v31_logit,
            seg_prob=seg_prob,
            schema=schema,
            area_thresholds=area_thresholds,
            topk_fracs=topk_fracs,
        )
        meta_probability = None
        meta_logit = None
        if self.meta_params is not None:
            # LogisticRegression 추론을 numpy로 직접 구현:
            #   1) StandardScaler 표준화: (x - mean) / scale
            #   2) 선형 결합: scaled . coef + intercept  -> meta_logit
            #   3) sigmoid -> meta_probability (최종 abnormal_probability)
            x_arr = np.asarray(features, dtype=np.float64)
            scaled = (x_arr - self.meta_params["scaler_mean"]) / self.meta_params["scaler_scale"]
            meta_logit = float(np.dot(scaled, self.meta_params["coef"]) + self.meta_params["intercept"])
            meta_probability = _stable_sigmoid(meta_logit)
        return {
            "v31_probability": v31_probability,
            "v31_logit": v31_logit,
            "features": features,
            "meta_logit": meta_logit,
            "meta_probability": meta_probability,
        }

    @staticmethod
    def _normalize_meta_params(params: dict[str, Any] | None) -> dict[str, np.ndarray | float] | None:
        # 체크포인트에 저장된 sklearn 파라미터를 추론에 바로 쓸 numpy 형태로 표준화한다.
        if params is None:
            return None
        coef = np.asarray(params["coef"], dtype=np.float64)
        # sklearn 이진 분류 계수는 [1, N] 형태로 저장되므로 1차원 [N]으로 편다.
        if coef.ndim == 2:
            if coef.shape[0] != 1:
                raise ValueError(f"Expected binary coef shape [1,N], got {coef.shape}")
            coef = coef[0]
        intercept = np.asarray(params["intercept"], dtype=np.float64)
        if intercept.size != 1:
            raise ValueError(f"Expected one intercept, got {intercept.shape}")
        scaler_scale = np.asarray(params["scaler_scale"], dtype=np.float64)
        # 분산이 0인 특징(상수열)은 scale=0 -> 0으로 나누기 방지를 위해 1로 치환.
        scaler_scale = np.where(scaler_scale == 0.0, 1.0, scaler_scale)
        return {
            "scaler_mean": np.asarray(params["scaler_mean"], dtype=np.float64),
            "scaler_scale": scaler_scale,
            "coef": coef,
            "intercept": float(intercept.reshape(-1)[0]),
        }


def _stable_sigmoid(value: float) -> float:
    # 수치 안정적 sigmoid: exp 오버플로를 피하려고 부호에 따라 식을 나눠 계산한다.
    if value >= 0.0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


def _bf16_autocast(x: torch.Tensor, *, enabled: bool):
    # bfloat16 autocast는 'CUDA에서 bf16 지원 + 호출자가 명시적으로 켰을 때'만 실제 활성화.
    # 그 외(CPU, 미지원 GPU, enabled=False)에는 autocast가 사실상 no-op이 된다.
    amp_active = bool(
        enabled
        and x.is_cuda
        and torch.cuda.is_available()
        and torch.cuda.is_bf16_supported()
    )
    device_type = "cuda" if x.is_cuda else "cpu"
    return torch.autocast(device_type=device_type, dtype=torch.bfloat16, enabled=amp_active)
