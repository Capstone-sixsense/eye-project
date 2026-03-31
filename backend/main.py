from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from drscreen.infer.service import InferenceSession

os.makedirs("storage", exist_ok=True)
os.makedirs("results", exist_ok=True)
app.mount("/storage", StaticFiles(directory="storage"), name="storage") # 원본 접근용 추가
app.mount("/results", StaticFiles(directory="results"), name="results") # 분석 결과 접근용

_DEFAULT_CONFIG_PATH = "/ai/configs/base.yaml"

_session: InferenceSession | None = None
_session_error: str | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    del app
    global _session, _session_error
    config_path = os.environ.get("FUNDUS_CONFIG_PATH", _DEFAULT_CONFIG_PATH)
    checkpoint_path = os.environ.get("FUNDUS_CHECKPOINT_PATH") or None
    try:
        _session = InferenceSession.from_config_path(config_path, checkpoint_path=checkpoint_path)
        _session_error = None
    except FileNotFoundError as exc:
        _session = None
        _session_error = str(exc)
    except Exception as exc:
        _session = None
        _session_error = str(exc)
    yield
    _session = None

#테스트를 위한 가상의 AI 모델 인터페이스
def run_ai_inference(image_path: str):
    """
    AI 모델 분석 시뮬레이션
    Returns: label, probability, heatmap_image_object, metrics
    """
    # 실제 모델이 반환할 데이터 예시
    predicted_label = "Abnormal (DR)" # 망막병증 의심
    abnormal_probability = 0.88
    quality_warning = "None"
    quality_grade = "Good"
    
    # Grad-CAM 결과물이라고 가정하고 원본을 불러와 가공 (테스트용)
    heatmap_img = Image.open(image_path).convert("RGB")
    
    # 가상의 성능 지표 (합성용)
    mock_metrics = {
        'accuracy': 0.95, 'precision': 0.92, 'recall': 0.96, 'specificity': 0.94, 'f1': 0.94
    }
    
    return {
        "label": predicted_label,
        "probability": abnormal_probability,
        "warning": quality_warning,
        "grade": quality_grade,
        "heatmap": heatmap_img,
        "metrics": mock_metrics
    }

app = FastAPI(title="eye-project backend", lifespan=lifespan)


def _health_payload() -> dict[str, Any]:
    if _session is None:
        raise HTTPException(status_code=503, detail=f"Model not ready: {_session_error}")
    return {
        "status": "ok",
        "config_path": str(_session.config_path),
        "checkpoint_path": str(_session.checkpoint_path),
    }


async def _predict_from_upload(image: UploadFile) -> dict[str, Any]:
    if _session is None:
        raise HTTPException(status_code=503, detail=f"Model not ready: {_session_error}")

    image_bytes = await image.read()
    try:
        prediction = _session.predict_image_bytes(
            image_bytes,
            image_name=image.filename or "upload.png",
        )
    except Exception as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    return prediction.payload


@app.get("/")
def read_root() -> dict[str, str]:
    return {"message": "eye-project backend is running"}


@app.get("/health")
def health() -> dict[str, Any]:
    if _session is not None:
        return _health_payload()
    return JSONResponse(
        status_code=503,
        content={"status": "model_not_ready", "detail": _session_error},
    )


@app.post("/predict")
async def predict(image: UploadFile = File(...)) -> dict[str, Any]:
    return await _predict_from_upload(image)


@app.post("/analyze")
async def analyze(image: UploadFile = File(...)) -> dict[str, Any]:
    return await _predict_from_upload(image)
