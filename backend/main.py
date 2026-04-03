from __future__ import annotations

import os, shutil, logging, datetime
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from PIL import Image

from drscreen.infer.service import InferenceSession

from image_analyzer import check_image_quality, PassNonPass, resize_image_high_quality
from make_result_img import create_medical_report_image





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


os.makedirs("storage", exist_ok=True)
os.makedirs("results", exist_ok=True)
app.mount("/storage", StaticFiles(directory="storage"), name="storage") # 원본 접근용 추가
app.mount("/results", StaticFiles(directory="results"), name="results") # 분석 결과 접근용

# 로그 설정
logging.basicConfig(
    filename='server_errors.log',
    level=logging.ERROR,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

@app.post("/analyze")
async def analyze(image: UploadFile = File(...)) -> dict[str, Any]:
    if _session is None:
        raise HTTPException(status_code=503, detail="AI 모델 로딩 중입니다.")

    UPLOAD_DIR = "storage"
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    
    # 파일 경로 설정
    raw_path = os.path.join(UPLOAD_DIR, f"raw_{image.filename}")
    proc_path = os.path.join(UPLOAD_DIR, image.filename)

    try:
        # 전처리 및 저장
        content = await image.read()
        with open(raw_path, "wb") as f: f.write(content)
        resize_image_high_quality(raw_path, proc_path, (448, 448))

        # 품질 필터링 
        q_res = check_image_quality(UPLOAD_DIR, image.filename)
        if not PassNonPass(q_res)['is_acceptable']:
            return {"status": "fail", "message": "이미지 품질 미달", "details": q_res}

        # AI 추론
        with open(proc_path, "rb") as f:
            pred = _session.predict_image_bytes(f.read(), image_name=image.filename)

        # 리포트 이미지 합성
        # heatmap_overlay가 없으면 전처리된 원본 이미지를 fallback으로 사용
        ai_image = pred.heatmap_overlay if pred.heatmap_overlay is not None else Image.open(proc_path)
        prob = pred.payload.get("abnormal_probability", 0.0)
        metrics = {
            "accuracy": prob,
            "precision": prob,
            "recall": prob,
            "specificity": 1.0 - prob,
            "f1": prob,
        }
        report_path = create_medical_report_image(
            original_filename=image.filename,
            ai_image=ai_image,
            metrics=metrics,
        )

        return {
            "status": "success",
            "label": pred.payload.get("predicted_label"),
            "abnormal_probability": prob,
            "report_url": report_path,
            "original_url": raw_path,
        }

    except Exception as e:
        logging.error(f"Error: {e}")
        raise HTTPException(status_code=500, detail="분석 실패")
    finally:
        if os.path.exists(proc_path): os.remove(proc_path) # 임시파일 정리

    #return await pred
