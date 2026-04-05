from __future__ import annotations

import os, shutil, logging, datetime, time
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from PIL import Image

from drscreen.infer.service import InferenceSession

from image_analyzer import check_image_quality, PassNonPass, resize_image_high_quality
from make_result_img import create_medical_report_image





_DEFAULT_CONFIG_PATH = "/ai/configs/base.yaml"

_session: InferenceSession | None = None
_session_error: str | None = None


def _configure_cpu_threads() -> None:
    """Docker CPU 추론 시 스레드 수 (OMP/MKL은 compose에서, PyTorch는 여기서)."""
    try:
        import torch

        inter = os.environ.get("TORCH_NUM_INTEROP_THREADS", "").strip()
        if inter.isdigit() and int(inter) > 0:
            torch.set_num_interop_threads(int(inter))
        raw = os.environ.get("TORCH_NUM_THREADS", "").strip()
        if raw.isdigit() and int(raw) > 0:
            torch.set_num_threads(int(raw))
    except Exception:
        pass


@asynccontextmanager
async def lifespan(app: FastAPI):
    del app
    global _session, _session_error
    config_path = os.environ.get("FUNDUS_CONFIG_PATH", _DEFAULT_CONFIG_PATH)
    checkpoint_path = os.environ.get("FUNDUS_CHECKPOINT_PATH") or None

    try:
        _configure_cpu_threads()
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

# Flutter 웹(예: localhost:8080) → API(8000) 교차 출처: 브라우저가 JSON·이미지 응답을 읽으려면 필요
_cors_origins = os.environ.get(
    "CORS_ALLOW_ORIGINS",
    "http://localhost:8080,http://127.0.0.1:8080,http://localhost:3000,http://127.0.0.1:3000",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in _cors_origins.split(",") if o.strip()],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


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
    print(
        f"[analyze] 요청 도착 (filename={image.filename!r}, 업로드 수신·추론 시작)",
        flush=True,
    )
    if _session is None:
        raise HTTPException(
            status_code=503,
            detail=_session_error or "AI 모델이 준비되지 않았습니다.",
        )

    UPLOAD_DIR = "storage"
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    
    # 파일 경로 설정
    raw_path = os.path.join(UPLOAD_DIR, f"raw_{image.filename}")
    proc_path = os.path.join(UPLOAD_DIR, image.filename)

    try:
        t0 = time.perf_counter()
        # 전처리 및 저장
        content = await image.read()
        name = image.filename or "upload"
        print(f"[analyze] 수신: filename={name!r}, bytes={len(content)}", flush=True)
        with open(raw_path, "wb") as f: f.write(content)
        resize_image_high_quality(raw_path, proc_path, (448, 448))

        # CleanVision(Imagelab)은 이미지 1장도 수십 초~수 분 걸릴 수 있음 → 개발 시 SKIP_CLEANVISION=1
        skip_cv = os.environ.get("SKIP_CLEANVISION", "").lower() in ("1", "true", "yes")
        if not skip_cv:
            q_res = check_image_quality(UPLOAD_DIR, name)
            if not PassNonPass(q_res)["is_acceptable"]:
                return {"status": "fail", "message": "이미지 품질 미달", "details": q_res}
        else:
            print("[analyze] SKIP_CLEANVISION=1 — CleanVision 품질 검사 생략", flush=True)

        # AI 추론 (CPU + EfficientNet 등은 여기서 대부분의 시간 소요)
        with open(proc_path, "rb") as f:
            pred = _session.predict_image_bytes(f.read(), image_name=name)

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
            original_filename=name,
            ai_image=ai_image,
            metrics=metrics,
        )

        print(f"[analyze] 완료: {time.perf_counter() - t0:.1f}s", flush=True)
        return {
            "status": "success",
            "label": pred.payload.get("predicted_label"),
            "abnormal_probability": prob,
            "report_url": report_path.replace("\\", "/"),
            "original_url": raw_path.replace("\\", "/"),
        }

    except Exception as e:
        logging.error(f"Error: {e}")
        raise HTTPException(status_code=500, detail="분석 실패")
    finally:
        if os.path.exists(proc_path): os.remove(proc_path) # 임시파일 정리

    #return await pred
