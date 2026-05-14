from __future__ import annotations

import os, shutil, logging, datetime, time
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.responses import Response
from fastapi.concurrency import run_in_threadpool
from PIL import Image

from drscreen.infer.service import InferenceSession

from make_result_img import render_report_image_bytes
from models.quickqual_wrapper import QuickQualWrapper
import traceback

import history
import crypto  # 키 검증을 위해 import만 해도 충분 (모듈 로드 시 검증)
import io


_DEFAULT_CONFIG_PATH = "/ai/configs/base.yaml"
UPLOAD_DIR = history.UPLOAD_DIR # "storage"
RESULTS_DIR = history.RESULTS_DIR # "results"

# 'bad' 확률이 이 값을 넘으면 경고 (응답에 포함). 거부하려면 REJECT_BAD_QUALITY=True
QUICKQUAL_BAD_THRESHOLD = float(os.environ.get("QUICKQUAL_BAD_THRESHOLD", "0.7"))
REJECT_BAD_QUALITY = os.environ.get("REJECT_BAD_QUALITY", "false").lower() == "true"

_session: InferenceSession | None = None
_session_error: str | None = None
_quickqual: QuickQualWrapper | None = None
_quickqual_error: str | None = None

logging.basicConfig(
    filename="server_errors.log",
    level=logging.ERROR,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


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
    """서버 시작/종료 시 모델 로드/해제."""
    del app
    global _session, _session_error, _quickqual, _quickqual_error

    config_path = os.environ.get("FUNDUS_CONFIG_PATH", _DEFAULT_CONFIG_PATH)
    checkpoint_path = os.environ.get("FUNDUS_CHECKPOINT_PATH") or None
    #개발중인 AI 모델
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
        
    #QuickQual 모델
    try:
        svm_filename = os.environ.get(
            "QUICKQUAL_SVM_FILENAME", "quickqual_dn121_512.pkl"
        )
        _quickqual = QuickQualWrapper(svm_filename=svm_filename)
        _quickqual_error = None
    except Exception as exc:
        _quickqual = None
        _quickqual_error = f"{type(exc).__name__}: {exc}"
        logger.error(f"[lifespan] QuickQual load failed:\n{traceback.format_exc()}")

    yield
    _session = None
    _quickqual = None


os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

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


@app.get("/")
def read_root() -> dict[str, str]:
    return {"message": "eye-project backend is running"}


@app.get("/health")
def health() -> dict[str, Any]:
    payload: dict[str, Any] = {
        "diagnosis_model": "ok" if _session else "not_ready",
        "quickqual": "ok" if _quickqual else "not_ready",
    }
    if _session is None:
        payload["diagnosis_error"] = _session_error
    if _quickqual is None:
        payload["quickqual_error"] = _quickqual_error
    status = 200 if (_session and _quickqual) else 503
    return JSONResponse(status_code=status, content=payload)

@app.post("/analyze")
async def analyze(image: UploadFile = File(...)) -> dict[str, Any]:
    name = image.filename or "upload.png"

    print(
        f"[analyze] 요청 도착 (filename={image.filename!r}, 업로드 수신·추론 시작)",
        flush=True,
    )

    if _session is None:
        raise HTTPException(
            status_code=503,
            detail=_session_error or "AI 모델이 준비되지 않았습니다.",
        )
    if _quickqual is None:
        raise HTTPException(
            status_code=503,
            detail=_quickqual_error or "QuickQual 모델이 준비되지 않았습니다.",
        )
    
    # 분석 ID 발급 (이력 조회 시 raw, report, metadata를 한 번에 묶어주는 키)
    record_id = history.make_record_id()

    # 원본 확장자 보존 (raw 파일명에 사용)
    _, ext = os.path.splitext(image.filename or "upload.png")
    ext = ext.lstrip(".").lower() or "png"

    #타이머 시작
    t0 = time.perf_counter()

    try:
        # 이미지 수신 및 원본 저장
        content = await image.read()
        print(f"[analyze] 수신: filename={name!r}, bytes={len(content)}", flush=True)

        raw_path = history.save_raw_image(record_id, content, ext=ext)
        print(f"[analyze] 원본 암호화 저장: {raw_path}", flush=True)

        # 이미지 로드 /  메모리에서 PIL 로드 (디스크 평문 경로 거치지 않음)
        try:
            raw_img = Image.open(io.BytesIO(content)).convert("RGB")
        except Exception:
            raise HTTPException(status_code=400, detail="유효한 이미지 파일이 아닙니다.")

        #QuickQual 전처리 + 품질 평가
        t_qq = time.perf_counter()
        preprocessed_img, quality = _quickqual.preprocess_and_score(raw_img)
        print(
            f"[analyze] QuickQual 완료 ({time.perf_counter() - t_qq:.2f}s) "
            f"label={quality['label']} bad={quality['bad']:.3f}",
            flush=True,
        )

        #품질 게이트 ----
        quality_warning = None
        if quality["bad"] >= QUICKQUAL_BAD_THRESHOLD:
            quality_warning = (
                f"이미지 품질이 낮습니다 (bad 확률={quality['bad']:.2f}). "
                "결과 신뢰도가 떨어질 수 있습니다."
            )
            if REJECT_BAD_QUALITY:
                raise HTTPException(
                    status_code=422,
                    detail={
                        "code": "low_image_quality",
                        "message": quality_warning,
                        "quality": quality,
                    },
                )
        
        #타이머 종료
        print(
            f"[analyze] 전처리 완료 (총 {time.perf_counter() - t0:.2f}s)",
            flush=True,
        )

        """
        # AI 추론 (CPU + EfficientNet 등은 여기서 대부분의 시간 소요)
        with open(proc_path, "rb") as f:
            pred = _session.predict_image_bytes(f.read(), image_name=name)
        """
        # 진단 추론 — 전처리된 이미지를 바이트로 직접 전달 (proc_path 디스크 쓰기 제거)
        proc_buf = io.BytesIO()
        preprocessed_img.save(proc_buf, format="PNG")
        pred = _session.predict_image_bytes(proc_buf.getvalue(), image_name=name)

        # 리포트 이미지 합성
        # heatmap_overlay가 없으면 전처리된 원본 이미지를 fallback으로 사용
        #ai_image = pred.heatmap_overlay if pred.heatmap_overlay is not None else Image.open(proc_path)
        ai_image = pred.heatmap_overlay if pred.heatmap_overlay is not None else preprocessed_img
        report_bytes = render_report_image_bytes(ai_image)
        report_path = history.save_report_image(record_id, report_bytes)


        prob = pred.payload.get("abnormal_probability", 0.0)
        decision_threshold = pred.payload.get("decision_threshold")
        eval_metrics = pred.payload.get("eval_metrics")
        metrics = dict(eval_metrics) if isinstance(eval_metrics, dict) else {}
        metrics.setdefault("decision_threshold", decision_threshold)
        evidence = {
            "evidence_type": pred.payload.get("evidence_type"),
            "heatmap_path": pred.payload.get("heatmap_path"),
            "lesion_map_path": pred.payload.get("lesion_map_path"),
            "lesion_summary": pred.payload.get("lesion_summary"),
            "evidence_warning": pred.payload.get("evidence_warning"),
            "xai_error_code": pred.payload.get("xai_error_code"),
            "xai_no_region": pred.payload.get("xai_no_region"),
        }

        # 이력 메타데이터 저장(URL은 동적 엔드포인트 경로)
        history.save_metadata(
            record_id,
            original_filename=name,
            raw_url=f"/image/raw/{record_id}",       # 동적 엔드포인트
            report_url=f"/image/report/{record_id}", # 동적 엔드포인트
            label=pred.payload.get("predicted_label"),
            abnormal_probability=prob,
            quality=quality,
            metrics=metrics,
            evidence=evidence,
        )


        print(f"[analyze] 완료: {time.perf_counter() - t0:.1f}s", flush=True)
        return {
            "status": "success",
            "id": record_id,
            "label": pred.payload.get("predicted_label"),
            "abnormal_probability": prob,
            "decision_threshold": decision_threshold,
            "metrics": metrics,
            "eval_metrics": metrics,
            "evidence_type": evidence["evidence_type"],
            "heatmap_path": evidence["heatmap_path"],
            "lesion_map_path": evidence["lesion_map_path"],
            "lesion_summary": evidence["lesion_summary"],
            "evidence_warning": evidence["evidence_warning"],
            "xai_error_code": pred.payload.get("xai_error_code"),
            "xai_no_region": pred.payload.get("xai_no_region"),
            "quality": quality,
            "quality_warning": quality_warning,
            "report_url": f"/image/report/{record_id}",
            "raw_url": f"/image/raw/{record_id}",
            "original_url": f"/image/raw/{record_id}",  # 기존 키 호환용
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[analyze] {name}: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail="분석 실패") from e


# ---------------------------------------------------------------
# 이미지 스트리밍 (복호화 후 응답)
# ---------------------------------------------------------------
@app.get("/image/raw/{record_id}")
def serve_raw_image(record_id: str) -> Response:
    """복호화된 raw 이미지 바이트 응답."""
    if not history.RECORD_ID_PATTERN.match(record_id):
        raise HTTPException(status_code=404, detail="raw image not found")
    result = history.load_raw_bytes(record_id)
    if result is None:
        raise HTTPException(status_code=404, detail="raw image not found")
    data, ext = result
    media_type = {
        "png": "image/png",
        "jpg": "image/jpeg",
        "jpeg": "image/jpeg",
        "webp": "image/webp",
        "bmp": "image/bmp",
        "tif": "image/tiff",
        "tiff": "image/tiff",
    }.get(ext.lower(), "application/octet-stream")
    return Response(content=data, media_type=media_type)

@app.get("/image/report/{record_id}")
def serve_report_image(record_id: str) -> Response:
    """복호화된 분석 결과 이미지 응답 (PNG)."""
    if not history.RECORD_ID_PATTERN.match(record_id):
        raise HTTPException(status_code=404, detail="report image not found")
    data = history.load_report_bytes(record_id)
    if data is None:
        raise HTTPException(status_code=404, detail="report image not found")
    return Response(content=data, media_type="image/png")



# ---------------------------------------------------------------
# history_list  (GET /history)
# ---------------------------------------------------------------
# 분석 이력 목록을 최신순으로 반환하는 REST 엔드포인트.
#
# [역할]
#   - 프론트의 "이력 조회" 첫 화면에서 호출한다.
#   - 이미지나 메트릭을 모두 포함한 요약 리스트를 한 번에 내려준다.
#
# [동작 순서]
#   1) 쿼리스트링(limit, offset) 검증.
#      - limit는 1..200 범위로 제한해 한 번의 응답이 너무 커지지 않도록 한다.
#      - offset은 음수 차단.
#   2) history.list_records(limit, offset) 으로 실제 데이터 수집.
#   3) history.count_records() 로 전체 건수 계산
#      → 프론트가 "전체 87건 중 1~50" 같은 페이지 인디케이터를 만들 수 있게 한다.
#   4) 응답 dict를 그대로 반환 (FastAPI가 JSON 직렬화).
#
# [응답 스키마]
#   {
#     "total": 87,
#     "limit": 50,
#     "offset": 0,
#     "items": [ <save_metadata가 저장한 dict들> ]
#   }
#
# [에러]
#   - limit 범위 위반: 400 limit must be 1..200
#   - offset 음수    : 400 offset must be >= 0
@app.get("/history")
def history_list(limit: int = 50, offset: int = 0) -> dict[str, Any]:
    """분석 이력 목록 (최신순). 페이지네이션 지원."""
    if limit < 1 or limit > 200:
        raise HTTPException(status_code=400, detail="limit must be 1..200")
    if offset < 0:
        raise HTTPException(status_code=400, detail="offset must be >= 0")

    records = history.list_records(limit=limit, offset=offset)
    return {
        "total": history.count_records(),
        "limit": limit,
        "offset": offset,
        "items": records,
    }

@app.get("/history/{record_id}")
def history_detail(record_id: str) -> dict[str, Any]:
    """특정 분석 이력 단건 조회."""
    if not history.RECORD_ID_PATTERN.match(record_id):
        raise HTTPException(status_code=404, detail="record not found")
    record = history.load_record(record_id)
    if record is None:
        raise HTTPException(status_code=404, detail="record not found")
    return record

@app.delete("/history/{record_id}")
def history_delete(record_id: str) -> dict[str, Any]:
    """특정 분석 이력과 관련 파일(raw, report, metadata) 일괄 삭제."""
    if not history.RECORD_ID_PATTERN.match(record_id):
        raise HTTPException(status_code=404, detail="record not found")
    record = history.load_record(record_id)
    if record is None:
        raise HTTPException(status_code=404, detail="record not found")
    history.delete_record_files(record_id)
    return {"status": "deleted", "id": record_id}
