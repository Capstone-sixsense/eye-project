from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import JSONResponse

from drscreen.infer.service import InferenceSession


_DEFAULT_CONFIG_PATH = "/ai/configs/base.yaml"

_session: InferenceSession | None = None
_session_error: str | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _session, _session_error
    config_path = os.environ.get("FUNDUS_CONFIG_PATH", _DEFAULT_CONFIG_PATH)
    checkpoint_path = os.environ.get("FUNDUS_CHECKPOINT_PATH") or None
    try:
        _session = InferenceSession.from_config_path(config_path, checkpoint_path=checkpoint_path)
    except FileNotFoundError as exc:
        _session_error = str(exc)
    except Exception as exc:
        _session_error = str(exc)
    yield
    _session = None


app = FastAPI(title="eye-project backend", lifespan=lifespan)


@app.get("/")
def read_root() -> dict[str, str]:
    return {"message": "eye-project backend is running"}


@app.get("/health")
def health() -> dict[str, Any]:
    if _session is not None:
        return {
            "status": "ok",
            "config_path": str(_session.config_path),
            "checkpoint_path": str(_session.checkpoint_path),
        }
    return JSONResponse(
        status_code=503,
        content={"status": "model_not_ready", "detail": _session_error},
    )


@app.post("/predict")
async def predict(image: UploadFile = File(...)) -> dict[str, Any]:
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
