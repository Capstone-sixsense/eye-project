from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image

from drscreen.data.transforms import FundusPreprocess, build_eval_transform
from drscreen.infer.pipeline import InferenceResult, run_single_image_inference
from drscreen.models.build import build_model
from drscreen.models.profiles import get_model_profile
from drscreen.quality.quickqual import QuickQualAssessor
from drscreen.settings import (
    ensure_runtime_directories,
    load_app_config,
    merge_dicts,
    resolve_project_path,
)
from drscreen.train.runner import resolve_device
from drscreen.xai.gradcam import generate_gradcam


def _resolve_config_context(config_path: str | Path) -> tuple[Path, Path, dict[str, Any]]:
    resolved_config_path = Path(config_path).resolve()
    project_root = resolved_config_path.parents[1]
    base_path = None
    candidate_base = resolved_config_path.parent / "base.yaml"
    if resolved_config_path.name != "base.yaml" and candidate_base.exists():
        base_path = candidate_base
    config = load_app_config(resolved_config_path, base_path=base_path)
    ensure_runtime_directories(config, project_root)
    return resolved_config_path, project_root, config


def _build_effective_infer_config(
    runtime_config: dict[str, Any],
    checkpoint: dict[str, Any],
) -> dict[str, Any]:
    checkpoint_config = checkpoint.get("config")
    effective_config = runtime_config
    if isinstance(checkpoint_config, dict):
        effective_config = merge_dicts(checkpoint_config, runtime_config)

    effective_config = merge_dicts(
        effective_config,
        {
            "model": {
                "architecture": checkpoint.get(
                    "architecture",
                    effective_config["model"]["architecture"],
                ),
                "num_outputs": checkpoint.get(
                    "num_outputs",
                    effective_config["model"]["num_outputs"],
                ),
                "pretrained": False,
            },
            "labels": {
                "names": checkpoint.get(
                    "label_names",
                    effective_config["labels"]["names"],
                )
            },
        },
    )
    return effective_config


def _sanitize_stem(name: str) -> str:
    stem = Path(name or "upload").stem or "upload"
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", stem)
    return cleaned[:80] or "upload"


def _build_timestamped_path(directory: Path, stem: str, suffix: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    return directory / f"{stem}_{timestamp}{suffix}"


def _build_retina_mask(image: Image.Image) -> np.ndarray:
    rgb = np.asarray(image.convert("RGB"))
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    _, thresholded = cv2.threshold(gray, 8, 255, cv2.THRESH_BINARY)

    min_dim = min(gray.shape[:2])
    kernel_size = max(3, min(11, (min_dim // 50) * 2 + 1))
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    cleaned = cv2.morphologyEx(thresholded, cv2.MORPH_CLOSE, kernel)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(cleaned)
    if num_labels <= 1:
        return np.ones(gray.shape, dtype=np.float32)

    largest_label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return (labels == largest_label).astype(np.float32)


def _render_gradcam_overlay(image: Image.Image, heatmap: torch.Tensor) -> Image.Image:
    normalized = heatmap.detach().cpu().clamp(0.0, 1.0).numpy().astype(np.float32)
    retina_mask = _build_retina_mask(image)
    resized = cv2.resize(
        normalized,
        dsize=image.size,
        interpolation=cv2.INTER_LINEAR,
    )
    resized *= retina_mask

    # Suppress weak activations so the overlay highlights only strong evidence regions.
    threshold = 0.45
    emphasized = np.clip((resized - threshold) / (1.0 - threshold), 0.0, 1.0)
    emphasized = np.power(emphasized, 0.8, dtype=np.float32)

    heat_uint8 = np.uint8(np.clip(resized, 0.0, 1.0) * 255.0)
    heat_bgr = cv2.applyColorMap(heat_uint8, cv2.COLORMAP_TURBO)
    heat_rgb = cv2.cvtColor(heat_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)

    original = np.asarray(image.convert("RGB"), dtype=np.float32)
    alpha_mask = emphasized[..., None] * 0.82
    overlay = (original * (1.0 - alpha_mask)) + (heat_rgb * alpha_mask)
    return Image.fromarray(np.uint8(np.clip(overlay, 0.0, 255.0)))


@dataclass(slots=True)
class SavedInferenceArtifacts:
    prediction_path: Path | None
    heatmap_path: Path | None


@dataclass(slots=True)
class SingleImagePrediction:
    result: InferenceResult
    payload: dict[str, Any]
    original_image: Image.Image
    heatmap_overlay: Image.Image | None
    saved: SavedInferenceArtifacts


@dataclass(slots=True)
class InferenceSession:
    config_path: Path
    project_root: Path
    config: dict[str, Any]
    checkpoint_path: Path
    device: torch.device
    model: torch.nn.Module
    eval_transform: Any
    label_names: tuple[str, ...]
    prediction_dir: Path
    heatmap_dir: Path
    quality_assessor: QuickQualAssessor | None
    preprocessor: FundusPreprocess | None

    @classmethod
    def from_config_path(
        cls,
        config_path: str | Path,
        *,
        checkpoint_path: str | Path | None = None,
    ) -> InferenceSession:
        resolved_config_path, project_root, config = _resolve_config_context(config_path)
        resolved_checkpoint_path = resolve_project_path(
            project_root,
            checkpoint_path or config["infer"]["checkpoint_path"],
        )
        checkpoint = torch.load(resolved_checkpoint_path, map_location="cpu")
        effective_config = _build_effective_infer_config(config, checkpoint)

        device_name = str(
            effective_config.get("infer", {}).get("device")
            or effective_config.get("train", {}).get("device", "cpu")
        )
        device = resolve_device(device_name)
        architecture = str(effective_config["model"]["architecture"])
        num_outputs = int(effective_config["model"]["num_outputs"])

        model = build_model(
            architecture,
            pretrained=False,
            num_outputs=num_outputs,
            use_attention=bool(effective_config["model"].get("use_attention", False)),
        ).to(device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()

        profile = get_model_profile(architecture)
        data_cfg = effective_config["data"]
        infer_cfg = effective_config.get("infer", {})
        use_preprocessing = bool(
            infer_cfg.get("use_preprocessing", data_cfg.get("use_preprocessing", False))
        )
        eval_transform = build_eval_transform(
            crop_size=int(data_cfg["image_size"]),
            resize_size=int(data_cfg["resize_size"]),
            interpolation=profile.interpolation,
            mean=profile.mean,
            std=profile.std,
            use_preprocessing=False,
        )
        preprocess_size = int(data_cfg.get("preprocess_size", 0)) or None
        preprocessor = FundusPreprocess(output_size=preprocess_size) if use_preprocessing else None
        quality_assessor = QuickQualAssessor.from_config(effective_config, project_root, device)

        prediction_dir = resolve_project_path(project_root, effective_config["infer"]["prediction_dir"])
        heatmap_dir = resolve_project_path(project_root, effective_config["infer"]["heatmap_dir"])
        prediction_dir.mkdir(parents=True, exist_ok=True)
        heatmap_dir.mkdir(parents=True, exist_ok=True)

        return cls(
            config_path=resolved_config_path,
            project_root=project_root,
            config=effective_config,
            checkpoint_path=resolved_checkpoint_path,
            device=device,
            model=model,
            eval_transform=eval_transform,
            label_names=tuple(effective_config["labels"]["names"]),
            prediction_dir=prediction_dir,
            heatmap_dir=heatmap_dir,
            quality_assessor=quality_assessor,
            preprocessor=preprocessor,
        )

    def predict_image_path(
        self,
        image_path: str | Path,
        *,
        save_outputs: bool = True,
    ) -> SingleImagePrediction:
        resolved_image_path = Path(image_path).resolve()
        with Image.open(resolved_image_path) as image:
            prediction = self.predict_pil_image(
                image,
                image_name=resolved_image_path.name,
                save_outputs=save_outputs,
            )
        return prediction

    def predict_image_bytes(
        self,
        image_bytes: bytes,
        *,
        image_name: str = "upload.png",
        save_outputs: bool = True,
    ) -> SingleImagePrediction:
        with Image.open(BytesIO(image_bytes)) as image:
            prediction = self.predict_pil_image(
                image,
                image_name=image_name,
                save_outputs=save_outputs,
            )
        return prediction

    def predict_pil_image(
        self,
        image: Image.Image,
        *,
        image_name: str = "upload.png",
        save_outputs: bool = True,
    ) -> SingleImagePrediction:
        original_image = image.convert("RGB")
        raw_image = np.asarray(original_image)
        if self.preprocessor is not None:
            original_image = self.preprocessor(original_image)
        image_tensor = self.eval_transform(original_image).to(self.device)

        quality_cfg = self.config["quality"]
        result = run_single_image_inference(
            model=self.model,
            image_tensor=image_tensor,
            raw_image=raw_image,
            label_names=self.label_names,
            blur_threshold=float(quality_cfg["blur_score_min"]),
            brightness_threshold=float(quality_cfg["brightness_mean_min"]),
            low_quality_action=str(quality_cfg["action_on_low_quality"]),
            quality_assessor=self.quality_assessor,
        )

        heatmap_overlay = None
        try:
            gradcam = generate_gradcam(self.model, image_tensor.unsqueeze(0))
            heatmap_overlay = _render_gradcam_overlay(original_image, gradcam.heatmap[0])
        except Exception:
            heatmap_overlay = None

        saved = SavedInferenceArtifacts(prediction_path=None, heatmap_path=None)
        if save_outputs:
            saved = self._save_outputs(
                image_name=image_name,
                result=result,
                heatmap_overlay=heatmap_overlay,
            )

        payload = result.to_dict()
        payload["checkpoint_path"] = str(self.checkpoint_path)
        payload["prediction_path"] = str(saved.prediction_path) if saved.prediction_path else None
        payload["heatmap_path"] = str(saved.heatmap_path) if saved.heatmap_path else None

        return SingleImagePrediction(
            result=result,
            payload=payload,
            original_image=original_image,
            heatmap_overlay=heatmap_overlay,
            saved=saved,
        )

    def _save_outputs(
        self,
        *,
        image_name: str,
        result: InferenceResult,
        heatmap_overlay: Image.Image | None,
    ) -> SavedInferenceArtifacts:
        stem = _sanitize_stem(image_name)
        prediction_path = _build_timestamped_path(self.prediction_dir, stem, ".json")
        heatmap_path = None

        if heatmap_overlay is not None:
            heatmap_path = _build_timestamped_path(self.heatmap_dir, stem, ".png")
            heatmap_overlay.save(heatmap_path)

        payload = result.to_dict()
        payload["checkpoint_path"] = str(self.checkpoint_path)
        payload["heatmap_path"] = str(heatmap_path) if heatmap_path else None
        prediction_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

        return SavedInferenceArtifacts(
            prediction_path=prediction_path,
            heatmap_path=heatmap_path,
        )
