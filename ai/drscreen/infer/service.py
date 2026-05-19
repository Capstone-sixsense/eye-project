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
from drscreen.infer.payload import InferencePayload
from drscreen.infer.pipeline import InferenceResult, run_single_image_inference
from drscreen.models.build import build_model
from drscreen.models.profiles import get_model_profile
from drscreen.utils.checkpoint import load_state_from_checkpoint
from drscreen.settings import (
    build_effective_checkpoint_config,
    ensure_runtime_directories,
    find_classification_metrics_path,
    get_run_evaluation_dir,
    load_app_config,
    resolve_checkpoint_path,
    resolve_project_path,
)
from drscreen.train.model_setup import resolve_device
from drscreen.xai.gradcam import generate_gradcam
from drscreen.xai.iou import LESION_CODES


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


def _sanitize_stem(name: str) -> str:
    stem = Path(name or "upload").stem or "upload"
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", stem)
    return cleaned[:80] or "upload"


def _build_timestamped_path(directory: Path, stem: str, suffix: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    return directory / f"{stem}_{timestamp}{suffix}"


def _as_valid_threshold(value: Any) -> float | None:
    if value is None:
        return None
    try:
        threshold = float(value)
    except (TypeError, ValueError):
        return None
    if 0.0 <= threshold <= 1.0:
        return threshold
    return None


def _checkpoint_optimal_threshold(checkpoint: dict[str, Any]) -> float | None:
    for key in ("optimal_threshold", "decision_threshold"):
        threshold = _as_valid_threshold(checkpoint.get(key))
        if threshold is not None:
            return threshold
    return None


def _mean_metric(section: Any) -> float | None:
    if not isinstance(section, dict):
        return None
    return _as_valid_threshold(section.get("mean"))


def _xai_block_label(infer_cfg: dict[str, Any]) -> str:
    raw = infer_cfg.get("gradcam_target_block")
    if raw is None:
        return "default"
    if isinstance(raw, int):
        return f"block{raw}"
    value = str(raw).strip().lower()
    if not value or value in {"default", "last"}:
        return "default"
    if value.isdigit():
        return f"block{value}"
    return value


def _xai_block_index(infer_cfg: dict[str, Any]) -> int | None:
    raw = infer_cfg.get("gradcam_target_block")
    if raw is None:
        return None
    if isinstance(raw, int):
        return raw
    value = str(raw).strip().lower()
    if not value or value in {"default", "last"}:
        return None
    if value.startswith("block"):
        value = value.removeprefix("block")
    if value.isdigit():
        return int(value)
    raise ValueError(f"Unsupported gradcam_target_block: {raw!r}")


def _resolve_xai_target_layer(
    model: torch.nn.Module,
    infer_cfg: dict[str, Any],
) -> torch.nn.Module | None:
    block_index = _xai_block_index(infer_cfg)
    if block_index is None:
        return None
    blocks = getattr(model, "blocks", getattr(model, "features", None))
    if blocks is None:
        raise ValueError("Model has neither .blocks nor .features attribute")
    return blocks[block_index]


def _load_xai_eval_metrics(
    project_root: Path,
    version: str,
    infer_cfg: dict[str, Any],
) -> dict[str, Any] | None:
    block_label = _xai_block_label(infer_cfg)
    split = str(infer_cfg.get("xai_eval_split", "test"))
    method = str(infer_cfg.get("gradcam_method", "gradcam")).strip().lower() or "gradcam"
    eval_dir = get_run_evaluation_dir(project_root, version)
    candidates = [
        eval_dir / f"xai_iou_{version}_{method}_{block_label}_{split}.json",
        eval_dir / f"xai_iou_{version}_{block_label}_{split}.json",
    ]
    path = next((candidate for candidate in candidates if candidate.exists()), None)
    if path is None:
        return None

    try:
        with open(path, encoding="utf-8") as handle:
            data = json.load(handle)
    except Exception:
        return None

    aggregate = data.get("aggregate", {})
    thresholds = aggregate.get("thresholds", {}) if isinstance(aggregate, dict) else {}

    metrics: dict[str, Any] = {
        "xai_eval_split": data.get("split", split),
        "xai_eval_target_block": data.get("target_block", block_label),
        "xai_eval_n": data.get("n_images"),
        "xai_pointing_game": _mean_metric(aggregate.get("pointing_game")),
        "xai_auprc": _mean_metric(aggregate.get("auprc")),
        "xai_auc_iou": _mean_metric(aggregate.get("auc_iou")),
    }
    for top_key in ("top10", "top20", "top30"):
        top_metrics = thresholds.get(top_key, {})
        if isinstance(top_metrics, dict):
            value = _as_valid_threshold(top_metrics.get("mean_iou_union"))
            metrics[f"xai_iou_{top_key}"] = value

    return {key: value for key, value in metrics.items() if value is not None}


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


def _render_gradcam_overlay(
    image: Image.Image, heatmap: torch.Tensor
) -> tuple[Image.Image, bool]:
    normalized = heatmap.detach().cpu().clamp(0.0, 1.0).numpy().astype(np.float32)
    retina_mask = _build_retina_mask(image)
    resized = cv2.resize(
        normalized,
        dsize=image.size,
        interpolation=cv2.INTER_LINEAR,
    )
    resized *= retina_mask

    # Suppress weak activations so the overlay highlights only strong evidence regions.
    activation_threshold = 0.45
    emphasized = np.clip(
        (resized - activation_threshold) / (1.0 - activation_threshold),
        0.0,
        1.0,
    )
    emphasized = np.power(emphasized, 0.8, dtype=np.float32)

    retina_pixel_count = float(retina_mask.sum())
    if retina_pixel_count > 0:
        active_ratio = float((emphasized > 0).sum()) / retina_pixel_count
    else:
        active_ratio = 0.0
    xai_no_region = active_ratio < 0.01

    heat_uint8 = np.uint8(np.clip(resized, 0.0, 1.0) * 255.0)
    heat_bgr = cv2.applyColorMap(heat_uint8, cv2.COLORMAP_TURBO)
    heat_rgb = cv2.cvtColor(heat_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)

    original = np.asarray(image.convert("RGB"), dtype=np.float32)
    alpha_mask = emphasized[..., None] * 0.82
    overlay = (original * (1.0 - alpha_mask)) + (heat_rgb * alpha_mask)
    return Image.fromarray(np.uint8(np.clip(overlay, 0.0, 255.0))), xai_no_region


_LESION_COLORS = {
    "MA": np.array([52.0, 152.0, 219.0], dtype=np.float32),
    "HE": np.array([231.0, 76.0, 60.0], dtype=np.float32),
    "EX": np.array([241.0, 196.0, 15.0], dtype=np.float32),
    "SE": np.array([46.0, 204.0, 113.0], dtype=np.float32),
    "union": np.array([255.0, 92.0, 64.0], dtype=np.float32),
}


def _resize_probability_channels(
    lesion_prob: torch.Tensor,
    image_size: tuple[int, int],
) -> np.ndarray:
    probabilities = lesion_prob.detach().cpu().float().clamp(0.0, 1.0)
    if probabilities.ndim == 2:
        probabilities = probabilities.unsqueeze(0)
    if probabilities.ndim != 3:
        raise ValueError(f"Expected lesion probability [C,H,W], got {tuple(probabilities.shape)}")
    channels = probabilities.numpy().astype(np.float32)
    return np.stack(
        [
            cv2.resize(channel, dsize=image_size, interpolation=cv2.INTER_LINEAR)
            for channel in channels
        ],
        axis=0,
    )


def _summarize_lesion_probabilities(
    channels: np.ndarray,
    retina_mask: np.ndarray,
    threshold: float,
) -> dict[str, Any]:
    active = retina_mask > 0
    retina_pixel_count = float(active.sum())
    if retina_pixel_count <= 0:
        retina_pixel_count = float(channels.shape[-1] * channels.shape[-2])
        active = np.ones(channels.shape[-2:], dtype=bool)

    summary: dict[str, Any] = {}
    if channels.shape[0] == len(LESION_CODES):
        labels = list(LESION_CODES)
    else:
        labels = [f"channel_{idx}" for idx in range(channels.shape[0])]

    for idx, label in enumerate(labels):
        channel = channels[idx] * retina_mask
        active_values = channel[active]
        summary[label] = {
            "presence_score": float(active_values.max()) if active_values.size else 0.0,
            "mean_score": float(active_values.mean()) if active_values.size else 0.0,
            "area_ratio": float(((channel >= threshold) & active).sum()) / retina_pixel_count,
        }

    union = channels.max(axis=0) * retina_mask
    union_active = union[active]
    summary["union"] = {
        "presence_score": float(union_active.max()) if union_active.size else 0.0,
        "mean_score": float(union_active.mean()) if union_active.size else 0.0,
        "area_ratio": float(((union >= threshold) & active).sum()) / retina_pixel_count,
        "threshold": float(threshold),
    }
    return summary


def _render_lesion_overlay(
    image: Image.Image,
    lesion_prob: torch.Tensor,
    *,
    lesion_threshold: float = 0.5,
) -> tuple[Image.Image, bool, dict[str, Any], str | None]:
    retina_mask = _build_retina_mask(image)
    channels = _resize_probability_channels(lesion_prob, image.size)
    channels *= retina_mask[None, ...]
    union = channels.max(axis=0)

    summary = _summarize_lesion_probabilities(channels, retina_mask, lesion_threshold)
    union_summary = summary["union"]
    xai_no_region = (
        float(union_summary["presence_score"]) < lesion_threshold
        or float(union_summary["area_ratio"]) < 0.001
    )
    evidence_warning = "LESION_EVIDENCE_LOW_CONFIDENCE" if xai_no_region else None

    denominator = max(1.0 - lesion_threshold, 1e-6)
    emphasized = np.clip((union - lesion_threshold) / denominator, 0.0, 1.0)
    emphasized = np.power(emphasized, 0.7, dtype=np.float32)

    if channels.shape[0] == len(LESION_CODES):
        color_map = np.zeros((*union.shape, 3), dtype=np.float32)
        for idx, code in enumerate(LESION_CODES):
            color_map += channels[idx][..., None] * _LESION_COLORS[code]
        color_map = np.clip(color_map, 0.0, 255.0)
    else:
        heat_uint8 = np.uint8(np.clip(union, 0.0, 1.0) * 255.0)
        heat_bgr = cv2.applyColorMap(heat_uint8, cv2.COLORMAP_TURBO)
        color_map = cv2.cvtColor(heat_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)

    original = np.asarray(image.convert("RGB"), dtype=np.float32)
    alpha_mask = emphasized[..., None] * 0.82
    overlay = (original * (1.0 - alpha_mask)) + (color_map * alpha_mask)
    return (
        Image.fromarray(np.uint8(np.clip(overlay, 0.0, 255.0))),
        xai_no_region,
        summary,
        evidence_warning,
    )


@dataclass(slots=True)
class SavedInferenceArtifacts:
    prediction_path: Path | None
    heatmap_path: Path | None
    lesion_map_path: Path | None


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
    preprocessor: FundusPreprocess | None
    eval_metrics: dict | None
    decision_threshold: float

    @classmethod
    def from_config_path(
        cls,
        config_path: str | Path,
        *,
        checkpoint_path: str | Path | None = None,
    ) -> InferenceSession:
        resolved_config_path, project_root, config = _resolve_config_context(config_path)
        resolved_checkpoint_path = resolve_checkpoint_path(
            project_root,
            checkpoint_path or config["infer"]["checkpoint_path"],
        )
        checkpoint = torch.load(resolved_checkpoint_path, map_location="cpu", weights_only=False)
        effective_config = build_effective_checkpoint_config(config, checkpoint)

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
            attention_mode=effective_config["model"].get("attention_mode"),
            use_ibn=bool(effective_config["model"].get("use_ibn", False)),
            use_aux_seg=bool(effective_config["model"].get("use_aux_seg", False)),
            aux_seg_block=int(effective_config["model"].get("aux_seg_block", 2)),
            aux_seg_output_size=int(effective_config["model"].get("aux_seg_output_size", 512)),
            aux_seg_channels=int(effective_config["model"].get("aux_seg_channels", 1)),
            use_gated_pooling=bool(effective_config["model"].get("use_gated_pooling", False)),
            use_mil_attention=bool(effective_config["model"].get("use_mil_attention", False)),
            decoder_type=str(effective_config["model"].get("decoder_type", "single_block")),
            decoder_blocks=(
                [int(block) for block in effective_config["model"]["decoder_blocks"]]
                if effective_config["model"].get("decoder_blocks") is not None
                else None
            ),
            bagnet_patch_size=int(effective_config["model"].get("bagnet_patch_size", 33)),
            bagnet_patch_stride=int(effective_config["model"].get("bagnet_patch_stride", 8)),
            bagnet_hidden_channels=int(effective_config["model"].get("bagnet_hidden_channels", 128)),
            bagnet_depth=int(effective_config["model"].get("bagnet_depth", 4)),
            bagnet_dropout=float(effective_config["model"].get("bagnet_dropout", 0.15)),
            bagnet_aggregation=str(effective_config["model"].get("bagnet_aggregation", "mean")),
            concept_block=int(effective_config["model"].get("concept_block", 4)),
            concept_channels=int(effective_config["model"].get("concept_channels", 4)),
            concept_head_hidden_channels=(
                int(effective_config["model"]["concept_head_hidden_channels"])
                if effective_config["model"].get("concept_head_hidden_channels") is not None
                else None
            ),
            concept_dropout=float(effective_config["model"].get("concept_dropout", 0.3)),
        ).to(device)
        load_state_from_checkpoint(model, checkpoint)
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
        use_align = bool(infer_cfg.get("use_align", data_cfg.get("use_align", False)))
        preprocessor = FundusPreprocess(output_size=preprocess_size, align=use_align) if use_preprocessing else None
        prediction_dir = resolve_project_path(project_root, effective_config["infer"]["prediction_dir"])
        heatmap_dir = resolve_project_path(project_root, effective_config["infer"]["heatmap_dir"])
        prediction_dir.mkdir(parents=True, exist_ok=True)
        heatmap_dir.mkdir(parents=True, exist_ok=True)

        optimal_threshold = _checkpoint_optimal_threshold(checkpoint)
        version = str(effective_config.get("project", {}).get("version", ""))
        eval_metrics_path = find_classification_metrics_path(
            project_root,
            version,
            split_name="external_test",
            checkpoint_stem="best",
        )
        eval_metrics = None
        if eval_metrics_path.exists():
            try:
                with open(eval_metrics_path, encoding="utf-8") as _f:
                    _data = json.load(_f)
                _opt = _data.get("metrics_at_optimal_threshold", {})
                eval_optimal_threshold = _as_valid_threshold(_data.get("optimal_threshold"))
                if eval_optimal_threshold is not None and optimal_threshold is None:
                    optimal_threshold = eval_optimal_threshold
                eval_metrics = {
                    "auroc": _data.get("metrics", {}).get("auroc"),
                    "accuracy": _opt.get("accuracy"),
                    "sensitivity": _opt.get("sensitivity"),
                    "specificity": _opt.get("specificity"),
                    "precision": _opt.get("precision"),
                    "f1": _opt.get("f1"),
                    "optimal_threshold": _data.get("optimal_threshold"),
                }
            except Exception:
                eval_metrics = None

        decision_threshold = (
            optimal_threshold
            if optimal_threshold is not None
            else _as_valid_threshold(infer_cfg.get("threshold")) or 0.5
        )
        effective_config.setdefault("infer", {})["threshold"] = decision_threshold
        if eval_metrics is not None:
            eval_metrics["decision_threshold"] = decision_threshold
        xai_eval_metrics = _load_xai_eval_metrics(project_root, version, infer_cfg)
        if xai_eval_metrics:
            eval_metrics = {**(eval_metrics or {}), **xai_eval_metrics}

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
            preprocessor=preprocessor,
            eval_metrics=eval_metrics,
            decision_threshold=decision_threshold,
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
        if self.preprocessor is not None:
            original_image = self.preprocessor(original_image)
        image_tensor = self.eval_transform(original_image).to(self.device)

        result = run_single_image_inference(
            model=self.model,
            image_tensor=image_tensor,
            label_names=self.label_names,
            threshold=self.decision_threshold,
        )

        heatmap_overlay = None
        xai_error_code = None
        xai_no_region = False
        lesion_summary = None
        evidence_warning = None
        infer_cfg = self.config.get("infer", {})
        evidence_type = str(infer_cfg.get("evidence_type", "cam_research")).strip().lower()
        if evidence_type in {"lesion_segmentation", "lesion_evidence", "segmentation"}:
            evidence_type = "lesion_segmentation"
            try:
                if not hasattr(self.model, "predict_seg"):
                    raise ValueError("Model does not expose predict_seg().")
                lesion_prob = self.model.predict_seg(image_tensor.unsqueeze(0))[0]
                lesion_threshold = (
                    _as_valid_threshold(infer_cfg.get("lesion_threshold"))
                    or 0.5
                )
                (
                    heatmap_overlay,
                    xai_no_region,
                    lesion_summary,
                    evidence_warning,
                ) = _render_lesion_overlay(
                    original_image,
                    lesion_prob,
                    lesion_threshold=lesion_threshold,
                )
            except Exception:
                xai_error_code = "XAI_002"
                evidence_warning = "LESION_EVIDENCE_UNAVAILABLE"
        elif evidence_type in {"grounded_classifier", "bagnet", "patch_logits"}:
            evidence_type = "grounded_classifier"
            try:
                if not hasattr(self.model, "get_evidence_map"):
                    raise ValueError("Model does not expose get_evidence_map().")
                evidence = self.model.get_evidence_map(image_tensor.unsqueeze(0))[0, 0]
                heatmap_overlay, xai_no_region = _render_gradcam_overlay(original_image, evidence)
                evidence_warning = "GROUNDED_EVIDENCE_LOW_CONFIDENCE" if xai_no_region else None
            except Exception:
                xai_error_code = "XAI_003"
                evidence_warning = "GROUNDED_EVIDENCE_UNAVAILABLE"
        else:
            evidence_type = "cam_research"
            try:
                gradcam_method = infer_cfg.get("gradcam_method", "gradcam")
                target_layer = _resolve_xai_target_layer(
                    self.model,
                    infer_cfg,
                )
                gradcam = generate_gradcam(
                    self.model,
                    image_tensor.unsqueeze(0),
                    target_layer=target_layer,
                    method=gradcam_method,
                )
                heatmap_overlay, xai_no_region = _render_gradcam_overlay(original_image, gradcam.heatmap[0])
            except Exception:
                xai_error_code = "XAI_001"

        stem = _sanitize_stem(image_name)
        prediction_path: Path | None = None
        heatmap_path: Path | None = None
        lesion_map_path: Path | None = None
        if save_outputs:
            prediction_path = _build_timestamped_path(self.prediction_dir, stem, ".json")
            if heatmap_overlay is not None:
                heatmap_path = _build_timestamped_path(self.heatmap_dir, stem, ".png")
                if evidence_type in {"lesion_segmentation", "grounded_classifier"}:
                    lesion_map_path = heatmap_path

        payload = InferencePayload(
            predicted_index=result.predicted_index,
            predicted_label=result.predicted_label,
            abnormal_probability=result.abnormal_probability,
            decision_threshold=self.decision_threshold,
            checkpoint_path=str(self.checkpoint_path),
            prediction_path=str(prediction_path) if prediction_path else None,
            heatmap_path=str(heatmap_path) if heatmap_path else None,
            xai_error_code=xai_error_code,
            xai_no_region=xai_no_region,
            evidence_type=evidence_type,
            lesion_map_path=str(lesion_map_path) if lesion_map_path else None,
            lesion_summary=lesion_summary,
            evidence_warning=evidence_warning,
            eval_metrics=self.eval_metrics,
        ).to_dict()

        if save_outputs:
            if heatmap_overlay is not None and heatmap_path is not None:
                heatmap_overlay.save(heatmap_path)
            prediction_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

        return SingleImagePrediction(
            result=result,
            payload=payload,
            original_image=original_image,
            heatmap_overlay=heatmap_overlay,
            saved=SavedInferenceArtifacts(
                prediction_path=prediction_path,
                heatmap_path=heatmap_path,
                lesion_map_path=lesion_map_path,
            ),
        )
