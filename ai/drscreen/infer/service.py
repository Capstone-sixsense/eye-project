"""추론 세션 오케스트레이션: 이미지 한 장 -> 백엔드 페이로드 전체 흐름.

이 모듈의 InferenceSession이 배포 추론의 최상위 진입점이다. 책임은 크게 둘:

1. 구성(from_config_path): config + 체크포인트를 읽어 모델을 만들고, 입력 transform,
   전처리기(FundusPreprocess), 판정 임계값, eval 메트릭을 준비한다.
2. 추론(predict_pil_image): 전처리 -> transform -> 모델 forward(융합이면 메타 분류기,
   선택적으로 hflip TTA) -> evidence 오버레이 생성 -> 페이로드/아티팩트 저장.

evidence_type에 따라 근거 시각화가 갈린다:
- lesion_segmentation: v8b 병변맵 오버레이(현재 배포). 실패 시 xai_error_code="XAI_002".
- grounded_classifier : BagNet 등 patch-logit 오버레이. 실패 시 "XAI_003".
- cam_research        : Grad-CAM/Layer-CAM 오버레이. 실패 시 "XAI_001".

파일 앞부분의 `_`-접두 헬퍼들은 메트릭 JSON 파싱, 임계값 검증, 오버레이 렌더링,
망막 영역 마스크 추출 등 추론을 구성하는 작은 부품들이다.
"""

from __future__ import annotations

import json
import re
import warnings
from dataclasses import dataclass
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image

from drscreen.data.transforms import (
    FundusPreprocess,
    build_eval_transform,
    is_preprocessed_image_path,
    preprocess_kwargs_from_config,
)
from drscreen.infer.payload import InferencePayload
from drscreen.infer.pipeline import InferenceResult, run_single_image_inference
from drscreen.models.build import build_model
from drscreen.models.profiles import get_model_profile
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
from drscreen.utils.checkpoint import load_state_from_checkpoint
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
    thresholds = checkpoint.get("thresholds")
    if isinstance(thresholds, dict):
        for key in ("fusion_score", "classification", "v31_legacy"):
            threshold = _as_valid_threshold(thresholds.get(key))
            if threshold is not None:
                return threshold
    return None


def _mean_metric(section: Any) -> float | None:
    if not isinstance(section, dict):
        return None
    return _as_valid_threshold(section.get("mean"))


def _as_non_negative_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    if parsed < 0:
        return None
    return parsed


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
    evidence_type = str(infer_cfg.get("evidence_type", "cam_research")).strip().lower()
    if evidence_type in {"lesion_segmentation", "lesion_evidence", "segmentation"}:
        metrics = _load_lesion_segmentation_eval_metrics(
            project_root,
            version,
            split=split,
        )
        if metrics:
            return metrics

    compact_dir = Path(project_root) / "artifacts" / "evaluations"
    compact_candidates = [
        compact_dir / f"xai_{version}_{method}_{block_label}_{split}_best_metrics.json",
        compact_dir / f"xai_{version}_{block_label}_{split}_best_metrics.json",
    ]
    for path in compact_candidates:
        if not path.exists():
            continue
        metrics = _load_compact_xai_eval_metrics(path, split=split, block_label=block_label)
        if metrics:
            return metrics

    eval_dir = get_run_evaluation_dir(project_root, version)
    raw_candidates = [
        eval_dir / f"xai_iou_{version}_{method}_{block_label}_{split}.json",
        eval_dir / f"xai_iou_{version}_{block_label}_{split}.json",
    ]
    for path in raw_candidates:
        if not path.exists():
            continue
        metrics = _load_raw_xai_eval_metrics(path, split=split, block_label=block_label)
        if metrics:
            return metrics

    return None


def _load_lesion_segmentation_eval_metrics(
    project_root: Path,
    version: str,
    *,
    split: str,
) -> dict[str, Any] | None:
    compact_dir = Path(project_root) / "artifacts" / "evaluations"
    compact_candidates = [
        compact_dir / f"xai_{version}_lesion_segmentation_{split}_best_metrics.json",
        compact_dir / f"xai_{version}_segmentation_{split}_best_metrics.json",
    ]
    for path in compact_candidates:
        if not path.exists():
            continue
        metrics = _load_compact_xai_eval_metrics(
            path,
            split=split,
            block_label="lesion_segmentation",
        )
        if metrics:
            metrics.setdefault("xai_evidence_type", "lesion_segmentation")
            return metrics
    return None


def _load_raw_xai_eval_metrics(
    path: Path,
    *,
    split: str,
    block_label: str,
) -> dict[str, Any] | None:
    try:
        with open(path, encoding="utf-8") as handle:
            data = json.load(handle)
    except Exception:
        return None
    if not isinstance(data, dict):
        return None

    aggregate = data.get("aggregate", {})
    thresholds = aggregate.get("thresholds", {}) if isinstance(aggregate, dict) else {}

    metrics: dict[str, Any] = {
        "xai_eval_split": data.get("split", split),
        "xai_eval_target_block": data.get("target_block", block_label),
        "xai_eval_n": _as_non_negative_int(data.get("n_images")),
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


def _load_compact_xai_eval_metrics(
    path: Path,
    *,
    split: str,
    block_label: str,
) -> dict[str, Any] | None:
    try:
        with open(path, encoding="utf-8") as handle:
            data = json.load(handle)
    except Exception:
        return None
    if not isinstance(data, dict):
        return None

    metric_source = data.get("metrics", {})
    if not isinstance(metric_source, dict):
        metric_source = {}

    def pick(*keys: str) -> Any:
        for key in keys:
            if key in metric_source and metric_source[key] is not None:
                return metric_source[key]
            if key in data and data[key] is not None:
                return data[key]
        return None

    metrics: dict[str, Any] = {
        "xai_eval_split": pick("xai_eval_split", "split") or split,
        "xai_eval_target_block": pick("xai_eval_target_block", "target_block") or block_label,
        "xai_eval_n": _as_non_negative_int(pick("xai_eval_n", "n_images")),
        "xai_pointing_game": _as_valid_threshold(pick("xai_pointing_game")),
        "xai_auprc": _as_valid_threshold(pick("xai_auprc")),
        "xai_auc_iou": _as_valid_threshold(pick("xai_auc_iou")),
        "xai_iou_top10": _as_valid_threshold(pick("xai_iou_top10")),
        "xai_iou_top20": _as_valid_threshold(pick("xai_iou_top20")),
        "xai_iou_top30": _as_valid_threshold(pick("xai_iou_top30")),
    }
    for key, value in metric_source.items():
        if not key.startswith("xai_") or key in metrics or value is None:
            continue
        if isinstance(value, bool) or isinstance(value, int):
            metrics[key] = value
        elif isinstance(value, float):
            metrics[key] = float(value)
        elif isinstance(value, str):
            metrics[key] = value
    return {key: value for key, value in metrics.items() if value is not None}


def _build_retina_mask(image: Image.Image) -> np.ndarray:
    # 안저 원형 영역만 1로 표시하는 마스크. 검은 배경에 evidence가 칠해지는 것을 막아
    # 활성 비율/면적 계산이 망막 내부만 대상으로 하도록 한다.
    # 방법: 밝기 임계화 -> 모폴로지 정리 -> 가장 큰 연결요소(=망막 원판) 선택.
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


def _config_float(
    config: dict[str, Any] | None,
    key: str,
    default: float,
    *,
    minimum: float,
    maximum: float,
) -> float:
    if config is None:
        return default
    try:
        value = float(config.get(key, default))
    except (TypeError, ValueError):
        return default
    if not np.isfinite(value):
        return default
    return float(np.clip(value, minimum, maximum))


def _render_gradcam_overlay(
    image: Image.Image,
    heatmap: torch.Tensor,
    infer_cfg: dict[str, Any] | None = None,
) -> tuple[Image.Image, bool]:
    normalized = heatmap.detach().cpu().clamp(0.0, 1.0).numpy().astype(np.float32)
    retina_mask = _build_retina_mask(image)
    resized = cv2.resize(
        normalized,
        dsize=image.size,
        interpolation=cv2.INTER_LINEAR,
    )
    resized *= retina_mask

    # Suppress weak activations while keeping enough mid-strength evidence visible.
    activation_threshold = _config_float(
        infer_cfg,
        "heatmap_activation_threshold",
        0.25,
        minimum=0.0,
        maximum=0.95,
    )
    emphasized = np.clip(
        (resized - activation_threshold) / (1.0 - activation_threshold),
        0.0,
        1.0,
    )
    heatmap_gamma = _config_float(
        infer_cfg,
        "heatmap_gamma",
        0.65,
        minimum=0.1,
        maximum=3.0,
    )
    emphasized = np.power(emphasized, heatmap_gamma, dtype=np.float32)

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
    heatmap_alpha = _config_float(
        infer_cfg,
        "heatmap_alpha",
        0.86,
        minimum=0.0,
        maximum=1.0,
    )
    alpha_mask = emphasized[..., None] * heatmap_alpha
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
            segmenter_encoder=str(effective_config["model"].get("segmenter_encoder", "resnet50")),
            segmenter_out_channels=int(effective_config["model"].get("segmenter_out_channels", 4)),
            segmenter_decoder_channels=(
                [int(channel) for channel in effective_config["model"]["segmenter_decoder_channels"]]
                if effective_config["model"].get("segmenter_decoder_channels") is not None
                else None
            ),
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
        preprocess_options = preprocess_kwargs_from_config(data_cfg, infer_cfg)
        preprocessor = (
            FundusPreprocess(
                output_size=preprocess_size,
                align=use_align,
                **preprocess_options,
            )
            if use_preprocessing
            else None
        )
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
            prefer_compact=True,
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

        # 판정 임계값 우선순위: 체크포인트/메트릭의 optimal_threshold > config의
        # infer.threshold > 최후 기본값 0.5. 아티팩트 임계값을 최우선으로 신뢰한다.
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
        # Footgun 방어: 이미 오프라인 전처리(geometry + Ben Graham)된 이미지에
        # Ben Graham을 또 적용하면 정확도가 급락한다(meta AUROC ~0.93 -> ~0.80).
        # 입력 경로가 전처리본으로 보이는데 config가 다시 Ben Graham을 켰다면 경고한다.
        if (
            self.preprocessor is not None
            and getattr(self.preprocessor, "_apply_ben_graham", False)
            and is_preprocessed_image_path(resolved_image_path)
        ):
            warnings.warn(
                f"Input {resolved_image_path} looks already offline-preprocessed "
                "(geometry + Ben Graham), but the active config applies Ben Graham again "
                "(infer.use_preprocessing: true). This double-applies Ben Graham and degrades "
                "accuracy (meta AUROC ~0.93 -> ~0.80). For preprocessed inputs use a config "
                "with infer.use_preprocessing: false (or data/infer apply_ben_graham: false).",
                stacklevel=2,
            )
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
        # 1) 선택적 라이브 전처리(QuickQual crop/Ben Graham 등) -> 2) eval transform(resize/
        # crop/normalize) -> 모델 입력 텐서. 배포 config는 preprocess_mode=none(백엔드가
        # 이미 QuickQual을 수행)이라 보통 preprocessor가 None이다.
        original_image = image.convert("RGB")
        if self.preprocessor is not None:
            original_image = self.preprocessor(original_image)
        image_tensor = self.eval_transform(original_image).to(self.device)

        infer_cfg = self.config.get("infer", {})
        fusion_output: dict[str, Any] | None = None
        cached_lesion_prob: torch.Tensor | None = None
        fusion_summary: dict[str, Any] | None = None
        # 융합 경로: 메타 분류기를 쓰는 모델은 predict_fusion_score로 최종 확률을 얻는다.
        # (일반 분류기는 아래 else의 run_single_image_inference를 사용.)
        if bool(infer_cfg.get("use_meta_classifier", False)):
            if not hasattr(self.model, "predict_fusion_score"):
                raise RuntimeError("use_meta_classifier=True requires predict_fusion_score().")
            amp_enabled = bool(infer_cfg.get("amp", False))
            tta_mode = str(infer_cfg.get("tta_mode", "none")).strip().lower()
            if tta_mode in {"", "none", "off", "false"}:
                tta_mode = "none"
            if tta_mode not in {"none", "hflip", "hflip_feature_recalc"}:
                raise ValueError(f"Unsupported infer.tta_mode: {tta_mode}")

            fusion_output = self.model.predict_fusion_score(
                image_tensor.unsqueeze(0),
                amp_enabled=amp_enabled,
            )
            meta_prob = fusion_output.get("meta_probability")
            if meta_prob is None:
                raise RuntimeError("Fusion model did not produce meta_probability.")
            if tta_mode in {"hflip", "hflip_feature_recalc"}:
                flipped_tensor = torch.flip(image_tensor, dims=[2])
                flipped_output = self.model.predict_fusion_score(
                    flipped_tensor.unsqueeze(0),
                    amp_enabled=amp_enabled,
                )
                flipped_meta_prob = flipped_output.get("meta_probability")
                if flipped_meta_prob is None:
                    raise RuntimeError("Fusion hflip view did not produce meta_probability.")
                if tta_mode == "hflip_feature_recalc":
                    cached_seg = fusion_output.get("seg_prob")
                    flipped_seg = flipped_output.get("seg_prob")
                    if not isinstance(cached_seg, torch.Tensor) or not isinstance(flipped_seg, torch.Tensor):
                        raise RuntimeError("Fusion feature-recalc TTA requires seg_prob tensors.")
                    averaged_seg = (cached_seg[0] + torch.flip(flipped_seg[0], dims=[2])) * 0.5
                    recalc_output = self.model.predict_fusion_from_components(
                        v31_probability=float(fusion_output["v31_probability"]),
                        v31_logit=float(fusion_output["v31_logit"]),
                        seg_prob=averaged_seg,
                    )
                    recalc_meta_prob = recalc_output.get("meta_probability")
                    if recalc_meta_prob is None:
                        raise RuntimeError("Fusion feature-recalc TTA did not produce meta_probability.")
                    abnormal_probability = float(recalc_meta_prob)
                    fusion_output = {**fusion_output, **recalc_output}
                else:
                    abnormal_probability = (float(meta_prob) + float(flipped_meta_prob)) / 2.0
            else:
                abnormal_probability = float(meta_prob)
            predicted_index = int(abnormal_probability >= self.decision_threshold)
            predicted_label = self.label_names[predicted_index]
            result = InferenceResult(
                predicted_index=predicted_index,
                predicted_label=predicted_label,
                abnormal_probability=abnormal_probability,
            )
            cached_seg = fusion_output.get("seg_prob")
            if cached_lesion_prob is None and isinstance(cached_seg, torch.Tensor):
                cached_lesion_prob = cached_seg[0]
            feature_extraction = getattr(self.model, "feature_extraction", {}) or {}
            fusion_summary = {
                "v31_prob_pre_meta": float(fusion_output["v31_probability"]),
                "v31_logit_pre_meta": float(fusion_output["v31_logit"]),
                "meta_prob": abnormal_probability,
                "fusion_threshold": float(self.decision_threshold),
                "feature_schema_len": len(getattr(self.model, "feature_schema", []) or []),
                "fusion_features_first10": [float(v) for v in fusion_output.get("features", [])[:10]],
                "feature_area_thresholds": feature_extraction.get("area_thresholds"),
                "feature_topk_fracs": feature_extraction.get("topk_fracs"),
            }
        else:
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
        # evidence(근거 시각화) 생성은 config의 evidence_type으로 분기한다. 어떤 분기든
        # 실패는 예측 자체를 막지 않고 xai_error_code로만 표시한다(추론은 best-effort).
        evidence_type = str(infer_cfg.get("evidence_type", "cam_research")).strip().lower()
        if evidence_type in {"lesion_segmentation", "lesion_evidence", "segmentation"}:
            evidence_type = "lesion_segmentation"
            try:
                # 융합 경로에서 이미 계산한 병변맵이 있으면 재사용(중복 forward 회피).
                if cached_lesion_prob is not None:
                    lesion_prob = cached_lesion_prob
                else:
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
                if fusion_summary is not None:
                    lesion_summary = {**lesion_summary, **fusion_summary}
            except Exception:
                xai_error_code = "XAI_002"
                evidence_warning = "LESION_EVIDENCE_UNAVAILABLE"
        elif evidence_type in {"grounded_classifier", "bagnet", "patch_logits"}:
            evidence_type = "grounded_classifier"
            try:
                if not hasattr(self.model, "get_evidence_map"):
                    raise ValueError("Model does not expose get_evidence_map().")
                evidence = self.model.get_evidence_map(image_tensor.unsqueeze(0))[0, 0]
                heatmap_overlay, xai_no_region = _render_gradcam_overlay(
                    original_image,
                    evidence,
                    infer_cfg,
                )
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
                heatmap_overlay, xai_no_region = _render_gradcam_overlay(
                    original_image,
                    gradcam.heatmap[0],
                    infer_cfg,
                )
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
