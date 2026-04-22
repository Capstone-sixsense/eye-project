from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import yaml


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config root must be a mapping: {path}")
    return data


def merge_dicts(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = merge_dicts(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_app_config(config_path: str | Path, base_path: str | Path | None = None) -> dict[str, Any]:
    config = load_yaml(config_path)
    if base_path is None:
        return config
    return merge_dicts(load_yaml(base_path), config)


def resolve_project_path(project_root: str | Path, value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return Path(project_root) / path


def build_effective_checkpoint_config(
    runtime_config: Mapping[str, Any],
    checkpoint: Mapping[str, Any],
) -> dict[str, Any]:
    """Build an effective config by merging the checkpoint's saved config with the runtime config.

    Ensures model architecture, num_outputs, and label_names from the checkpoint
    take precedence, and pretrained is forced to False (weights come from the checkpoint).
    """
    checkpoint_config = checkpoint.get("config")
    checkpoint_model_config = checkpoint_config.get("model", {}) if isinstance(checkpoint_config, dict) else {}
    effective_config = dict(runtime_config)
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
                "use_attention": checkpoint_model_config.get(
                    "use_attention",
                    effective_config["model"].get("use_attention", False),
                ),
                "use_mixstyle": checkpoint_model_config.get(
                    "use_mixstyle",
                    effective_config["model"].get("use_mixstyle", False),
                ),
                "use_ibn": checkpoint_model_config.get(
                    "use_ibn",
                    effective_config["model"].get("use_ibn", False),
                ),
                "grad_checkpointing": checkpoint_model_config.get(
                    "grad_checkpointing",
                    effective_config["model"].get("grad_checkpointing", False),
                ),
                "classifier_dropout": checkpoint_model_config.get(
                    "classifier_dropout",
                    effective_config["model"].get("classifier_dropout", 0.0),
                ),
                "zero_init_classifier": checkpoint_model_config.get(
                    "zero_init_classifier",
                    effective_config["model"].get("zero_init_classifier", False),
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


def ensure_runtime_directories(config: Mapping[str, Any], project_root: str | Path) -> None:
    root = Path(project_root)
    train_cfg = config.get("train", {})
    infer_cfg = config.get("infer", {})

    paths = [
        train_cfg.get("checkpoint_dir"),
        infer_cfg.get("prediction_dir"),
        infer_cfg.get("heatmap_dir"),
    ]
    for value in paths:
        if value:
            resolve_project_path(root, value).mkdir(parents=True, exist_ok=True)
