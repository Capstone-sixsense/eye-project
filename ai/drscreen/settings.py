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
