from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import yaml

RUN_PRIMARY_GROUPS: dict[str, str] = {
    "effnet_1shot": "00_baselines_and_early",
    "v3": "00_baselines_and_early",
    "v4": "00_baselines_and_early",
    "v5": "00_baselines_and_early",
    "ssl_effnet_clean": "01_ssl_lineage",
    "effnet_ssl_clean": "01_ssl_lineage",
    "effnet_ssl_clean_focal": "01_ssl_lineage",
    "v4.1": "01_ssl_lineage",
    "v4b": "01_ssl_lineage",
    "v4b_alpha_only": "01_ssl_lineage",
    "v6": "01_ssl_lineage",
    "v6_alpha_only": "01_ssl_lineage",
    "v6_gamma_only": "01_ssl_lineage",
    "v7_messidor_train": "02_domain_generalization",
    "v8_mixstyle": "02_domain_generalization",
    "v9_fda": "02_domain_generalization",
    "v10_swad": "02_domain_generalization",
    "v11_fda_swad": "02_domain_generalization",
    "v12_fda_imagenet": "02_domain_generalization",
    "v13_fda_swad": "02_domain_generalization",
    "v14_ibn": "02_domain_generalization",
    "v15_fda_a10": "02_domain_generalization",
    "v16_focal_g1": "02_domain_generalization",
    "v17_focal_g2": "02_domain_generalization",
    "v18_focal_g3": "02_domain_generalization",
    "v19_swad_focal_g2": "02_domain_generalization",
    "v20_coral": "02_domain_generalization",
    "v7_512_messidor_train": "03_resolution_layercam",
    "v17_512_focal_g2": "03_resolution_layercam",
    "v21_512_focal_g2": "03_resolution_layercam",
    "v21_512_layercam": "03_resolution_layercam",
    "v24_multitask": "04_lesion_supervision",
    "v25_multitask_l1": "04_lesion_supervision",
    "v26_multitask_l3": "04_lesion_supervision",
    "v27_mil_attention": "04_lesion_supervision",
    "v28_no_attention": "05_xai_attention_ablation",
    "v29_with_attention": "05_xai_attention_ablation",
    "v30_gated_pooling": "06_xai_classifier_routing",
    "v31_no_se_gated": "07_lesion_evidence",
    "v32_lesion_seg_evidence": "07_lesion_evidence",
    "v33_per_lesion_routing": "07_lesion_evidence",
}


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


def get_run_group(version: str) -> str:
    return RUN_PRIMARY_GROUPS.get(version, "99_misc")


def get_run_artifact_dir(project_root: str | Path, version: str) -> Path:
    return Path(project_root) / "artifacts" / "runs" / get_run_group(version) / version


def get_run_checkpoint_dir(project_root: str | Path, version: str) -> Path:
    return get_run_artifact_dir(project_root, version) / "checkpoints"


def get_run_evaluation_dir(project_root: str | Path, version: str) -> Path:
    return get_run_artifact_dir(project_root, version) / "evaluations"


def get_run_log_dir(project_root: str | Path, version: str) -> Path:
    return get_run_artifact_dir(project_root, version) / "logs"


def _legacy_checkpoint_parts(path: Path) -> tuple[str, tuple[str, ...]] | None:
    parts = path.parts
    for i in range(len(parts) - 2):
        if parts[i] == "artifacts" and parts[i + 1] == "checkpoints":
            version = parts[i + 2]
            if version.endswith(".pt"):
                return None
            return version, tuple(parts[i + 3 :])
    return None


def resolve_checkpoint_path(
    project_root: str | Path,
    value: str | Path,
    *,
    version: str | None = None,
) -> Path:
    resolved = resolve_project_path(project_root, value)
    if resolved.exists():
        return resolved

    legacy = _legacy_checkpoint_parts(resolved)
    if legacy is not None:
        legacy_version, remaining = legacy
        candidate = get_run_checkpoint_dir(project_root, legacy_version).joinpath(*remaining)
        if candidate.exists():
            return candidate

    if version:
        candidate = get_run_checkpoint_dir(project_root, version) / resolved.name
        if candidate.exists():
            return candidate

    return resolved


def classification_metrics_filename(
    split_name: str,
    version: str,
    checkpoint_stem: str = "best",
) -> str:
    return f"{split_name}_{version}_{checkpoint_stem}_metrics.json"


def find_classification_metrics_path(
    project_root: str | Path,
    version: str,
    *,
    split_name: str = "external_test",
    checkpoint_stem: str = "best",
) -> Path:
    filename = classification_metrics_filename(split_name, version, checkpoint_stem)
    candidates = [
        get_run_evaluation_dir(project_root, version) / filename,
        Path(project_root) / "artifacts" / "evaluations" / filename,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def build_effective_checkpoint_config(
    runtime_config: Mapping[str, Any],
    checkpoint: Mapping[str, Any],
) -> dict[str, Any]:
    """Build an effective config by merging the checkpoint's saved config with the runtime config.

    Ensures model architecture, num_outputs, label_names, and project version from
    the checkpoint take precedence, and pretrained is forced to False (weights come
    from the checkpoint).
    """
    checkpoint_config = checkpoint.get("config")
    checkpoint_project_config = (
        checkpoint_config.get("project", {}) if isinstance(checkpoint_config, dict) else {}
    )
    checkpoint_model_config = checkpoint_config.get("model", {}) if isinstance(checkpoint_config, dict) else {}
    effective_config = dict(runtime_config)
    if isinstance(checkpoint_config, dict):
        effective_config = merge_dicts(checkpoint_config, runtime_config)

    metadata_overrides: dict[str, Any] = {}
    checkpoint_version = checkpoint_project_config.get("version")
    if checkpoint_version:
        metadata_overrides["project"] = {"version": checkpoint_version}

    effective_config = merge_dicts(
        effective_config,
        {
            **metadata_overrides,
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
                "attention_mode": checkpoint_model_config.get(
                    "attention_mode",
                    effective_config["model"].get("attention_mode"),
                ),
                "use_ibn": checkpoint_model_config.get(
                    "use_ibn",
                    effective_config["model"].get("use_ibn", False),
                ),
                "grad_checkpointing": checkpoint_model_config.get(
                    "grad_checkpointing",
                    effective_config["model"].get("grad_checkpointing", False),
                ),
                "use_aux_seg": checkpoint_model_config.get(
                    "use_aux_seg",
                    effective_config["model"].get("use_aux_seg", False),
                ),
                "aux_seg_block": checkpoint_model_config.get(
                    "aux_seg_block",
                    effective_config["model"].get("aux_seg_block", 2),
                ),
                "aux_seg_output_size": checkpoint_model_config.get(
                    "aux_seg_output_size",
                    effective_config["model"].get("aux_seg_output_size", 512),
                ),
                "aux_seg_channels": checkpoint_model_config.get(
                    "aux_seg_channels",
                    effective_config["model"].get("aux_seg_channels", 1),
                ),
                "use_gated_pooling": checkpoint_model_config.get(
                    "use_gated_pooling",
                    effective_config["model"].get("use_gated_pooling", False),
                ),
                "use_mil_attention": checkpoint_model_config.get(
                    "use_mil_attention",
                    effective_config["model"].get("use_mil_attention", False),
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
    version = str(config.get("project", {}).get("version", "")).strip()

    paths = [
        train_cfg.get("checkpoint_dir"),
        infer_cfg.get("prediction_dir"),
        infer_cfg.get("heatmap_dir"),
    ]
    for value in paths:
        if value:
            resolve_project_path(root, value).mkdir(parents=True, exist_ok=True)
    if version:
        get_run_checkpoint_dir(root, version).mkdir(parents=True, exist_ok=True)
        get_run_evaluation_dir(root, version).mkdir(parents=True, exist_ok=True)
        get_run_log_dir(root, version).mkdir(parents=True, exist_ok=True)
