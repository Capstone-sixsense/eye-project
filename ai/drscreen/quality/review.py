from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd


def build_flagged_quality_frame(
    measurements: pd.DataFrame,
    *,
    blur_threshold: float,
    brightness_threshold: float,
    image_root: str | Path,
) -> pd.DataFrame:
    required_columns = {"image_path", "split", "domain", "blur_score", "mean_brightness"}
    missing = required_columns.difference(measurements.columns)
    if missing:
        raise ValueError(f"Measurements are missing columns: {sorted(missing)}")

    frame = measurements.copy()
    frame["blur_flag"] = frame["blur_score"] < blur_threshold
    frame["brightness_flag"] = frame["mean_brightness"] < brightness_threshold
    frame["flag_reason"] = np.select(
        [
            frame["blur_flag"] & frame["brightness_flag"],
            frame["blur_flag"],
            frame["brightness_flag"],
        ],
        ["both", "blur", "brightness"],
        default="ok",
    )
    frame = frame[frame["flag_reason"] != "ok"].copy()

    blur_scale = max(float(blur_threshold), 1e-6)
    brightness_scale = max(float(brightness_threshold), 1e-6)
    frame["blur_margin"] = np.maximum(0.0, blur_threshold - frame["blur_score"])
    frame["brightness_margin"] = np.maximum(0.0, brightness_threshold - frame["mean_brightness"])
    frame["severity_score"] = (
        frame["blur_margin"] / blur_scale + frame["brightness_margin"] / brightness_scale
    )
    image_root = Path(image_root)
    frame["absolute_image_path"] = frame["image_path"].map(
        lambda relative: str((image_root / str(relative)).resolve())
    )
    frame["review_status"] = ""
    frame["review_note"] = ""
    frame = frame.sort_values(
        ["domain", "flag_reason", "severity_score", "image_path"],
        ascending=[True, True, False, True],
        kind="stable",
    ).reset_index(drop=True)
    return frame


def _spaced_indices(length: int, count: int) -> list[int]:
    if count >= length:
        return list(range(length))
    indices = np.linspace(0, length - 1, count)
    unique_indices = sorted({int(round(index)) for index in indices})
    candidate = 0
    while len(unique_indices) < count:
        if candidate not in unique_indices:
            unique_indices.append(candidate)
        candidate += 1
    return sorted(unique_indices[:count])


def sample_review_candidates(flagged: pd.DataFrame, *, per_group: int) -> pd.DataFrame:
    if per_group <= 0:
        raise ValueError("per_group must be positive.")
    if flagged.empty:
        return flagged.copy()

    samples: list[pd.DataFrame] = []
    for (_, _), group in flagged.groupby(["domain", "flag_reason"], sort=True):
        ordered = group.sort_values(
            ["severity_score", "image_path"],
            ascending=[False, True],
            kind="stable",
        ).reset_index(drop=True)
        take_indices = _spaced_indices(len(ordered), per_group)
        sampled = ordered.iloc[take_indices].copy()
        sampled["sample_group_size"] = len(ordered)
        samples.append(sampled)

    return pd.concat(samples, ignore_index=True) if samples else flagged.iloc[0:0].copy()


def build_review_summary(
    *,
    flagged: pd.DataFrame,
    sampled: pd.DataFrame,
    blur_threshold: float,
    brightness_threshold: float,
    per_group: int,
) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "thresholds": {
            "blur_score_min": blur_threshold,
            "brightness_mean_min": brightness_threshold,
        },
        "sampling": {
            "strategy": "severity_spread_per_domain_reason",
            "per_group": per_group,
        },
        "flagged_rows": int(len(flagged)),
        "sampled_rows": int(len(sampled)),
        "flagged_by_group": {},
        "sampled_by_group": {},
    }

    flagged_grouped = flagged.groupby(["domain", "flag_reason"]).size()
    for (domain, reason), count in flagged_grouped.items():
        summary["flagged_by_group"][f"{domain}:{reason}"] = int(count)

    sampled_grouped = sampled.groupby(["domain", "flag_reason"]).size()
    for (domain, reason), count in sampled_grouped.items():
        summary["sampled_by_group"][f"{domain}:{reason}"] = int(count)

    return summary


def write_review_outputs(
    *,
    flagged: pd.DataFrame,
    sampled: pd.DataFrame,
    summary: Mapping[str, Any],
    output_dir: str | Path,
) -> tuple[Path, Path, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    flagged_path = output_dir / "quality_flagged_all.csv"
    sample_path = output_dir / "quality_review_sample.csv"
    summary_path = output_dir / "quality_review_summary.json"

    flagged.to_csv(flagged_path, index=False)
    sampled.to_csv(sample_path, index=False)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return flagged_path, sample_path, summary_path
