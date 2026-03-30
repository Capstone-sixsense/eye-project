from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
from PIL import Image

from drscreen.quality.checks import compute_blur_score, compute_mean_brightness


DEFAULT_PERCENTILES = (1, 5, 10, 25, 50, 75, 90, 95, 99)


@dataclass(frozen=True, slots=True)
class CalibrationConfig:
    blur_percentile: float = 5.0
    brightness_percentile: float = 5.0
    output_dir: str = "artifacts/quality"


def load_rgb_array(image_path: str | Path) -> np.ndarray:
    return np.asarray(Image.open(image_path).convert("RGB"))


def measure_quality_for_manifest(
    manifest_path: str | Path,
    image_root: str | Path,
    *,
    limit: int | None = None,
) -> pd.DataFrame:
    manifest = pd.read_csv(manifest_path)
    required_columns = {"image_path", "split"}
    missing = required_columns.difference(manifest.columns)
    if missing:
        raise ValueError(f"Manifest is missing columns: {sorted(missing)}")

    image_root = Path(image_root)
    frame = manifest.copy()
    if limit is not None:
        frame = frame.head(limit).copy()

    rows: list[dict[str, Any]] = []
    for row in frame.itertuples(index=False):
        relative_path = Path(str(row.image_path))
        image_path = image_root / relative_path
        image = load_rgb_array(image_path)
        rows.append(
            {
                "image_path": relative_path.as_posix(),
                "split": str(row.split),
                "domain": str(getattr(row, "domain", "unknown")),
                "label": int(getattr(row, "label", -1)),
                "blur_score": compute_blur_score(image),
                "mean_brightness": compute_mean_brightness(image),
            }
        )

    return pd.DataFrame(rows)


def _series_summary(series: pd.Series, recommended_percentile: float) -> dict[str, float]:
    return {
        "min": float(series.min()),
        "max": float(series.max()),
        "mean": float(series.mean()),
        "std": float(series.std(ddof=0)),
        "recommended_threshold": float(series.quantile(recommended_percentile / 100.0)),
        **{
            f"p{percentile:02d}": float(series.quantile(percentile / 100.0))
            for percentile in DEFAULT_PERCENTILES
        },
    }


def _summarize_group(
    frame: pd.DataFrame,
    *,
    blur_percentile: float,
    brightness_percentile: float,
    current_blur_threshold: float,
    current_brightness_threshold: float,
) -> dict[str, Any]:
    if frame.empty:
        raise ValueError("Cannot summarize empty calibration frame.")

    blur_summary = _series_summary(frame["blur_score"], blur_percentile)
    brightness_summary = _series_summary(frame["mean_brightness"], brightness_percentile)
    return {
        "rows": int(len(frame)),
        "current_threshold_failures": {
            "blur_score_below_threshold": int((frame["blur_score"] < current_blur_threshold).sum()),
            "brightness_below_threshold": int(
                (frame["mean_brightness"] < current_brightness_threshold).sum()
            ),
            "any_quality_flag": int(
                (
                    (frame["blur_score"] < current_blur_threshold)
                    | (frame["mean_brightness"] < current_brightness_threshold)
                ).sum()
            ),
        },
        "blur_score": blur_summary,
        "mean_brightness": brightness_summary,
    }


def build_calibration_report(
    measurements: pd.DataFrame,
    *,
    blur_percentile: float,
    brightness_percentile: float,
    current_blur_threshold: float,
    current_brightness_threshold: float,
) -> dict[str, Any]:
    report: dict[str, Any] = {
        "method": {
            "type": "percentile_heuristic",
            "note": (
                "Ground-truth image quality labels were not provided, so recommended thresholds "
                "are estimated from the lower-tail percentile of the observed dataset distribution."
            ),
            "blur_percentile": blur_percentile,
            "brightness_percentile": brightness_percentile,
        },
        "current_thresholds": {
            "blur_score_min": current_blur_threshold,
            "brightness_mean_min": current_brightness_threshold,
        },
        "recommended_thresholds": {},
        "global": _summarize_group(
            measurements,
            blur_percentile=blur_percentile,
            brightness_percentile=brightness_percentile,
            current_blur_threshold=current_blur_threshold,
            current_brightness_threshold=current_brightness_threshold,
        ),
        "by_domain": {},
        "by_domain_split": {},
    }
    report["recommended_thresholds"] = {
        "blur_score_min": report["global"]["blur_score"]["recommended_threshold"],
        "brightness_mean_min": report["global"]["mean_brightness"]["recommended_threshold"],
    }

    for domain, group in measurements.groupby("domain", sort=True):
        report["by_domain"][str(domain)] = _summarize_group(
            group,
            blur_percentile=blur_percentile,
            brightness_percentile=brightness_percentile,
            current_blur_threshold=current_blur_threshold,
            current_brightness_threshold=current_brightness_threshold,
        )

    for (domain, split), group in measurements.groupby(["domain", "split"], sort=True):
        report["by_domain_split"][f"{domain}:{split}"] = _summarize_group(
            group,
            blur_percentile=blur_percentile,
            brightness_percentile=brightness_percentile,
            current_blur_threshold=current_blur_threshold,
            current_brightness_threshold=current_brightness_threshold,
        )

    return report


def resolve_calibration_config(config: Mapping[str, Any]) -> CalibrationConfig:
    quality_cfg = config.get("quality", {})
    calibration_cfg = quality_cfg.get("calibration", {})
    return CalibrationConfig(
        blur_percentile=float(calibration_cfg.get("blur_percentile", 5.0)),
        brightness_percentile=float(calibration_cfg.get("brightness_percentile", 5.0)),
        output_dir=str(calibration_cfg.get("output_dir", "artifacts/quality")),
    )


def write_calibration_outputs(
    *,
    measurements: pd.DataFrame,
    report: Mapping[str, Any],
    output_dir: str | Path,
) -> tuple[Path, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    measurements_path = output_dir / "quality_measurements.csv"
    report_path = output_dir / "quality_calibration_report.json"

    measurements.to_csv(measurements_path, index=False)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return measurements_path, report_path


def format_report_excerpt(report: Mapping[str, Any]) -> str:
    recommended = report["recommended_thresholds"]
    global_summary = report["global"]
    return "\n".join(
        [
            "Quality calibration summary",
            f"recommended.blur_score_min: {recommended['blur_score_min']:.3f}",
            f"recommended.brightness_mean_min: {recommended['brightness_mean_min']:.3f}",
            (
                "current_failures.any_quality_flag: "
                f"{global_summary['current_threshold_failures']['any_quality_flag']}"
            ),
            f"rows: {global_summary['rows']}",
        ]
    )


def calibration_to_dict(config: CalibrationConfig) -> dict[str, Any]:
    return asdict(config)
