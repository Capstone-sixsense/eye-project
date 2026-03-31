from __future__ import annotations

import argparse
from pathlib import Path

from drscreen.quality.calibration import (
    build_calibration_report,
    format_report_excerpt,
    measure_quality_for_manifest,
    resolve_calibration_config,
    write_calibration_outputs,
)
from drscreen.settings import load_app_config, resolve_project_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calibrate blur/brightness thresholds from manifest.")
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional row limit for quick inspection runs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).resolve()
    project_root = config_path.parents[1]
    base_path = None
    candidate_base = config_path.parent / "base.yaml"
    if config_path.name != "base.yaml" and candidate_base.exists():
        base_path = candidate_base

    config = load_app_config(config_path, base_path=base_path)
    calibration_cfg = resolve_calibration_config(config)

    manifest_path = resolve_project_path(project_root, config["data"]["manifest_path"])
    image_root = resolve_project_path(project_root, config["data"]["image_root"])
    output_dir = resolve_project_path(project_root, calibration_cfg.output_dir)

    measurements = measure_quality_for_manifest(
        manifest_path=manifest_path,
        image_root=image_root,
        limit=args.limit,
    )
    report = build_calibration_report(
        measurements,
        blur_percentile=calibration_cfg.blur_percentile,
        brightness_percentile=calibration_cfg.brightness_percentile,
        current_blur_threshold=float(config["quality"]["blur_score_min"]),
        current_brightness_threshold=float(config["quality"]["brightness_mean_min"]),
    )
    measurements_path, report_path = write_calibration_outputs(
        measurements=measurements,
        report=report,
        output_dir=output_dir,
    )

    print(format_report_excerpt(report))
    print(f"measurements_path: {measurements_path}")
    print(f"report_path: {report_path}")


if __name__ == "__main__":
    main()
