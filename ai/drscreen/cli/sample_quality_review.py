from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from drscreen.quality.review import (
    build_flagged_quality_frame,
    build_review_summary,
    sample_review_candidates,
    write_review_outputs,
)
from drscreen.settings import load_app_config, resolve_project_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sample flagged quality cases for manual review.")
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    parser.add_argument(
        "--measurements",
        default=None,
        help="Optional measurement CSV path. Defaults to artifacts/quality/quality_measurements.csv.",
    )
    parser.add_argument(
        "--per-group",
        type=int,
        default=12,
        help="Number of review samples per domain x flag_reason group.",
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
    image_root = resolve_project_path(project_root, config["data"]["image_root"])
    calibration_output_dir = resolve_project_path(
        project_root,
        config["quality"]["calibration"]["output_dir"],
    )
    measurements_path = (
        Path(args.measurements).resolve()
        if args.measurements
        else calibration_output_dir / "quality_measurements.csv"
    )
    review_output_dir = calibration_output_dir / "review"

    measurements = pd.read_csv(measurements_path)
    flagged = build_flagged_quality_frame(
        measurements,
        blur_threshold=float(config["quality"]["blur_score_min"]),
        brightness_threshold=float(config["quality"]["brightness_mean_min"]),
        image_root=image_root,
    )
    sampled = sample_review_candidates(flagged, per_group=args.per_group)
    summary = build_review_summary(
        flagged=flagged,
        sampled=sampled,
        blur_threshold=float(config["quality"]["blur_score_min"]),
        brightness_threshold=float(config["quality"]["brightness_mean_min"]),
        per_group=args.per_group,
    )
    flagged_path, sample_path, summary_path = write_review_outputs(
        flagged=flagged,
        sampled=sampled,
        summary=summary,
        output_dir=review_output_dir,
    )

    print("Quality review sample created")
    print(f"flagged_rows: {summary['flagged_rows']}")
    print(f"sampled_rows: {summary['sampled_rows']}")
    print(f"flagged_path: {flagged_path}")
    print(f"sample_path: {sample_path}")
    print(f"summary_path: {summary_path}")


if __name__ == "__main__":
    main()
