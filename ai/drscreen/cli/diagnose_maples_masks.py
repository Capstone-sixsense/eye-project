"""MAPLES-DR 학습 마스크 배선(wiring)과 픽셀 희소성 진단.

MAPLES 마스크가 manifest 행에 제대로 연결됐는지, 그리고 마스크가 사실상 비어 있는
(R0 등 픽셀이 거의 없는) 경우가 얼마나 되는지 점검한다. 빈 마스크가 잘못된 음성
supervision을 주는 문제(AI_HANDOFF Phase 4-C)를 잡기 위한 도구다.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from drscreen.data.mask_providers import MAPLESTrainMaskProvider

_R_GRADE_RE = re.compile(r"^R(\d+)$")
_CHANNEL_NAMES = ("MA", "HE", "EX", "CWS")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Diagnose MAPLES-DR training mask wiring and pixel sparsity."
    )
    parser.add_argument(
        "--manifest",
        default="data/processed/manifest_with_maples_preprocessed.csv",
        help="Manifest CSV containing domain='MAPLES' rows.",
    )
    parser.add_argument(
        "--maples-root",
        default="data/raw/MAPLES-DR/AdditionalData",
        help="MAPLES-DR AdditionalData root containing annotations/.",
    )
    parser.add_argument(
        "--maples-diagnosis",
        default="data/raw/MAPLES-DR/MAPLES-DR/train/diagnosis.csv",
        help="MAPLES-DR train diagnosis CSV with name, DR, ME columns.",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=512,
        help="Mask size passed to MAPLESTrainMaskProvider (default: 512).",
    )
    parser.add_argument(
        "--out",
        default=".omc/research/phase4c_d1_maples_mask_stats.json",
        help="Output JSON path.",
    )
    return parser.parse_args()


def _parse_r_grade(value: Any) -> int | None:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    text = str(value).strip()
    match = _R_GRADE_RE.match(text)
    if match:
        return int(match.group(1))
    try:
        return int(float(text))
    except ValueError:
        return None


def _grade_label(grade: int | None) -> str:
    if grade is None:
        return "unknown"
    return f"R{grade}"


def _grade_bucket(grade: int | None) -> str:
    if grade is None:
        return "unknown"
    return "R0" if grade == 0 else "R1+"


def _safe_counts(series: pd.Series) -> dict[str, int]:
    return {
        str(key): int(value)
        for key, value in series.fillna("nan").astype(str).value_counts().sort_index().items()
    }


def _stats(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {
            "n": 0,
            "mean": None,
            "median": None,
            "p10": None,
            "p90": None,
            "min": None,
            "max": None,
        }
    arr = np.asarray(values, dtype=np.float64)
    return {
        "n": int(arr.size),
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "p10": float(np.percentile(arr, 10)),
        "p90": float(np.percentile(arr, 90)),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def _nested_stats(rows: list[dict[str, Any]], *, key: str) -> dict[str, Any]:
    return {
        "union": _stats([float(row[key]["union"]) for row in rows]),
        "by_channel": {
            channel: _stats([float(row[key]["by_channel"][channel]) for row in rows])
            for channel in _CHANNEL_NAMES
        },
    }


def _load_diagnosis_grades(path: Path) -> dict[str, int]:
    frame = pd.read_csv(path)
    if "name" not in frame.columns or "DR" not in frame.columns:
        raise ValueError(f"MAPLES diagnosis CSV must contain name and DR columns: {path}")
    grades: dict[str, int] = {}
    for row in frame.itertuples(index=False):
        grade = _parse_r_grade(row.DR)
        if grade is not None:
            grades[str(row.name)] = grade
    return grades


def _summarize_group(rows: list[dict[str, Any]]) -> dict[str, Any]:
    valid_rows = [row for row in rows if row["valid"]]
    return {
        "rows": len(rows),
        "valid_rows": len(valid_rows),
        "valid_rate": (len(valid_rows) / len(rows)) if rows else None,
        "pixel_ratio_valid_rows": _nested_stats(valid_rows, key="pixel_ratio"),
    }


def main() -> None:
    args = _parse_args()
    manifest_path = Path(args.manifest)
    maples_root = Path(args.maples_root)
    diagnosis_path = Path(args.maples_diagnosis)
    output_path = Path(args.out)

    manifest = pd.read_csv(manifest_path)
    required = {"image_path", "domain"}
    missing = required - set(manifest.columns)
    if missing:
        raise ValueError(f"Manifest missing required columns {sorted(missing)}: {manifest_path}")

    diagnosis_grades = _load_diagnosis_grades(diagnosis_path)
    provider = MAPLESTrainMaskProvider(maples_root / "annotations", channels=4)
    size = int(args.size)
    denom = float(size * size)

    maples_rows = manifest[manifest["domain"].astype(str) == "MAPLES"].copy()
    row_outputs: list[dict[str, Any]] = []
    for row in maples_rows.itertuples(index=False):
        image_path = str(row.image_path)
        stem = Path(image_path).stem
        manifest_grade = _parse_r_grade(getattr(row, "original_grade", None))
        diagnosis_grade = diagnosis_grades.get(stem)
        grade = manifest_grade if manifest_grade is not None else diagnosis_grade
        mask, valid = provider.load(image_path, "MAPLES", size)
        channel_ratios = {
            channel: float(mask[i].sum().item() / denom)
            for i, channel in enumerate(_CHANNEL_NAMES)
        }
        union_ratio = float(mask.amax(dim=0).sum().item() / denom)
        row_outputs.append(
            {
                "image_id": str(getattr(row, "image_id", stem)),
                "image_path": image_path,
                "manifest_grade": manifest_grade,
                "diagnosis_grade": diagnosis_grade,
                "grade": grade,
                "grade_label": _grade_label(grade),
                "grade_bucket": _grade_bucket(grade),
                "valid": bool(valid),
                "label": int(row.label) if hasattr(row, "label") else None,
                "pixel_ratio": {
                    "union": union_ratio,
                    "by_channel": channel_ratios,
                },
            }
        )

    by_grade = {
        grade: _summarize_group([row for row in row_outputs if row["grade_label"] == grade])
        for grade in sorted({row["grade_label"] for row in row_outputs})
    }
    by_bucket = {
        bucket: _summarize_group([row for row in row_outputs if row["grade_bucket"] == bucket])
        for bucket in ("R0", "R1+", "unknown")
        if any(row["grade_bucket"] == bucket for row in row_outputs)
    }
    valid_rows = [row for row in row_outputs if row["valid"]]

    output = {
        "manifest": str(manifest_path),
        "maples_root": str(maples_root),
        "maples_diagnosis": str(diagnosis_path),
        "mask_size": size,
        "manifest_summary": {
            "total_rows": int(len(manifest)),
            "domain_counts": _safe_counts(manifest["domain"]),
            "split_counts": _safe_counts(manifest["split"]) if "split" in manifest.columns else {},
            "maples_rows": int(len(maples_rows)),
            "maples_by_original_grade": (
                _safe_counts(maples_rows["original_grade"])
                if "original_grade" in maples_rows.columns else {}
            ),
        },
        "valid_rate": (len(valid_rows) / len(row_outputs)) if row_outputs else None,
        "pixel_stats": {
            "all_valid": _nested_stats(valid_rows, key="pixel_ratio"),
            "by_grade": by_grade,
            "R0": by_bucket.get("R0", _summarize_group([])),
            "R1+": by_bucket.get("R1+", _summarize_group([])),
        },
        "rows": row_outputs,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")

    r0_mean = output["pixel_stats"]["R0"]["pixel_ratio_valid_rows"]["union"]["mean"]
    r1p_mean = output["pixel_stats"]["R1+"]["pixel_ratio_valid_rows"]["union"]["mean"]
    print(f"Manifest rows      : {len(manifest)}")
    print(f"MAPLES rows        : {len(row_outputs)}")
    print(f"MAPLES valid rate  : {output['valid_rate']:.4f}" if output["valid_rate"] is not None else "MAPLES valid rate  : N/A")
    print(f"R0 union mean      : {r0_mean}")
    print(f"R1+ union mean     : {r1p_mean}")
    print(f"Saved              : {output_path}")


if __name__ == "__main__":
    main()
