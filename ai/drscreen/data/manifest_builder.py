from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd


@dataclass(frozen=True, slots=True)
class ManifestSummary:
    rows: int
    train_rows: int
    val_rows: int
    test_rows: int
    external_test_rows: int
    domains: dict[str, int]


def binary_label_from_grade(grade: int) -> int:
    return 0 if grade == 0 else 1


def _ensure_columns(frame: pd.DataFrame, required: Iterable[str], file_path: Path) -> None:
    missing = sorted(set(required).difference(frame.columns))
    if missing:
        raise ValueError(f"{file_path} is missing columns: {missing}")


def _normalize_columns(frame: pd.DataFrame) -> pd.DataFrame:
    normalized = frame.copy()
    normalized.columns = [str(column).strip() for column in normalized.columns]
    return normalized


def _build_aptos_rows(raw_root: Path) -> list[dict[str, object]]:
    dataset_root = raw_root / "APTOS"
    sources = [
        ("train_1.csv", Path("train_images") / "train_images", "train", "train_1"),
        ("valid.csv", Path("val_images") / "val_images", "val", "valid"),
        ("test.csv", Path("test_images") / "test_images", "test", "test"),
    ]

    rows: list[dict[str, object]] = []
    for csv_name, image_dir, split, source_split in sources:
        csv_path = dataset_root / csv_name
        frame = _normalize_columns(pd.read_csv(csv_path))
        _ensure_columns(frame, {"id_code", "diagnosis"}, csv_path)
        for row in frame.itertuples(index=False):
            image_id = str(row.id_code)
            original_grade = int(row.diagnosis)
            relative_image_path = Path("APTOS") / image_dir / f"{image_id}.png"
            if not (raw_root / relative_image_path).exists():
                raise FileNotFoundError(f"APTOS image not found: {relative_image_path}")
            rows.append(
                {
                    "image_id": image_id,
                    "image_path": relative_image_path.as_posix(),
                    "label": binary_label_from_grade(original_grade),
                    "original_grade": original_grade,
                    "split": split,
                    "domain": "APTOS",
                    "source_split": source_split,
                }
            )
    return rows


def _build_idrid_rows(raw_root: Path) -> list[dict[str, object]]:
    dataset_root = raw_root / "IDRiD"
    grading_root = dataset_root / "B. Disease Grading"
    sources = [
        (
            grading_root / "2. Groundtruths" / "a. IDRiD_Disease Grading_Training Labels.csv",
            grading_root / "1. Original Images" / "a. Training Set",
            "train",
            "training",
        ),
        (
            grading_root / "2. Groundtruths" / "b. IDRiD_Disease Grading_Testing Labels.csv",
            grading_root / "1. Original Images" / "b. Testing Set",
            "test",
            "testing",
        ),
    ]

    rows: list[dict[str, object]] = []
    for csv_path, image_dir, split, source_split in sources:
        frame = _normalize_columns(pd.read_csv(csv_path))
        _ensure_columns(frame, {"Image name", "Retinopathy grade", "Risk of macular edema"}, csv_path)
        for row in frame.to_dict(orient="records"):
            image_id = str(row["Image name"])
            original_grade = int(row["Retinopathy grade"])
            edema_grade = int(row["Risk of macular edema"])
            relative_image_path = image_dir.relative_to(raw_root) / f"{image_id}.jpg"
            if not (raw_root / relative_image_path).exists():
                raise FileNotFoundError(f"IDRiD image not found: {relative_image_path}")
            rows.append(
                {
                    "image_id": image_id,
                    "image_path": relative_image_path.as_posix(),
                    "label": binary_label_from_grade(original_grade),
                    "original_grade": original_grade,
                    "macular_edema_grade": edema_grade,
                    "split": split,
                    "domain": "IDRiD",
                    "source_split": source_split,
                }
            )
    return rows


_MESSIDOR_IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".tif", ".tiff", ".png")


def _resolve_messidor_image_path(image_dir: Path, image_id: str) -> Path | None:
    stem = Path(image_id).stem
    for ext in _MESSIDOR_IMAGE_EXTENSIONS:
        candidate = image_dir / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    return None


def _build_messidor_rows(raw_root: Path, *, split: str = "external_test") -> list[dict[str, object]]:
    dataset_root = raw_root / "Messidor"
    csv_path = dataset_root / "messidor_data.csv"
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Messidor annotation file not found: {csv_path}\n"
            "Expected layout: data/raw/Messidor/messidor_data.csv with columns "
            "'image_id', 'adjudicated_dr_grade', and images in data/raw/Messidor/images/"
        )

    frame = _normalize_columns(pd.read_csv(csv_path))
    _ensure_columns(frame, {"image_id", "adjudicated_dr_grade"}, csv_path)

    image_dir = dataset_root / "images"
    if not image_dir.is_dir():
        image_dir = dataset_root

    rows: list[dict[str, object]] = []
    for row in frame.itertuples(index=False):
        image_id = str(row.image_id)
        original_grade = int(row.adjudicated_dr_grade)
        resolved = _resolve_messidor_image_path(image_dir, image_id)
        if resolved is None:
            raise FileNotFoundError(
                f"Messidor image not found for id '{image_id}' in {image_dir}"
            )
        relative_image_path = resolved.relative_to(raw_root)
        rows.append(
            {
                "image_id": image_id,
                "image_path": relative_image_path.as_posix(),
                "label": binary_label_from_grade(original_grade),
                "original_grade": original_grade,
                "split": split,
                "domain": "Messidor",
                "source_split": "messidor",
            }
        )
    return rows


_DDR_IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")


def _build_ddr_rows(raw_root: Path) -> list[dict[str, object]]:
    dataset_root = raw_root / "ddr"
    csv_path = dataset_root / "DR_grading.csv"
    if not csv_path.exists():
        raise FileNotFoundError(
            f"DDR annotation file not found: {csv_path}\n"
            "Expected layout: data/raw/ddr/DR_grading.csv with columns "
            "'id_code', 'diagnosis', and images in data/raw/ddr/DR_grading/DR_grading/"
        )

    frame = _normalize_columns(pd.read_csv(csv_path))
    _ensure_columns(frame, {"id_code", "diagnosis"}, csv_path)

    image_dir = dataset_root / "DR_grading" / "DR_grading"
    if not image_dir.is_dir():
        raise FileNotFoundError(f"DDR image directory not found: {image_dir}")

    rows: list[dict[str, object]] = []
    missing: list[str] = []
    for row in frame.itertuples(index=False):
        image_id = str(row.id_code)
        original_grade = int(row.diagnosis)
        candidate = image_dir / image_id
        if not candidate.exists():
            missing.append(image_id)
            continue
        relative_image_path = candidate.relative_to(raw_root)
        rows.append(
            {
                "image_id": image_id,
                "image_path": relative_image_path.as_posix(),
                "label": binary_label_from_grade(original_grade),
                "original_grade": original_grade,
                "split": "external_test",
                "domain": "DDR",
                "source_split": "ddr",
            }
        )
    if missing:
        raise FileNotFoundError(
            f"DDR: {len(missing)} images not found in {image_dir}. "
            f"First missing: {missing[0]}"
        )
    return rows


def build_manifest_frame(
    raw_root: str | Path,
    *,
    include_messidor: bool = False,
    messidor_as_train: bool = False,
    include_ddr: bool = False,
) -> pd.DataFrame:
    """Build a manifest DataFrame from the raw dataset root.

    Args:
        raw_root: Path to data/raw/.
        include_messidor: Include Messidor images. When ``messidor_as_train`` is
            False (default), Messidor rows go to ``external_test``; when True
            they go to ``train``.
        messidor_as_train: Move Messidor from external_test into the train split.
            Requires ``include_messidor=True``.
        include_ddr: Include DDR images as ``external_test``.
    """
    raw_root = Path(raw_root)
    rows = [*_build_aptos_rows(raw_root), *_build_idrid_rows(raw_root)]
    if include_messidor:
        messidor_split = "train" if messidor_as_train else "external_test"
        rows.extend(_build_messidor_rows(raw_root, split=messidor_split))
    if include_ddr:
        rows.extend(_build_ddr_rows(raw_root))
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise ValueError("No dataset rows were collected.")

    frame = frame.sort_values(["split", "domain", "image_id"], kind="stable").reset_index(drop=True)
    return frame


def summarize_manifest(frame: pd.DataFrame) -> ManifestSummary:
    split_counts = frame["split"].value_counts().to_dict()
    return ManifestSummary(
        rows=len(frame),
        train_rows=int(split_counts.get("train", 0)),
        val_rows=int(split_counts.get("val", 0)),
        test_rows=int(split_counts.get("test", 0)),
        external_test_rows=int(split_counts.get("external_test", 0)),
        domains={str(key): int(value) for key, value in frame["domain"].value_counts().items()},
    )


def write_manifest(
    raw_root: str | Path,
    output_path: str | Path,
    *,
    include_messidor: bool = False,
    messidor_as_train: bool = False,
    include_ddr: bool = False,
) -> ManifestSummary:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame = build_manifest_frame(
        raw_root,
        include_messidor=include_messidor,
        messidor_as_train=messidor_as_train,
        include_ddr=include_ddr,
    )
    frame.to_csv(output_path, index=False)
    return summarize_manifest(frame)
