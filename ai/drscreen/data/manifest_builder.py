from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
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


_MAPLES_R_GRADE_RE = re.compile(r"^R(\d+)$")


def _build_maples_rows(raw_root: Path) -> list[dict[str, object]]:
    """Add MAPLES-DR training rows pointing at MESSIDOR images.

    MAPLES-DR provides pixel-level lesion masks for a subset of MESSIDOR
    images. For the binary DR screening task we use the diagnosis CSV's R
    grade (R0 → 0, R1+ → 1, matching the standard Messidor R-scale binary
    mapping). The image path is the MESSIDOR file the mask annotates. R1+
    rows use ``domain='MAPLES'`` so a MAPLES-aware mask provider can dispatch
    on them. R0 rows are kept as ``domain='Messidor'`` duplicates because
    their lesion masks are effectively empty and should not contribute
    negative pixel-level supervision.

    Only the MAPLES-DR ``train`` split is exported here. The ``test`` split
    is reserved as a clean external evaluation cohort.
    """
    maples_root = raw_root / "MAPLES-DR" / "MAPLES-DR" / "train"
    diag_csv = maples_root / "diagnosis.csv"
    if not diag_csv.exists():
        raise FileNotFoundError(
            f"MAPLES-DR train diagnosis CSV not found: {diag_csv}\n"
            "Expected layout: data/raw/MAPLES-DR/MAPLES-DR/train/diagnosis.csv "
            "with columns 'name, DR, ME'."
        )

    messidor_image_dir = raw_root / "Messidor" / "images"
    if not messidor_image_dir.is_dir():
        messidor_image_dir = raw_root / "Messidor"

    frame = _normalize_columns(pd.read_csv(diag_csv))
    _ensure_columns(frame, {"name", "DR"}, diag_csv)

    rows: list[dict[str, object]] = []
    for row in frame.itertuples(index=False):
        stem = str(row.name)
        dr_token = str(row.DR).strip()
        m = _MAPLES_R_GRADE_RE.match(dr_token)
        if not m:
            continue
        original_grade = int(m.group(1))
        resolved = _resolve_messidor_image_path(messidor_image_dir, stem)
        if resolved is None:
            # MAPLES references a MESSIDOR image not present locally — skip.
            continue
        relative_image_path = resolved.relative_to(raw_root)
        label = binary_label_from_grade(original_grade)
        rows.append(
            {
                "image_id": stem,
                "image_path": relative_image_path.as_posix(),
                "label": label,
                "original_grade": original_grade,
                "split": "train",
                "domain": "MAPLES" if label == 1 else "Messidor",
                "source_split": "maples_train" if label == 1 else "maples_train_no_mask",
            }
        )
    return rows


_DDR_IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")


def _tjdr_has_lesion(mask_path: Path) -> bool:
    from PIL import Image as PILImage

    arr = np.array(PILImage.open(mask_path), dtype=np.uint8)
    if arr.ndim == 3:
        arr = arr[..., 0]
    return bool((arr > 0).any())


def _build_tjdr_rows(raw_root: Path, *, include_test: bool = False) -> list[dict[str, object]]:
    """Add TJDR lesion-mask rows.

    TJDR annotations are palette PNGs with integer labels:
    0=background, 1=EX, 2=HE, 3=MA, 4=SE. The manifest rows point at the
    raw fundus images; the mask provider performs label-to-channel mapping.

    Only the TJDR train split is exported by default. Keeping TJDR test out of
    ``split='test'`` avoids mixing it with the existing IDRiD test evaluation
    split. Pass ``include_test=True`` only when a dedicated TJDR split is needed;
    those rows are written as ``split='tjdr_test'``.
    """
    dataset_root = raw_root / "TJDR"
    rows: list[dict[str, object]] = []
    split_specs = [("train", "train")]
    if include_test:
        split_specs.append(("test", "tjdr_test"))

    for source_split, manifest_split in split_specs:
        image_dir = dataset_root / source_split / "image"
        ann_dir = dataset_root / source_split / "annotation"
        if not image_dir.is_dir():
            raise FileNotFoundError(f"TJDR image directory not found: {image_dir}")
        if not ann_dir.is_dir():
            raise FileNotFoundError(f"TJDR annotation directory not found: {ann_dir}")

        images = {p.stem: p for p in sorted(image_dir.glob("*.png"))}
        annotations = {p.stem: p for p in sorted(ann_dir.glob("*.png"))}
        missing_annotations = sorted(set(images) - set(annotations))
        missing_images = sorted(set(annotations) - set(images))
        if missing_annotations or missing_images:
            raise FileNotFoundError(
                "TJDR image/annotation pairs are incomplete for "
                f"{source_split}: missing_annotations={len(missing_annotations)}, "
                f"missing_images={len(missing_images)}"
            )

        for stem, image_path in images.items():
            ann_path = annotations[stem]
            has_lesion = _tjdr_has_lesion(ann_path)
            relative_image_path = image_path.relative_to(raw_root)
            rows.append(
                {
                    "image_id": stem,
                    "image_path": relative_image_path.as_posix(),
                    "label": 1 if has_lesion else 0,
                    "original_grade": 1 if has_lesion else 0,
                    "split": manifest_split,
                    "domain": "TJDR",
                    "source_split": f"tjdr_{source_split}",
                }
            )
    return rows


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
    include_maples: bool = False,
    include_tjdr: bool = False,
    include_tjdr_test: bool = False,
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
    if include_maples:
        rows.extend(_build_maples_rows(raw_root))
    if include_tjdr:
        rows.extend(_build_tjdr_rows(raw_root, include_test=include_tjdr_test))
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
    include_maples: bool = False,
    include_tjdr: bool = False,
    include_tjdr_test: bool = False,
) -> ManifestSummary:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame = build_manifest_frame(
        raw_root,
        include_messidor=include_messidor,
        messidor_as_train=messidor_as_train,
        include_ddr=include_ddr,
        include_maples=include_maples,
        include_tjdr=include_tjdr,
        include_tjdr_test=include_tjdr_test,
    )
    frame.to_csv(output_path, index=False)
    return summarize_manifest(frame)
