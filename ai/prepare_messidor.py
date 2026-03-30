"""Prepare Messidor dataset for manifest builder.

Reads the 12 Annotation_BaseXX.xls files and:
  1. Creates messidor_data.csv with columns image_id, adjudicated_dr_grade
  2. Hardlinks all .tif images into a flat images/ directory

Run from any working directory:
    python prepare_messidor.py
"""
from __future__ import annotations

import csv
import os
import sys
from pathlib import Path

import xlrd

MESSIDOR_ROOT = Path(__file__).parent / "data" / "raw" / "messidor"
BASE_NAMES = [
    "Base11", "Base12", "Base13", "Base14",
    "Base21", "Base22", "Base23", "Base24",
    "Base31", "Base32", "Base33", "Base34",
]


def prepare() -> None:
    images_dir = MESSIDOR_ROOT / "images"
    images_dir.mkdir(exist_ok=True)

    rows: list[dict[str, object]] = []
    missing_images: list[str] = []

    for base in BASE_NAMES:
        xls_path = MESSIDOR_ROOT / f"Annotation_{base}.xls"
        if not xls_path.exists():
            print(f"[WARN] XLS not found: {xls_path}", file=sys.stderr)
            continue

        wb = xlrd.open_workbook(str(xls_path))
        sh = wb.sheet_by_index(0)
        headers = [str(c).strip() for c in sh.row_values(0)]
        img_col = headers.index("Image name")
        grade_col = headers.index("Retinopathy grade")

        # Some bases are double-nested (Base11/Base11/), others are flat (Base12/)
        nested = MESSIDOR_ROOT / base / base
        img_src_dir = nested if nested.is_dir() else MESSIDOR_ROOT / base

        for r in range(1, sh.nrows):
            row_vals = sh.row_values(r)
            img_name = str(row_vals[img_col]).strip()
            if not img_name:
                continue
            grade = int(row_vals[grade_col])

            src = img_src_dir / img_name
            dst = images_dir / img_name

            if not src.exists():
                missing_images.append(str(src))
            elif not dst.exists():
                os.link(str(src), str(dst))

            rows.append({"image_id": img_name, "adjudicated_dr_grade": grade})

    csv_path = MESSIDOR_ROOT / "messidor_data.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["image_id", "adjudicated_dr_grade"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"CSV written: {csv_path} ({len(rows)} rows)")
    print(f"Images dir:  {images_dir} ({sum(1 for _ in images_dir.iterdir())} files)")

    if missing_images:
        print(f"\n[WARN] {len(missing_images)} images not found:")
        for p in missing_images[:10]:
            print(f"  {p}")
        if len(missing_images) > 10:
            print(f"  ... and {len(missing_images) - 10} more")


if __name__ == "__main__":
    prepare()
