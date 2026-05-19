from __future__ import annotations

import argparse
from pathlib import Path

from drscreen.data.manifest_builder import write_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build manifest.csv from APTOS and IDRiD datasets.")
    parser.add_argument(
        "--project-root",
        default=".",
        help="Project root that contains data/raw and data/processed.",
    )
    parser.add_argument(
        "--raw-root",
        default="data/raw",
        help="Raw dataset root relative to project root.",
    )
    parser.add_argument(
        "--output",
        default="data/processed/manifest.csv",
        help="Output manifest path relative to project root.",
    )
    parser.add_argument(
        "--include-messidor",
        action="store_true",
        default=False,
        help=(
            "Include Messidor images. By default placed in external_test. "
            "Use --messidor-as-train to include in train split instead."
        ),
    )
    parser.add_argument(
        "--messidor-as-train",
        action="store_true",
        default=False,
        help=(
            "Move Messidor from external_test into the train split. "
            "Requires --include-messidor."
        ),
    )
    parser.add_argument(
        "--include-ddr",
        action="store_true",
        default=False,
        help=(
            "Include DDR dataset as external_test. "
            "Expects data/raw/ddr/DR_grading.csv and images in "
            "data/raw/ddr/DR_grading/DR_grading/."
        ),
    )
    parser.add_argument(
        "--include-maples",
        action="store_true",
        default=False,
        help=(
            "Include MAPLES-DR training rows pointing at the MESSIDOR images "
            "they annotate. R1+ rows are domain='MAPLES' for mask supervision; "
            "R0 rows are domain='Messidor' so empty masks do not act as pixel "
            "supervision. Test split (60 rows) is reserved for clean eval. "
            "Requires data/raw/MAPLES-DR/MAPLES-DR/train/diagnosis.csv and "
            "the corresponding MESSIDOR images under data/raw/Messidor/images/."
        ),
    )
    parser.add_argument(
        "--include-tjdr",
        action="store_true",
        default=False,
        help=(
            "Include TJDR train image/annotation pairs as domain='TJDR' rows. "
            "TJDR test rows are excluded by default so the existing IDRiD "
            "split='test' evaluation is not mixed with TJDR."
        ),
    )
    parser.add_argument(
        "--include-tjdr-test",
        action="store_true",
        default=False,
        help=(
            "Also include TJDR test pairs as split='tjdr_test'. "
            "Requires --include-tjdr."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.include_tjdr_test and not args.include_tjdr:
        raise ValueError("--include-tjdr-test requires --include-tjdr")
    project_root = Path(args.project_root).resolve()
    raw_root = project_root / args.raw_root
    output_path = project_root / args.output

    summary = write_manifest(
        raw_root=raw_root,
        output_path=output_path,
        include_messidor=args.include_messidor,
        messidor_as_train=args.messidor_as_train,
        include_ddr=args.include_ddr,
        include_maples=args.include_maples,
        include_tjdr=args.include_tjdr,
        include_tjdr_test=args.include_tjdr_test,
    )

    print("Manifest created")
    print(f"project_root:       {project_root}")
    print(f"raw_root:           {raw_root}")
    print(f"output_path:        {output_path}")
    print(f"rows:               {summary.rows}")
    print(f"train_rows:         {summary.train_rows}")
    print(f"val_rows:           {summary.val_rows}")
    print(f"test_rows:          {summary.test_rows}")
    print(f"external_test_rows: {summary.external_test_rows}")
    for domain, count in summary.domains.items():
        print(f"domain[{domain}]: {count}")


if __name__ == "__main__":
    main()
