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
            "Include Messidor external test set (split=external_test). "
            "Expects data/raw/Messidor/messidor_data.csv and images in "
            "data/raw/Messidor/images/."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = Path(args.project_root).resolve()
    raw_root = project_root / args.raw_root
    output_path = project_root / args.output

    summary = write_manifest(
        raw_root=raw_root,
        output_path=output_path,
        include_messidor=args.include_messidor,
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
