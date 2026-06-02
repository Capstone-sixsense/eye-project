from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from drscreen.data.manifest_builder import (
    rebalance_val_split,
    split_external_into_calibration_holdout,
    summarize_manifest,
    write_manifest,
)


def _parse_domain_quota(value: str | None) -> dict[str, int] | None:
    if not value:
        return None
    quotas: dict[str, int] = {}
    for token in value.split(","):
        item = token.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(
                "Domain quotas must use DOMAIN=N entries, for example: "
                "APTOS=150,IDRiD=150,Messidor=150"
            )
        domain, raw_count = item.split("=", 1)
        quotas[domain.strip()] = int(raw_count.strip())
    return quotas


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
        "--input-manifest",
        help=(
            "Existing manifest to post-process instead of rebuilding from raw data. "
            "Useful for preserving offline-preprocessed image paths."
        ),
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
        "--include-ddr-seg",
        action="store_true",
        default=False,
        help=(
            "Include DDR lesion-segmentation train/val image/mask pairs as "
            "domain='DDR_SEG' training rows. Expects "
            "data/raw/ddr/lesion_segmentation/images/{train,val} and "
            "annotations/{train,val}/{MA,HE,EX,SE}."
        ),
    )
    parser.add_argument(
        "--include-ddr-seg-test",
        action="store_true",
        default=False,
        help=(
            "Also include DDR lesion-segmentation test pairs as split='ddr_seg_test'. "
            "Requires --include-ddr-seg."
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
    parser.add_argument(
        "--split-external-calibration",
        action="store_true",
        default=False,
        help=(
            "Split rows with split='external_test' into deterministic "
            "external_calibration/external_holdout rows."
        ),
    )
    parser.add_argument(
        "--external-calibration-fraction",
        type=float,
        default=0.2,
        help="Fraction of external_test rows assigned to external_calibration.",
    )
    parser.add_argument(
        "--rebalance-val",
        action="store_true",
        default=False,
        help="Create split='val_mixed' from APTOS val plus IDRiD/Messidor train quotas.",
    )
    parser.add_argument(
        "--val-mixed-quota",
        default="APTOS=150,IDRiD=150,Messidor=150",
        help="Comma-separated DOMAIN=N quotas for --rebalance-val.",
    )
    parser.add_argument(
        "--split-seed",
        type=int,
        default=20260524,
        help="Seed used for deterministic calibration/holdout and val_mixed sampling.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.include_tjdr_test and not args.include_tjdr:
        raise ValueError("--include-tjdr-test requires --include-tjdr")
    if args.include_ddr_seg_test and not args.include_ddr_seg:
        raise ValueError("--include-ddr-seg-test requires --include-ddr-seg")
    project_root = Path(args.project_root).resolve()
    raw_root = project_root / args.raw_root
    output_path = project_root / args.output

    if args.input_manifest:
        input_path = project_root / args.input_manifest
        frame = pd.read_csv(input_path)
        if args.split_external_calibration:
            frame = split_external_into_calibration_holdout(
                frame,
                seed=args.split_seed,
                calibration_fraction=args.external_calibration_fraction,
            )
        if args.rebalance_val:
            frame = rebalance_val_split(
                frame,
                seed=args.split_seed,
                per_domain_quota=_parse_domain_quota(args.val_mixed_quota),
            )
        frame = frame.sort_values(["split", "domain", "image_id"], kind="stable").reset_index(drop=True)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(output_path, index=False)
        summary = summarize_manifest(frame)
    else:
        summary = write_manifest(
            raw_root=raw_root,
            output_path=output_path,
            include_messidor=args.include_messidor,
            messidor_as_train=args.messidor_as_train,
            include_ddr=args.include_ddr,
            include_ddr_seg=args.include_ddr_seg,
            include_ddr_seg_test=args.include_ddr_seg_test,
            include_maples=args.include_maples,
            include_tjdr=args.include_tjdr,
            include_tjdr_test=args.include_tjdr_test,
            split_external_calibration=args.split_external_calibration,
            external_calibration_fraction=args.external_calibration_fraction,
            rebalance_val=args.rebalance_val,
            split_seed=args.split_seed,
            val_mixed_quota=_parse_domain_quota(args.val_mixed_quota),
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
    for split, count in sorted(summary.splits.items()):
        print(f"split[{split}]: {count}")
    for domain, count in summary.domains.items():
        print(f"domain[{domain}]: {count}")


if __name__ == "__main__":
    main()
