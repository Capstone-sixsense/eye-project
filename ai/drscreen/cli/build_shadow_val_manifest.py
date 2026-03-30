from __future__ import annotations

import argparse
from pathlib import Path

from drscreen.data.manifest_variants import write_shadow_validation_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a manifest variant that augments val with a domain-matched shadow holdout."
    )
    parser.add_argument(
        "--project-root",
        default=".",
        help="Project root that contains data/processed.",
    )
    parser.add_argument(
        "--input",
        default="data/processed/manifest.csv",
        help="Input manifest path relative to project root.",
    )
    parser.add_argument(
        "--output",
        default="data/processed/manifest_val_plus_idrid_shadow.csv",
        help="Output manifest path relative to project root.",
    )
    parser.add_argument("--domain", default="IDRiD", help="Domain to copy from train into val.")
    parser.add_argument("--source-split", default="train", help="Split to sample from.")
    parser.add_argument("--destination-split", default="val", help="Split to augment.")
    parser.add_argument(
        "--reference-split",
        default="test",
        help="Split whose domain+label counts will be mirrored.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Sampling seed.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = Path(args.project_root).resolve()
    input_path = project_root / args.input
    output_path = project_root / args.output

    summary = write_shadow_validation_manifest(
        manifest_path=input_path,
        output_path=output_path,
        domain=args.domain,
        source_split=args.source_split,
        destination_split=args.destination_split,
        reference_split=args.reference_split,
        seed=args.seed,
    )

    print("Shadow validation manifest created")
    print(f"project_root: {project_root}")
    print(f"input_path:   {input_path}")
    print(f"output_path:  {output_path}")
    print(f"rows:         {summary.rows}")
    for split, count in summary.split_counts.items():
        print(f"split[{split}]: {count}")
    for domain, counts in summary.domain_split_counts.items():
        print(f"domain_split[{domain}]: {counts}")
    for domain, counts in summary.domain_label_counts.items():
        print(f"domain_label[{domain}]: {counts}")


if __name__ == "__main__":
    main()
