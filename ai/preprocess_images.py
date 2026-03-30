"""Offline preprocessing script.

Applies Ben Graham + Circular Crop + resize (data.preprocess_size) to every
image in the manifest and saves the results to data/processed/images/ as PNG.
Training then loads these pre-processed images with use_preprocessing: false.
Inference applies the same FundusPreprocess(output_size=preprocess_size) live,
ensuring train/infer parity.

Run:
    python preprocess_images.py [--config configs/base.yaml] [--workers N]
"""
from __future__ import annotations

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from PIL import Image

from drscreen.data.transforms import FundusPreprocess
from drscreen.settings import load_app_config

RAW_ROOT = Path(__file__).parent / "data" / "raw"
MANIFEST_PATH = Path(__file__).parent / "data" / "processed" / "manifest.csv"
OUTPUT_ROOT = Path(__file__).parent / "data" / "raw" / "processed" / "images"

_preprocessor: FundusPreprocess | None = None


def _process_one(args: tuple[str, Path, Path]) -> tuple[str, bool, str]:
    image_path_rel, raw_root, output_root = args
    src = raw_root / image_path_rel
    dst = output_root / Path(image_path_rel).with_suffix(".png").name
    if dst.exists():
        return image_path_rel, True, "skipped"
    try:
        with Image.open(src) as img:
            processed = _preprocessor(img)
        dst.parent.mkdir(parents=True, exist_ok=True)
        processed.save(dst, format="PNG", optimize=False)
        return image_path_rel, True, "ok"
    except Exception as exc:
        return image_path_rel, False, str(exc)


def main() -> None:
    global _preprocessor
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base.yaml", help="Path to YAML config.")
    parser.add_argument("--workers", type=int, default=1, help="Parallel workers (default 1 for Windows).")
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    base_path = config_path.parent / "base.yaml"
    config = load_app_config(config_path, base_path=base_path if config_path.name != "base.yaml" and base_path.exists() else None)
    preprocess_size = int(config["data"].get("preprocess_size", 0)) or None
    _preprocessor = FundusPreprocess(output_size=preprocess_size)
    print(f"Preprocessor: Ben Graham + Circular Crop, output_size={preprocess_size}")

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    frame = pd.read_csv(MANIFEST_PATH)
    image_paths = frame["image_path"].tolist()
    total = len(image_paths)
    print(f"Images to process: {total}")
    print(f"Output dir: {OUTPUT_ROOT}")

    tasks = [(p, RAW_ROOT, OUTPUT_ROOT) for p in image_paths]
    done = 0
    errors: list[str] = []

    if args.workers <= 1:
        for task in tasks:
            rel, ok, msg = _process_one(task)
            done += 1
            if not ok:
                errors.append(f"{rel}: {msg}")
            if done % 100 == 0 or done == total:
                print(f"  {done}/{total}", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(_process_one, t): t[0] for t in tasks}
            for future in as_completed(futures):
                rel, ok, msg = future.result()
                done += 1
                if not ok:
                    errors.append(f"{rel}: {msg}")
                if done % 100 == 0 or done == total:
                    print(f"  {done}/{total}", flush=True)

    print(f"\nDone: {total - len(errors)} ok, {len(errors)} errors")
    if errors:
        print("Errors:")
        for e in errors[:20]:
            print(f"  {e}")
        sys.exit(1)

    # Update manifest to point to processed images
    updated = frame.copy()
    updated["image_path"] = updated["image_path"].apply(
        lambda p: "processed/images/" + Path(p).with_suffix(".png").name
    )
    out_manifest = MANIFEST_PATH.parent / "manifest_preprocessed.csv"
    updated.to_csv(out_manifest, index=False)
    print(f"\nUpdated manifest: {out_manifest}")
    print("Set data.manifest_path: data/processed/manifest_preprocessed.csv in base.yaml to use it.")


if __name__ == "__main__":
    main()
