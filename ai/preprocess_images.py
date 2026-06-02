"""Offline preprocessing script.

Applies content-aware border crop + Ben Graham normalization + resize (data.preprocess_size)
to every image in the manifest. No quality filtering -- all images are included.

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

from drscreen.data.transforms import FundusPreprocess, preprocess_kwargs_from_config
from drscreen.settings import load_app_config

PROJECT_ROOT = Path(__file__).parent
RAW_ROOT = PROJECT_ROOT / "data" / "raw"
MANIFEST_PATH = PROJECT_ROOT / "data" / "processed" / "manifest.csv"
OUTPUT_ROOT = PROJECT_ROOT / "data" / "raw" / "processed" / "images"

_preprocessor: FundusPreprocess | None = None
_expected_size: int | None = None


def _resolve_under_project(project_root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else project_root / path


def _init_worker(
    preprocess_size: int | None,
    use_align: bool,
    preprocess_options: dict | None = None,
) -> None:
    global _preprocessor, _expected_size
    _preprocessor = FundusPreprocess(
        output_size=preprocess_size,
        align=use_align,
        **(preprocess_options or {}),
    )
    _expected_size = preprocess_size


def _existing_output_matches(path: Path) -> bool:
    if not path.exists():
        return False
    if _expected_size is None:
        return True
    try:
        with Image.open(path) as image:
            return image.size == (_expected_size, _expected_size)
    except Exception:
        return False


def _process_one(args: tuple[str, Path, Path, bool]) -> tuple[str, bool, str]:
    image_path_rel, raw_root, output_root, force = args
    src = raw_root / image_path_rel
    dst = output_root / Path(image_path_rel).with_suffix(".png")
    if not force and _existing_output_matches(dst):
        return image_path_rel, True, "skipped"
    try:
        if _preprocessor is None:
            raise RuntimeError("Preprocessor is not initialized.")
        with Image.open(src) as img:
            processed = _preprocessor(img.convert("RGB"))
        dst.parent.mkdir(parents=True, exist_ok=True)
        processed.save(dst, format="PNG", optimize=False)
        return image_path_rel, True, "ok"
    except Exception as exc:
        return image_path_rel, False, str(exc)


def main() -> None:
    global _preprocessor

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base.yaml", help="Path to YAML config.")
    parser.add_argument("--project-root", default=PROJECT_ROOT, help="AI project root.")
    parser.add_argument("--raw-root", default="data/raw", help="Raw image root relative to project root.")
    parser.add_argument(
        "--manifest",
        default="data/processed/manifest.csv",
        help="Source manifest relative to project root.",
    )
    parser.add_argument(
        "--output-root",
        default="data/raw/processed/images",
        help="Preprocessed image output dir. Must stay under raw-root for training manifests.",
    )
    parser.add_argument(
        "--out-manifest",
        default="data/processed/manifest_preprocessed.csv",
        help="Output manifest relative to project root.",
    )
    parser.add_argument("--workers", type=int, default=1, help="Parallel workers (default 1 for Windows).")
    parser.add_argument("--force", action="store_true", help="Reprocess existing outputs.")
    parser.add_argument("--limit", type=int, help="Process only the first N rows for smoke tests.")
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    config_path = _resolve_under_project(project_root, args.config).resolve()
    base_path = config_path.parent / "base.yaml"
    config = load_app_config(
        config_path,
        base_path=base_path if config_path.name != "base.yaml" and base_path.exists() else None,
    )

    preprocess_size = int(config["data"].get("preprocess_size", 0)) or None
    use_align = bool(config["data"].get("use_align", False))
    preprocess_options = preprocess_kwargs_from_config(config.get("data", {}))
    _init_worker(preprocess_size, use_align, preprocess_options)

    raw_root = _resolve_under_project(project_root, args.raw_root)
    manifest_path = _resolve_under_project(project_root, args.manifest)
    output_root = _resolve_under_project(project_root, args.output_root)
    out_manifest = _resolve_under_project(project_root, args.out_manifest)
    try:
        output_root.relative_to(raw_root)
    except ValueError as exc:
        raise ValueError(
            f"--output-root must be inside --raw-root so manifest image_path values "
            f"can stay relative to {raw_root}: {output_root}"
        ) from exc

    preprocess_mode = str(preprocess_options.get("preprocess_mode", "contentcrop"))
    print(
        f"Preprocessor: {preprocess_mode} + Ben Graham, "
        f"output_size={preprocess_size}, align={use_align}, options={preprocess_options}"
    )

    output_root.mkdir(parents=True, exist_ok=True)

    frame = pd.read_csv(manifest_path)
    if args.limit is not None:
        frame = frame.head(max(args.limit, 0)).copy()
    image_paths = frame["image_path"].tolist()
    total = len(image_paths)
    print(f"Images to process: {total}")
    print(f"Input manifest: {manifest_path}")
    print(f"Output dir: {output_root}")
    print(f"Output manifest: {out_manifest}")

    tasks = [(p, raw_root, output_root, bool(args.force)) for p in image_paths]
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
        with ProcessPoolExecutor(
            max_workers=args.workers,
            initializer=_init_worker,
            initargs=(preprocess_size, use_align, preprocess_options),
        ) as pool:
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

    updated = frame.copy()
    output_prefix = output_root.relative_to(raw_root)
    updated["image_path"] = updated["image_path"].apply(
        lambda p: (output_prefix / Path(p).with_suffix(".png")).as_posix()
    )
    out_manifest.parent.mkdir(parents=True, exist_ok=True)
    updated.to_csv(out_manifest, index=False)
    print(f"\nUpdated manifest: {out_manifest} ({len(updated)} rows)")
    print(f"Set data.manifest_path: {out_manifest.relative_to(project_root).as_posix()} to use it.")


if __name__ == "__main__":
    main()
