"""Phase 4-F S0: v31 lesion-presence probe and concept-label export.

This CLI is intentionally diagnostic. It tests whether the active v31
classifier block feature linearly separates lesion presence after domain
stratification, then writes a concept-label table for grounded-classifier
experiments.

(한글 요약) v31 분류기의 블록 특징이 '병변 유무'를 선형으로 구분하는지(도메인 층화 후)
점검하고, grounded-classifier 실험용 개념 라벨 표를 내보낸다.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
from dataclasses import replace
from pathlib import Path

import numpy as np
import yaml
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from drscreen.cli.diagnose_shortcut_audit import (
    BlockFeatureExtractor,
    ImageItem,
    _idrid_mask_items,
    _image_candidates,
    _maples_mask_items,
)
from drscreen.infer.service import InferenceSession
from drscreen.settings import resolve_project_path
from drscreen.xai.iou import LESION_CODES, load_lesion_masks, load_maples_masks


def _relative_to_data_root(project_root: Path, path: Path) -> str:
    data_root = project_root / "data/raw"
    try:
        return path.resolve().relative_to(data_root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _read_manifest_rows(project_root: Path, manifest_path: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    data_root = project_root / "data/raw"
    with manifest_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            image_path = data_root / str(row.get("image_path", "")).strip()
            if not image_path.exists():
                continue
            row = dict(row)
            row["_resolved_image_path"] = str(image_path)
            rows.append(row)
    return rows


def _manifest_normal_items(
    project_root: Path,
    manifest_rows: list[dict[str, str]],
    *,
    domain: str | None = None,
    domains: set[str] | None = None,
    split_values: set[str] | None = None,
) -> list[ImageItem]:
    allowed_domains = domains if domains is not None else ({domain} if domain is not None else None)
    items: list[ImageItem] = []
    for row in manifest_rows:
        row_domain = str(row.get("domain", "")).strip()
        if allowed_domains is not None and row_domain not in allowed_domains:
            continue
        if split_values is not None and str(row.get("split", "")).strip() not in split_values:
            continue
        if str(row.get("label", "")).strip() != "0":
            continue
        image_path = Path(row["_resolved_image_path"])
        items.append(
            ImageItem(
                path=image_path,
                domain=row_domain,
                split=str(row.get("split", "")),
                image_id=str(row.get("image_id", image_path.stem)),
                label=0,
            )
        )
    return items


def _image_rgb_mean(path: Path, size: int = 128) -> np.ndarray:
    with Image.open(path) as image:
        arr = np.asarray(image.convert("RGB").resize((size, size), Image.BILINEAR), dtype=np.float32)
    return arr.reshape(-1, 3).mean(axis=0)


def _select_color_matched_normals(
    positives: list[ImageItem],
    candidates: list[ImageItem],
    *,
    need: int,
) -> list[ImageItem]:
    if need <= 0 or not candidates:
        return []
    pos_means = [_image_rgb_mean(item.path) for item in positives[: min(len(positives), 40)]]
    target = np.mean(np.stack(pos_means, axis=0), axis=0) if pos_means else np.zeros(3, dtype=np.float32)
    scored: list[tuple[float, ImageItem]] = []
    for item in candidates:
        dist = float(np.linalg.norm(_image_rgb_mean(item.path) - target))
        scored.append((dist, item))
    scored.sort(key=lambda pair: pair[0])
    return [replace(item, domain="MESSIDOR_to_MAPLES_normal") for _, item in scored[:need]]


def _cap_balanced_binary(
    positives: list[ImageItem],
    normals: list[ImageItem],
    *,
    max_per_class: int,
    seed: int,
) -> list[ImageItem]:
    rng = random.Random(seed)
    positives = [replace(item, label=1) for item in positives]
    normals = [replace(item, label=0) for item in normals]
    rng.shuffle(positives)
    rng.shuffle(normals)
    n = min(max_per_class, len(positives), len(normals))
    selected = positives[:n] + normals[:n]
    rng.shuffle(selected)
    return selected


def _ci_from_values(values: list[float], *, bootstrap: int, seed: int) -> dict[str, float | None]:
    clean = np.asarray([v for v in values if v is not None and np.isfinite(v)], dtype=np.float64)
    if clean.size == 0:
        return {"mean": None, "std": None, "ci95_low": None, "ci95_high": None}
    if clean.size == 1:
        value = float(clean[0])
        return {"mean": value, "std": 0.0, "ci95_low": value, "ci95_high": value}
    rng = np.random.default_rng(seed)
    samples = rng.choice(clean, size=(bootstrap, clean.size), replace=True).mean(axis=1)
    return {
        "mean": float(clean.mean()),
        "std": float(clean.std(ddof=1)),
        "ci95_low": float(np.percentile(samples, 2.5)),
        "ci95_high": float(np.percentile(samples, 97.5)),
    }


def _cv_probe(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    seed: int,
    bootstrap: int,
) -> dict:
    counts = {str(label): int((labels == label).sum()) for label in sorted(set(labels.tolist()))}
    if len(counts) < 2:
        return {"status": "insufficient_classes", "class_counts_total": counts}

    min_count = min(counts.values())
    n_splits = min(5, min_count)
    if n_splits < 2:
        return {"status": "insufficient_folds", "class_counts_total": counts}

    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds: list[dict[str, float | int | None]] = []
    for fold_idx, (train_idx, test_idx) in enumerate(splitter.split(features, labels), start=1):
        clf = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
                solver="lbfgs",
                random_state=seed + fold_idx,
            ),
        )
        clf.fit(features[train_idx], labels[train_idx])
        pred = clf.predict(features[test_idx])
        proba = clf.predict_proba(features[test_idx])[:, 1]
        try:
            auc = float(roc_auc_score(labels[test_idx], proba))
        except ValueError:
            auc = None
        folds.append(
            {
                "fold": fold_idx,
                "n_train": int(len(train_idx)),
                "n_test": int(len(test_idx)),
                "auroc": auc,
                "accuracy": float(accuracy_score(labels[test_idx], pred)),
                "macro_f1": float(f1_score(labels[test_idx], pred, average="macro", zero_division=0)),
            }
        )

    return {
        "status": "ok",
        "n_splits": n_splits,
        "class_names": ["no_lesion", "lesion"],
        "class_counts_total": {"no_lesion": counts.get("0", 0), "lesion": counts.get("1", 0)},
        "metrics": {
            "auroc": _ci_from_values([fold["auroc"] for fold in folds], bootstrap=bootstrap, seed=seed),
            "accuracy": _ci_from_values([float(fold["accuracy"]) for fold in folds], bootstrap=bootstrap, seed=seed + 1),
            "macro_f1": _ci_from_values([float(fold["macro_f1"]) for fold in folds], bootstrap=bootstrap, seed=seed + 2),
        },
        "folds": folds,
    }


def _run_probe(
    extractor: BlockFeatureExtractor,
    items: list[ImageItem],
    *,
    name: str,
    batch_size: int,
    seed: int,
    bootstrap: int,
    normal_source: str,
) -> dict:
    features = extractor.extract(items, batch_size=batch_size)
    labels = np.asarray([int(item.label or 0) for item in items], dtype=np.int64)
    result = _cv_probe(features, labels, seed=seed, bootstrap=bootstrap)
    auroc = result.get("metrics", {}).get("auroc", {}).get("mean") if result.get("status") == "ok" else None
    result.update(
        {
            "probe": name,
            "normal_source": normal_source,
            "n_items": len(items),
            "viability": (
                "full" if auroc is not None and auroc >= 0.70
                else "low_confidence" if auroc is not None and auroc >= 0.50
                else "closed" if auroc is not None
                else "unknown"
            ),
        }
    )
    return result


def _concept_row_from_masks(
    *,
    image_id: str,
    image_path: str,
    domain: str,
    split: str,
    label: str,
    masks: dict[str, np.ndarray],
    source: str,
) -> dict[str, str | int | float]:
    concepts = {code: int(masks.get(code, np.zeros((1, 1), dtype=np.uint8)).sum() > 0) for code in LESION_CODES}
    return {
        "image_id": image_id,
        "image_path": image_path,
        "domain": domain,
        "split": split,
        "label": label,
        **concepts,
        "lesion_present": int(any(concepts.values())),
        "mask_valid": 1,
        "weak_label_valid": 1,
        "concept_source": source,
        "concept_confidence": 1.0,
    }


def _write_concept_labels(
    project_root: Path,
    manifest_rows: list[dict[str, str]],
    output_path: Path,
) -> dict:
    rows: list[dict[str, str | int | float]] = []
    seen: set[tuple[str, str, str]] = set()

    def _append(row: dict[str, str | int | float]) -> None:
        key = (str(row["domain"]), str(row["image_id"]), str(row["image_path"]))
        if key in seen:
            return
        seen.add(key)
        rows.append(row)

    # Pixel-level IDRiD segmentation labels.
    for split, split_dir in (("train", "a. Training Set"), ("test", "b. Testing Set")):
        image_dir = project_root / "data/raw/IDRiD/A. Segmentation/1. Original Images" / split_dir
        mask_dir = project_root / "data/raw/IDRiD/A. Segmentation/2. All Segmentation Groundtruths" / split_dir
        for image_path in sorted(image_dir.glob("*.jpg")) + sorted(image_dir.glob("*.png")):
            masks = load_lesion_masks(mask_dir, image_path.stem)
            _append(
                _concept_row_from_masks(
                    image_id=image_path.stem,
                    image_path=_relative_to_data_root(project_root, image_path),
                    domain="IDRiD_SEG",
                    split=split,
                    label="",
                    masks=masks,
                    source="idrid_segmentation_pixel_mask",
                )
            )

    # Pixel-level MAPLES labels.
    maples_root = project_root / "data/raw/MAPLES-DR/AdditionalData"
    record_path = maples_root / "dataset_record.yaml"
    annotations_dir = maples_root / "annotations"
    messidor_dir = project_root / "data/raw/messidor/images"
    with record_path.open("r", encoding="utf-8") as handle:
        record = yaml.safe_load(handle)
    for split in ("train", "test"):
        for stem in record[split]:
            image_path = next((p for p in _image_candidates(messidor_dir, stem) if p.exists()), None)
            if image_path is None:
                continue
            masks = load_maples_masks(annotations_dir, stem)
            _append(
                _concept_row_from_masks(
                    image_id=stem,
                    image_path=_relative_to_data_root(project_root, image_path),
                    domain="MAPLES",
                    split=split,
                    label="",
                    masks=masks,
                    source="maples_pixel_mask",
                )
            )

    # Weak normal labels from classification manifests. Abnormal unlabeled rows
    # are recorded as unknown so later runners can decide whether to use them.
    for row in manifest_rows:
        label = str(row.get("label", "")).strip()
        image_path = Path(row["_resolved_image_path"])
        if label == "0":
            concepts: dict[str, str | int | float] = dict.fromkeys(LESION_CODES, 0)
            _append(
                {
                    "image_id": str(row.get("image_id", image_path.stem)),
                    "image_path": _relative_to_data_root(project_root, image_path),
                    "domain": str(row.get("domain", "")),
                    "split": str(row.get("split", "")),
                    "label": label,
                    **concepts,
                    "lesion_present": 0,
                    "mask_valid": 0,
                    "weak_label_valid": 1,
                    "concept_source": "classification_label_zero_weak_normal",
                    "concept_confidence": 0.5,
                }
            )
        else:
            _append(
                {
                    "image_id": str(row.get("image_id", image_path.stem)),
                    "image_path": _relative_to_data_root(project_root, image_path),
                    "domain": str(row.get("domain", "")),
                    "split": str(row.get("split", "")),
                    "label": label,
                    **dict.fromkeys(LESION_CODES, ""),
                    "lesion_present": "",
                    "mask_valid": 0,
                    "weak_label_valid": 0,
                    "concept_source": "unknown_no_pixel_mask",
                    "concept_confidence": 0.0,
                }
            )

    fieldnames = [
        "image_id",
        "image_path",
        "domain",
        "split",
        "label",
        *LESION_CODES,
        "lesion_present",
        "mask_valid",
        "weak_label_valid",
        "concept_source",
        "concept_confidence",
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    by_source: dict[str, int] = {}
    for row in rows:
        by_source[str(row["concept_source"])] = by_source.get(str(row["concept_source"]), 0) + 1
    return {
        "path": str(output_path),
        "rows": len(rows),
        "by_source": by_source,
        "mask_valid_rows": sum(int(row["mask_valid"]) for row in rows),
        "weak_label_valid_rows": sum(int(row["weak_label_valid"]) for row in rows),
    }


def run_phase4f_s0(
    *,
    config_path: str,
    checkpoint: str | None,
    block_index: int,
    manifest: str,
    output: str,
    concept_output: str,
    max_per_class: int,
    batch_size: int,
    seed: int,
    bootstrap: int,
) -> dict:
    session = InferenceSession.from_config_path(config_path, checkpoint_path=checkpoint)
    extractor = BlockFeatureExtractor(session, block_index)
    manifest_path = resolve_project_path(session.project_root, manifest)
    manifest_rows = _read_manifest_rows(session.project_root, manifest_path)

    idrid_pos = _idrid_mask_items(session.project_root, "train") + _idrid_mask_items(session.project_root, "test")
    idrid_pos = [item for item in idrid_pos if item.label == 1]
    idrid_normals = _manifest_normal_items(
        session.project_root,
        manifest_rows,
        domain="IDRiD",
        split_values={"train", "val", "test"},
    )
    idrid_items = _cap_balanced_binary(idrid_pos, idrid_normals, max_per_class=max_per_class, seed=seed)

    maples_all = _maples_mask_items(session.project_root, "train") + _maples_mask_items(session.project_root, "test")
    maples_pos = [item for item in maples_all if item.label == 1]
    maples_normals = [item for item in maples_all if item.label == 0]
    target_normal_count = min(max_per_class, len(maples_pos))
    fallback_need = max(0, target_normal_count - len(maples_normals))
    messidor_normals = _manifest_normal_items(
        session.project_root,
        manifest_rows,
        domain="Messidor",
        split_values={"train", "val", "test"},
    )
    fallback_normals = _select_color_matched_normals(maples_pos, messidor_normals, need=fallback_need)
    maples_items = _cap_balanced_binary(
        maples_pos,
        maples_normals + fallback_normals,
        max_per_class=max_per_class,
        seed=seed,
    )

    pooled_items = list(idrid_items) + list(maples_items)
    random.Random(seed).shuffle(pooled_items)

    d12_a = _run_probe(
        extractor,
        idrid_items,
        name="D12-A_IDRiD",
        batch_size=batch_size,
        seed=seed,
        bootstrap=bootstrap,
        normal_source="IDRiD grade-0 rows from manifest",
    )
    d12_b = _run_probe(
        extractor,
        maples_items,
        name="D12-B_MAPLES",
        batch_size=batch_size,
        seed=seed,
        bootstrap=bootstrap,
        normal_source=f"MAPLES R0/no-lesion rows ({len(maples_normals)}) + Messidor grade-0 color-matched fallback ({len(fallback_normals)})",
    )
    d12_u = _run_probe(
        extractor,
        pooled_items,
        name="D12-U_IDRiD_MAPLES_POOLED",
        batch_size=batch_size,
        seed=seed,
        bootstrap=bootstrap,
        normal_source="pooled D12-A and D12-B normals",
    )

    concept_summary = _write_concept_labels(
        session.project_root,
        manifest_rows,
        resolve_project_path(session.project_root, concept_output),
    )

    d12_b_auroc = d12_b.get("metrics", {}).get("auroc", {}).get("mean")
    result = {
        "phase": "4-F S0",
        "created_by": "drscreen.cli.diagnose_v31_lesion_probe",
        "version": str(session.config.get("project", {}).get("version")),
        "config_path": str(session.config_path),
        "checkpoint_path": str(session.checkpoint_path),
        "manifest_path": str(manifest_path),
        "block_index": block_index,
        "seed": seed,
        "bootstrap_samples": bootstrap,
        "inputs": {
            "idrid_positive_mask_rows": len(idrid_pos),
            "idrid_normal_candidates": len(idrid_normals),
            "maples_positive_mask_rows": len(maples_pos),
            "maples_native_no_lesion_rows": len(maples_normals),
            "maples_normal_fallback_rows": len(fallback_normals),
        },
        "D12_A_IDRiD": d12_a,
        "D12_B_MAPLES": d12_b,
        "D12_U_POOLED": d12_u,
        "G1_viability_gate": {
            "basis": "D12-B MAPLES AUROC mean",
            "auroc": d12_b_auroc,
            "decision": (
                "G1_FULL" if d12_b_auroc is not None and d12_b_auroc >= 0.70
                else "G1_LOW_CONFIDENCE" if d12_b_auroc is not None and d12_b_auroc >= 0.50
                else "G1_CLOSED" if d12_b_auroc is not None
                else "G1_UNKNOWN"
            ),
        },
        "concept_labels": concept_summary,
    }

    output_path = resolve_project_path(session.project_root, output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Saved: {output_path}")
    print(f"Saved: {concept_summary['path']}")
    print(json.dumps(result["G1_viability_gate"], indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Phase 4-F S0 lesion-presence probe.")
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--checkpoint")
    parser.add_argument("--block-index", type=int, default=4)
    parser.add_argument("--manifest", default="data/processed/manifest_with_maples_r1plus.csv")
    parser.add_argument("--output", default=".omc/research/phase4f_v3_d12_v31_probe.json")
    parser.add_argument("--concept-output", default="data/processed/lesion_concept_labels.csv")
    parser.add_argument("--max-per-class", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bootstrap", type=int, default=2000)
    args = parser.parse_args()
    run_phase4f_s0(
        config_path=args.config,
        checkpoint=args.checkpoint,
        block_index=args.block_index,
        manifest=args.manifest,
        output=args.output,
        concept_output=args.concept_output,
        max_per_class=args.max_per_class,
        batch_size=args.batch_size,
        seed=args.seed,
        bootstrap=args.bootstrap,
    )


if __name__ == "__main__":
    main()
