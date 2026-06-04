"""D5-D7 shortcut audit for the active DR classifier.

(한글 요약) 활성 분류기가 병변 근거보다 '도메인/스타일 단서'에 더 의존하는지(shortcut)를
감사한다. 이 결과가 제품 문구를 '병변 때문에 분류'가 아니라 '분류 + 별도 탐지된 병변 후보'로
제한하는 근거가 된다(AI_HANDOFF Phase 4-E).

The audit tests whether the active classifier representation is more strongly
aligned with domain/style cues than with lesion evidence:

- D5: linear domain probe over block features.
- D6: linear lesion-presence probe over MAPLES lesion masks.
- D7: counterfactual style swap on lesion vs non-lesion pixels.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
from dataclasses import dataclass
from pathlib import Path
from collections.abc import Iterable

import numpy as np
import torch
import yaml
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from drscreen.infer.service import InferenceSession
from drscreen.settings import get_run_evaluation_dir, resolve_project_path
from drscreen.xai.iou import load_lesion_masks, load_maples_masks, union_mask


@dataclass(frozen=True)
class ImageItem:
    path: Path
    domain: str
    split: str
    image_id: str
    label: int | None = None
    masks: dict[str, np.ndarray] | None = None


def _image_candidates(base: Path, stem: str) -> list[Path]:
    return [base / f"{stem}{ext}" for ext in (".jpg", ".jpeg", ".png", ".tif", ".tiff")]


def _preprocess_image(session: InferenceSession, image: Image.Image) -> Image.Image:
    image = image.convert("RGB")
    if session.preprocessor is not None:
        image = session.preprocessor(image)
    return image


def _pick_manifest(project_root: Path, explicit: str | None = None) -> Path:
    if explicit:
        return resolve_project_path(project_root, explicit)
    candidates = [
        project_root / "data/processed/manifest_with_maples_r1plus_preprocessed.csv",
        project_root / "data/processed/manifest_with_maples_preprocessed.csv",
        project_root / "data/processed/manifest_preprocessed.csv",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError("No processed manifest found.")


def _read_manifest_items(
    project_root: Path,
    manifest_path: Path,
    *,
    max_per_domain: int,
    seed: int,
) -> list[ImageItem]:
    data_root = project_root / "data/raw"
    by_domain: dict[str, list[ImageItem]] = {"DDR": [], "IDRiD": [], "MAPLES": []}
    with manifest_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            domain = str(row.get("domain", "")).strip()
            if domain not in by_domain:
                continue
            split = str(row.get("split", "")).strip()
            if domain == "DDR" and split != "external_test":
                continue
            if domain == "MAPLES" and split not in {"train", "test"}:
                continue
            image_path = data_root / str(row.get("image_path", "")).strip()
            if not image_path.exists():
                continue
            label_raw = row.get("label", "")
            label = int(label_raw) if str(label_raw).strip() != "" else None
            by_domain[domain].append(
                ImageItem(
                    path=image_path,
                    domain=domain,
                    split=split,
                    image_id=str(row.get("image_id", image_path.stem)),
                    label=label,
                )
            )

    rng = random.Random(seed)
    selected: list[ImageItem] = []
    for domain, rows in by_domain.items():
        rng.shuffle(rows)
        selected.extend(rows[:max_per_domain])
    if len({item.domain for item in selected}) < 2:
        raise ValueError(f"Need at least two domains for D5 probe; got {len(selected)} rows.")
    return selected


def _idrid_mask_items(project_root: Path, split: str) -> list[ImageItem]:
    split_dir = {"train": "a. Training Set", "test": "b. Testing Set"}[split]
    image_dir = project_root / "data/raw/IDRiD/A. Segmentation/1. Original Images" / split_dir
    mask_dir = project_root / "data/raw/IDRiD/A. Segmentation/2. All Segmentation Groundtruths" / split_dir
    items: list[ImageItem] = []
    for image_path in sorted(image_dir.glob("*.jpg")) + sorted(image_dir.glob("*.png")):
        masks = load_lesion_masks(mask_dir, image_path.stem)
        if masks:
            lesion = union_mask(masks)
            items.append(
                ImageItem(
                    path=image_path,
                    domain="IDRiD",
                    split=split,
                    image_id=image_path.stem,
                    label=int(lesion is not None and lesion.sum() > 0),
                    masks=masks,
                )
            )
    return items


def _maples_mask_items(project_root: Path, split: str) -> list[ImageItem]:
    maples_root = project_root / "data/raw/MAPLES-DR/AdditionalData"
    record_path = maples_root / "dataset_record.yaml"
    annotations_dir = maples_root / "annotations"
    messidor_dir = project_root / "data/raw/messidor/images"
    with record_path.open("r", encoding="utf-8") as handle:
        record = yaml.safe_load(handle)

    items: list[ImageItem] = []
    for stem in record[split]:
        image_path = next((p for p in _image_candidates(messidor_dir, stem) if p.exists()), None)
        if image_path is None:
            continue
        masks = load_maples_masks(annotations_dir, stem)
        if not masks:
            continue
        lesion = union_mask(masks)
        items.append(
            ImageItem(
                path=image_path,
                domain="MAPLES",
                split=split,
                image_id=stem,
                label=int(lesion is not None and lesion.sum() > 0),
                masks=masks,
            )
        )
    return items


def _balanced_cap(items: list[ImageItem], *, max_per_class: int, seed: int) -> list[ImageItem]:
    rng = random.Random(seed)
    by_label: dict[int, list[ImageItem]] = {}
    for item in items:
        if item.label is None:
            continue
        by_label.setdefault(int(item.label), []).append(item)
    selected: list[ImageItem] = []
    for rows in by_label.values():
        rng.shuffle(rows)
        selected.extend(rows[:max_per_class])
    rng.shuffle(selected)
    return selected


class BlockFeatureExtractor:
    def __init__(self, session: InferenceSession, block_index: int) -> None:
        self.session = session
        blocks = getattr(session.model, "blocks", getattr(session.model, "features", None))
        if blocks is None:
            raise ValueError("Model exposes neither .blocks nor .features.")
        self.layer = blocks[block_index]

    def extract(self, items: list[ImageItem], *, batch_size: int) -> np.ndarray:
        features: list[np.ndarray] = []
        device = self.session.device
        model = self.session.model
        model.eval()

        for start in range(0, len(items), batch_size):
            batch_items = items[start : start + batch_size]
            tensors = []
            for item in batch_items:
                with Image.open(item.path) as image:
                    image = _preprocess_image(self.session, image)
                    tensors.append(self.session.eval_transform(image))
            x = torch.stack(tensors, dim=0).to(device)
            captured: list[torch.Tensor] = []

            def _hook(_module, _inputs, output):
                captured.append(output.detach())

            handle = self.layer.register_forward_hook(_hook)
            try:
                with torch.no_grad():
                    model(x)
            finally:
                handle.remove()
            if not captured:
                raise RuntimeError("Feature hook did not capture an activation.")
            feat = captured[-1]
            pooled = torch.nn.functional.adaptive_avg_pool2d(feat, 1).flatten(1)
            features.append(pooled.cpu().numpy().astype(np.float32))
        return np.concatenate(features, axis=0)


def _safe_roc_auc(y_true: np.ndarray, scores: np.ndarray, *, multi_class: str | None = None) -> float | None:
    try:
        if multi_class:
            return float(roc_auc_score(y_true, scores, multi_class=multi_class, average="macro"))
        if len(np.unique(y_true)) < 2:
            return None
        return float(roc_auc_score(y_true, scores))
    except ValueError:
        return None


def _fit_probe(
    x: np.ndarray,
    y: np.ndarray,
    *,
    seed: int,
    class_names: list[str],
) -> dict:
    stratify = y if min(np.bincount(y)) >= 2 else None
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.35,
        random_state=seed,
        stratify=stratify,
    )
    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            solver="lbfgs",
            random_state=seed,
        ),
    )
    clf.fit(x_train, y_train)
    pred = clf.predict(x_test)
    proba = clf.predict_proba(x_test)
    if len(class_names) == 2:
        auc = _safe_roc_auc(y_test, proba[:, 1])
    else:
        auc = _safe_roc_auc(y_test, proba, multi_class="ovr")
    return {
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
        "class_names": class_names,
        "class_counts_total": {
            class_names[i]: int((y == i).sum()) for i in range(len(class_names))
        },
        "accuracy": float(accuracy_score(y_test, pred)),
        "macro_f1": float(f1_score(y_test, pred, average="macro", zero_division=0)),
        "macro_ovr_auroc": auc,
    }


def _run_d5_domain_probe(
    session: InferenceSession,
    extractor: BlockFeatureExtractor,
    *,
    manifest_path: Path,
    max_per_domain: int,
    batch_size: int,
    seed: int,
) -> dict:
    items = _read_manifest_items(
        session.project_root,
        manifest_path,
        max_per_domain=max_per_domain,
        seed=seed,
    )
    domain_names = sorted({item.domain for item in items})
    domain_to_idx = {domain: idx for idx, domain in enumerate(domain_names)}
    features = extractor.extract(items, batch_size=batch_size)
    y = np.array([domain_to_idx[item.domain] for item in items], dtype=np.int64)
    result = _fit_probe(features, y, seed=seed, class_names=domain_names)
    result.update(
        {
            "probe": "D5_domain_probe",
            "manifest_path": str(manifest_path),
            "max_per_domain": max_per_domain,
            "shortcut_signal": bool(
                (result.get("macro_ovr_auroc") or 0.0) >= 0.90
                or result["macro_f1"] >= 0.85
            ),
        }
    )
    return result


def _run_d6_lesion_presence_probe(
    session: InferenceSession,
    extractor: BlockFeatureExtractor,
    *,
    max_per_class: int,
    batch_size: int,
    seed: int,
) -> dict:
    maples = _maples_mask_items(session.project_root, "train") + _maples_mask_items(
        session.project_root, "test"
    )
    idrid = _idrid_mask_items(session.project_root, "train") + _idrid_mask_items(
        session.project_root, "test"
    )

    def _probe(items: list[ImageItem], name: str) -> dict:
        capped = _balanced_cap(items, max_per_class=max_per_class, seed=seed)
        labels = sorted({int(item.label) for item in capped if item.label is not None})
        if len(labels) < 2:
            return {
                "probe": name,
                "status": "insufficient_classes",
                "class_counts_total": {
                    str(label): sum(1 for item in items if item.label == label) for label in labels
                },
            }
        features = extractor.extract(capped, batch_size=batch_size)
        y = np.array([int(item.label) for item in capped], dtype=np.int64)
        result = _fit_probe(features, y, seed=seed, class_names=["no_lesion", "lesion"])
        result.update({"probe": name, "max_per_class": max_per_class})
        result["weak_lesion_signal"] = bool((result.get("macro_ovr_auroc") or 0.0) < 0.70)
        return result

    pooled = maples + idrid
    return {
        "probe": "D6_lesion_presence_probe",
        "primary_maples_only": _probe(maples, "D6_maples_only"),
        "secondary_pooled_idrid_maples": {
            **_probe(pooled, "D6_pooled_idrid_maples"),
            "domain_confounding_warning": True,
        },
        "source_counts": {
            "maples": {
                "total": len(maples),
                "positive": sum(1 for item in maples if item.label == 1),
                "negative": sum(1 for item in maples if item.label == 0),
            },
            "idrid": {
                "total": len(idrid),
                "positive": sum(1 for item in idrid if item.label == 1),
                "negative": sum(1 for item in idrid if item.label == 0),
            },
        },
    }


def _dataset_stats(items: list[ImageItem], sample_size: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = random.Random(seed)
    rows = list(items)
    rng.shuffle(rows)
    pixels: list[np.ndarray] = []
    for item in rows[:sample_size]:
        with Image.open(item.path) as image:
            arr = np.asarray(image.convert("RGB").resize((256, 256), Image.BILINEAR), dtype=np.float32) / 255.0
        pixels.append(arr.reshape(-1, 3))
    stacked = np.concatenate(pixels, axis=0)
    return stacked.mean(axis=0), stacked.std(axis=0).clip(min=1e-4)


def _style_transfer_region(
    image: np.ndarray,
    region: np.ndarray,
    *,
    source_mean: np.ndarray,
    source_std: np.ndarray,
    target_mean: np.ndarray,
    target_std: np.ndarray,
) -> np.ndarray:
    out = image.copy()
    if not np.any(region):
        return out
    selected = out[region]
    selected = (selected - source_mean) / source_std * target_std + target_mean
    out[region] = np.clip(selected, 0.0, 1.0)
    return out


def _predict_prob(session: InferenceSession, arrays: Iterable[np.ndarray]) -> list[float]:
    tensors = []
    for arr in arrays:
        image = Image.fromarray(np.uint8(np.clip(arr, 0.0, 1.0) * 255.0))
        image = _preprocess_image(session, image)
        tensors.append(session.eval_transform(image))
    x = torch.stack(tensors, dim=0).to(session.device)
    with torch.no_grad():
        logits = session.model(x)
        probs = torch.sigmoid(logits).flatten()
    return [float(v) for v in probs.detach().cpu().numpy()]


def _run_d7_counterfactual_swap(
    session: InferenceSession,
    *,
    max_images: int,
    seed: int,
) -> dict:
    rng = np.random.default_rng(seed)
    idrid_items = _idrid_mask_items(session.project_root, "test")
    maples_items = _maples_mask_items(session.project_root, "test")
    idrid_stats = _dataset_stats(idrid_items, sample_size=min(30, len(idrid_items)), seed=seed)
    maples_stats = _dataset_stats(maples_items, sample_size=min(30, len(maples_items)), seed=seed + 1)
    source_by_domain = {"IDRiD": idrid_stats, "MAPLES": maples_stats}
    target_by_domain = {"IDRiD": maples_stats, "MAPLES": idrid_stats}

    positive_items = [item for item in idrid_items + maples_items if item.label == 1]
    random.Random(seed).shuffle(positive_items)
    positive_items = positive_items[:max_images]
    per_image = []

    for item in positive_items:
        with Image.open(item.path) as image:
            pil = image.convert("RGB")
            w, h = pil.size
            image_arr = np.asarray(pil, dtype=np.float32) / 255.0
        if item.domain == "IDRiD":
            split_dir = "b. Testing Set" if item.split == "test" else "a. Training Set"
            mask_dir = (
                session.project_root
                / "data/raw/IDRiD/A. Segmentation/2. All Segmentation Groundtruths"
                / split_dir
            )
            masks = load_lesion_masks(mask_dir, item.path.stem, target_size=(w, h))
        else:
            masks = load_maples_masks(
                session.project_root / "data/raw/MAPLES-DR/AdditionalData/annotations",
                item.path.stem,
                target_size=(w, h),
            )
        lesion = union_mask(masks)
        if lesion is None or lesion.sum() == 0:
            continue
        lesion_region = lesion.astype(bool)
        nonlesion_region = ~lesion_region
        lesion_count = int(lesion_region.sum())
        nonlesion_indices = np.flatnonzero(nonlesion_region.ravel())
        if len(nonlesion_indices) < lesion_count:
            continue
        sampled = rng.choice(nonlesion_indices, size=lesion_count, replace=False)
        matched_nonlesion_region = np.zeros(nonlesion_region.size, dtype=bool)
        matched_nonlesion_region[sampled] = True
        matched_nonlesion_region = matched_nonlesion_region.reshape(nonlesion_region.shape)

        source_mean, source_std = source_by_domain[item.domain]
        target_mean, target_std = target_by_domain[item.domain]
        lesion_swap = _style_transfer_region(
            image_arr,
            lesion_region,
            source_mean=source_mean,
            source_std=source_std,
            target_mean=target_mean,
            target_std=target_std,
        )
        matched_nonlesion_swap = _style_transfer_region(
            image_arr,
            matched_nonlesion_region,
            source_mean=source_mean,
            source_std=source_std,
            target_mean=target_mean,
            target_std=target_std,
        )
        full_nonlesion_swap = _style_transfer_region(
            image_arr,
            nonlesion_region,
            source_mean=source_mean,
            source_std=source_std,
            target_mean=target_mean,
            target_std=target_std,
        )
        original, lesion_prob, matched_prob, full_nonlesion_prob = _predict_prob(
            session,
            [image_arr, lesion_swap, matched_nonlesion_swap, full_nonlesion_swap],
        )
        per_image.append(
            {
                "image_id": item.image_id,
                "domain": item.domain,
                "lesion_area_ratio": lesion_count / float(lesion_region.size),
                "prob_original": original,
                "delta_lesion": abs(lesion_prob - original),
                "delta_matched_nonlesion": abs(matched_prob - original),
                "delta_full_nonlesion": abs(full_nonlesion_prob - original),
            }
        )

    def _mean(key: str) -> float | None:
        vals = [float(row[key]) for row in per_image if key in row]
        return float(np.mean(vals)) if vals else None

    lesion_delta = _mean("delta_lesion")
    matched_delta = _mean("delta_matched_nonlesion")
    full_delta = _mean("delta_full_nonlesion")
    return {
        "probe": "D7_counterfactual_style_swap",
        "n_images": len(per_image),
        "mean_abs_delta_lesion": lesion_delta,
        "mean_abs_delta_matched_nonlesion": matched_delta,
        "mean_abs_delta_full_nonlesion": full_delta,
        "matched_nonlesion_over_lesion": (
            matched_delta / lesion_delta if lesion_delta and matched_delta is not None else None
        ),
        "full_nonlesion_over_lesion": (
            full_delta / lesion_delta if lesion_delta and full_delta is not None else None
        ),
        "shortcut_signal": bool(
            lesion_delta is not None
            and matched_delta is not None
            and matched_delta > lesion_delta * 1.2
        ),
        "per_image": per_image,
    }


def run_audit(
    *,
    config_path: str,
    checkpoint: str | None,
    block_index: int,
    manifest: str | None,
    max_per_domain: int,
    max_per_class: int,
    max_counterfactual: int,
    batch_size: int,
    seed: int,
    output: str | None,
    research_output: str | None = ".omc/research/phase4e_shortcut_audit.json",
) -> dict:
    session = InferenceSession.from_config_path(config_path, checkpoint_path=checkpoint)
    extractor = BlockFeatureExtractor(session, block_index)
    manifest_path = _pick_manifest(session.project_root, manifest)

    d5 = _run_d5_domain_probe(
        session,
        extractor,
        manifest_path=manifest_path,
        max_per_domain=max_per_domain,
        batch_size=batch_size,
        seed=seed,
    )
    d6 = _run_d6_lesion_presence_probe(
        session,
        extractor,
        max_per_class=max_per_class,
        batch_size=batch_size,
        seed=seed,
    )
    d7 = _run_d7_counterfactual_swap(
        session,
        max_images=max_counterfactual,
        seed=seed,
    )

    shortcut_signals = [
        bool(d5.get("shortcut_signal")),
        bool(d7.get("shortcut_signal")),
    ]
    maples_lesion = d6.get("primary_maples_only", {})
    weak_lesion = bool(maples_lesion.get("weak_lesion_signal"))
    if weak_lesion:
        shortcut_signals.append(True)
    result = {
        "version": str(session.config.get("project", {}).get("version")),
        "config_path": str(session.config_path),
        "checkpoint_path": str(session.checkpoint_path),
        "decision_threshold": session.decision_threshold,
        "block_index": block_index,
        "seed": seed,
        "hypothesis": "classifier_shortcut_over_lesion_evidence",
        "decision": {
            "shortcut_hypothesis_supported": sum(shortcut_signals) >= 2,
            "signals": {
                "D5_domain_probe_high_domain_separability": bool(d5.get("shortcut_signal")),
                "D6_weak_maples_lesion_presence_probe": weak_lesion,
                "D7_matched_nonlesion_swap_more_impactful": bool(d7.get("shortcut_signal")),
            },
        },
        "D5_domain_probe": d5,
        "D6_lesion_presence_probe": d6,
        "D7_counterfactual_swap": d7,
    }

    if output is None:
        eval_dir = get_run_evaluation_dir(session.project_root, str(result["version"]))
        eval_dir.mkdir(parents=True, exist_ok=True)
        output_path = eval_dir / f"shortcut_audit_{result['version']}.json"
    else:
        output_path = resolve_project_path(session.project_root, output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    research_path = None
    if research_output:
        research_path = resolve_project_path(session.project_root, research_output)
        research_path.parent.mkdir(parents=True, exist_ok=True)
        research_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print(f"Saved: {output_path}")
    if research_path is not None:
        print(f"Saved: {research_path}")
    print(json.dumps(result["decision"], indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Run D5-D7 shortcut audit on the active classifier.")
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--checkpoint")
    parser.add_argument("--block-index", type=int, default=4)
    parser.add_argument("--manifest")
    parser.add_argument("--max-per-domain", type=int, default=120)
    parser.add_argument("--max-per-class", type=int, default=120)
    parser.add_argument("--max-counterfactual", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output")
    parser.add_argument(
        "--research-output",
        default=".omc/research/phase4e_shortcut_audit.json",
        help="Optional research JSON path. Pass an empty string to skip.",
    )
    args = parser.parse_args()
    run_audit(
        config_path=args.config,
        checkpoint=args.checkpoint,
        block_index=args.block_index,
        manifest=args.manifest,
        max_per_domain=args.max_per_domain,
        max_per_class=args.max_per_class,
        max_counterfactual=args.max_counterfactual,
        batch_size=args.batch_size,
        seed=args.seed,
        output=args.output,
        research_output=args.research_output or None,
    )


if __name__ == "__main__":
    main()
