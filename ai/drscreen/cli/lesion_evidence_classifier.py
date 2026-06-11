"""Phase 4-G G-4: calibrated classifier over standalone lesion evidence.

This diagnostic keeps the v31 deployment classifier untouched. It asks whether
the current best standalone segmenter (v8b) produces lesion evidence features
that are sufficient for binary DR classification.

(한글 요약) v8b 병변맵에서 뽑은 스칼라 특징'만'으로 이진 DR 분류가 가능한지 보는 진단이다.
v31 분류기는 건드리지 않으며, 결과적으로 v8b 특징 단독은 v31에 못 미쳐 배포로 승격되지 않았다.
"""
from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    roc_auc_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

from drscreen.data.transforms import build_eval_transform, preprocess_kwargs_from_config
from drscreen.infer.late_fusion_features import (
    base_lesion_feature_names,
    extended_lesion_feature_names,
    extract_extended_lesion_feature_values,
)
from drscreen.models.profiles import get_model_profile
from drscreen.models.seg_evidence import LesionSegEvidence
from drscreen.settings import (
    get_run_evaluation_dir,
    load_app_config,
    resolve_project_path,
)

LESION_CODES = ("MA", "HE", "EX", "SE")


@dataclass(frozen=True)
class EvidenceItem:
    image_id: str
    image_path: Path
    label: int
    split: str
    domain: str


class EvidenceDataset(Dataset):
    def __init__(self, items: list[EvidenceItem], transform) -> None:
        self.items = items
        self.transform = transform

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict:
        item = self.items[index]
        with Image.open(item.image_path) as image:
            tensor = self.transform(image.convert("RGB"))
        return {
            "image": tensor,
            "label": item.label,
            "image_id": item.image_id,
            "split": item.split,
            "domain": item.domain,
        }


def _read_items(config: dict, project_root: Path) -> list[EvidenceItem]:
    data_cfg = config["data"]
    manifest_path = resolve_project_path(project_root, data_cfg["manifest_path"])
    image_root = resolve_project_path(project_root, data_cfg.get("image_root", "data/raw"))
    items: list[EvidenceItem] = []
    with manifest_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            label_raw = str(row.get("label", "")).strip()
            if label_raw == "":
                continue
            image_rel = str(row.get("image_path", "")).strip()
            image_path = image_root / image_rel
            if not image_path.exists():
                continue
            try:
                label = int(float(label_raw))
            except ValueError:
                continue
            if label not in {0, 1}:
                label = int(label > 0)
            items.append(
                EvidenceItem(
                    image_id=str(row.get("image_id", image_path.stem)),
                    image_path=image_path,
                    label=label,
                    split=str(row.get("split", "")),
                    domain=str(row.get("domain", "")),
                )
            )
    if not items:
        raise ValueError(f"No labeled images found in {manifest_path}")
    return items


def _build_transform(seg_config: dict, project_root: Path):
    data_cfg = seg_config["data"]
    model_cfg = seg_config.get("model", {})
    profile = get_model_profile(str(model_cfg.get("encoder", "resnet50")))
    image_size = int(data_cfg.get("image_size", profile.crop_size))
    resize_size = int(data_cfg.get("resize_size", image_size))
    use_preprocessing = bool(data_cfg.get("use_preprocessing", False))
    _ = project_root
    return build_eval_transform(
        crop_size=image_size,
        resize_size=resize_size,
        interpolation=profile.interpolation,
        mean=profile.mean,
        std=profile.std,
        use_preprocessing=use_preprocessing,
        preprocess_kwargs=preprocess_kwargs_from_config(data_cfg),
    )


def _load_segmenter(seg_config: dict, project_root: Path, checkpoint_path: Path) -> torch.nn.Module:
    device_name = str(seg_config.get("train", {}).get("device", "cuda"))
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        device = torch.device("cpu")
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    ckpt_cfg = payload.get("config", seg_config)
    model_cfg = ckpt_cfg.get("model", seg_config.get("model", {}))
    model = LesionSegEvidence(
        encoder=str(model_cfg.get("encoder", "resnet50")),
        out_channels=int(model_cfg.get("out_channels", 4)),
        pretrained=False,
        decoder_channels=tuple(model_cfg.get("decoder_channels", [256, 128, 64, 32])),
    )
    model.load_state_dict(payload["model_state_dict"])
    model.to(device)
    model.eval()
    return model


def _feature_names(thresholds: list[float], topk_fracs: list[float]) -> list[str]:
    return [
        *base_lesion_feature_names(area_thresholds=thresholds, topk_fracs=topk_fracs),
        *extended_lesion_feature_names(),
    ]


def _extract_features(
    model: torch.nn.Module,
    items: list[EvidenceItem],
    *,
    transform,
    batch_size: int,
    num_workers: int,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, str | int]]]:
    thresholds = [0.05, 0.1, 0.2, 0.3, 0.5]
    topk_fracs = [0.001, 0.01, 0.05]
    dataset = EvidenceDataset(items, transform)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=next(model.parameters()).device.type == "cuda",
    )
    features: list[np.ndarray] = []
    labels: list[int] = []
    rows: list[dict[str, str | int]] = []
    device = next(model.parameters()).device

    with torch.no_grad():
        for batch in loader:
            image = batch["image"].to(device, non_blocking=True)
            probs = torch.sigmoid(model(image)).float()
            union = probs.amax(dim=1, keepdim=True)
            maps = torch.cat([probs, union], dim=1)
            flat = maps.flatten(2)
            parts = [
                flat.mean(dim=2),
                flat.amax(dim=2),
                flat.std(dim=2),
            ]
            for threshold in thresholds:
                parts.append((flat >= threshold).float().mean(dim=2))
            n_pixels = flat.shape[-1]
            for frac in topk_fracs:
                k = max(1, int(round(n_pixels * frac)))
                parts.append(torch.topk(flat, k=k, dim=2).values.mean(dim=2))
            batch_features = torch.cat(parts, dim=1).detach().cpu().numpy().astype(np.float32)
            extended_schema = extended_lesion_feature_names()
            extended_features = np.asarray(
                [
                    extract_extended_lesion_feature_values(probs[index], extended_schema)
                    for index in range(probs.shape[0])
                ],
                dtype=np.float32,
            )
            batch_features = np.concatenate([batch_features, extended_features], axis=1)
            features.append(batch_features)
            batch_labels = [int(v) for v in batch["label"].tolist()]
            labels.extend(batch_labels)
            rows.extend(
                {
                    "image_id": str(image_id),
                    "split": str(split),
                    "domain": str(domain),
                    "label": int(label),
                }
                for image_id, split, domain, label in zip(
                    batch["image_id"],
                    batch["split"],
                    batch["domain"],
                    batch_labels,
                    strict=True,
                )
            )

    return np.concatenate(features, axis=0), np.asarray(labels, dtype=np.int64), rows


def _safe_auc(labels: np.ndarray, scores: np.ndarray) -> float | None:
    if len(np.unique(labels)) < 2:
        return None
    return float(roc_auc_score(labels, scores))


def _classification_report(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict:
    pred = (scores >= threshold).astype(np.int64)
    if len(np.unique(labels)) == 2:
        tn, fp, fn, tp = confusion_matrix(labels, pred, labels=[0, 1]).ravel()
    else:
        tn = fp = fn = tp = 0
    sensitivity = tp / (tp + fn) if (tp + fn) else None
    specificity = tn / (tn + fp) if (tn + fp) else None
    return {
        "n": int(labels.size),
        "class_counts": {
            "normal": int((labels == 0).sum()),
            "abnormal": int((labels == 1).sum()),
        },
        "auroc": _safe_auc(labels, scores),
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(labels, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, pred)),
        "sensitivity": float(sensitivity) if sensitivity is not None else None,
        "specificity": float(specificity) if specificity is not None else None,
        "precision": float(precision_score(labels, pred, zero_division=0)),
        "f1": float(f1_score(labels, pred, zero_division=0)),
    }


def _choose_threshold(labels: np.ndarray, scores: np.ndarray, thresholds: np.ndarray) -> dict:
    candidates = []
    for threshold in thresholds:
        pred = (scores >= threshold).astype(np.int64)
        candidates.append(
            {
                "threshold": float(threshold),
                "balanced_accuracy": float(balanced_accuracy_score(labels, pred)),
                "accuracy": float(accuracy_score(labels, pred)),
                "f1": float(f1_score(labels, pred, zero_division=0)),
            }
        )
    best = max(candidates, key=lambda row: (row["balanced_accuracy"], row["f1"]))
    return {"best": best, "candidates": candidates}


def run(config_path: str) -> dict:
    config_path_obj = Path(config_path).resolve()
    project_root = config_path_obj.parents[1]
    config = load_app_config(config_path_obj)
    version = str(config["project"]["version"])

    seg_config_path = resolve_project_path(project_root, config["segmentation"]["config_path"])
    seg_config = load_app_config(seg_config_path)
    checkpoint_path = resolve_project_path(project_root, config["segmentation"]["checkpoint_path"])
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Segmenter checkpoint not found: {checkpoint_path}")

    items = _read_items(config, project_root)
    split_counts: dict[str, dict[str, int]] = {}
    for item in items:
        split_counts.setdefault(item.split, {"total": 0, "normal": 0, "abnormal": 0})
        split_counts[item.split]["total"] += 1
        split_counts[item.split]["abnormal" if item.label else "normal"] += 1

    model = _load_segmenter(seg_config, project_root, checkpoint_path)
    transform = _build_transform(seg_config, project_root)
    data_cfg = config["data"]
    features, labels, rows = _extract_features(
        model,
        items,
        transform=transform,
        batch_size=int(data_cfg.get("batch_size", 16)),
        num_workers=int(data_cfg.get("num_workers", 0)),
    )

    train_split = str(data_cfg.get("train_split", "train"))
    threshold_split = str(data_cfg.get("threshold_split", "val"))
    row_splits = np.asarray([str(row["split"]) for row in rows])
    row_domains = np.asarray([str(row["domain"]) for row in rows])
    train_mask = row_splits == train_split
    threshold_mask = row_splits == threshold_split
    train_domains = data_cfg.get("train_domains")
    if train_domains:
        train_domains_set = {str(domain) for domain in train_domains}
        train_mask &= np.isin(row_domains, list(train_domains_set))
    if not train_mask.any():
        raise ValueError(f"No rows for train_split={train_split}")
    if not threshold_mask.any():
        raise ValueError(f"No rows for threshold_split={threshold_split}")

    clf_cfg = config.get("classifier", {})
    threshold_cfg = clf_cfg.get("thresholds", {})
    thresholds = np.linspace(
        float(threshold_cfg.get("min", 0.01)),
        float(threshold_cfg.get("max", 0.99)),
        int(threshold_cfg.get("steps", 99)),
    )
    c_values = clf_cfg.get("c_values")
    if c_values:
        candidate_c_values = [float(value) for value in c_values]
    else:
        candidate_c_values = [float(clf_cfg.get("c_value", 1.0))]

    classifier_candidates: list[dict] = []
    selected_clf = None
    selected_scores = None
    threshold_selection = None
    best_rank: tuple[float, float] | None = None
    for c_value in candidate_c_values:
        clf = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=c_value,
                max_iter=5000,
                class_weight=str(clf_cfg.get("class_weight", "balanced")),
                solver="lbfgs",
                random_state=int(clf_cfg.get("seed", 43)),
            ),
        )
        clf.fit(features[train_mask], labels[train_mask])
        scores = clf.predict_proba(features)[:, 1]
        selection = _choose_threshold(
            labels[threshold_mask],
            scores[threshold_mask],
            thresholds,
        )
        val_report = _classification_report(
            labels[threshold_mask],
            scores[threshold_mask],
            float(selection["best"]["threshold"]),
        )
        candidate = {
            "c_value": c_value,
            "threshold_selection": selection["best"],
            "threshold_split_metrics": val_report,
        }
        classifier_candidates.append(candidate)
        rank = (
            selection["best"]["balanced_accuracy"],
            val_report.get("auroc") or 0.0,
        )
        if best_rank is None or rank > best_rank:
            selected_clf = clf
            selected_scores = scores
            threshold_selection = selection
            best_rank = rank

    assert selected_clf is not None
    assert selected_scores is not None
    assert threshold_selection is not None
    clf = selected_clf
    scores = selected_scores
    selected_threshold = float(threshold_selection["best"]["threshold"])

    eval_splits = [str(v) for v in data_cfg.get("eval_splits", [train_split, threshold_split])]
    by_split: dict[str, dict] = {}
    by_domain: dict[str, dict] = {}
    for split in eval_splits:
        mask = row_splits == split
        if mask.any():
            by_split[split] = {
                "selected_threshold": _classification_report(labels[mask], scores[mask], selected_threshold),
                "threshold_0_5": _classification_report(labels[mask], scores[mask], 0.5),
            }
    for split in eval_splits:
        split_mask = row_splits == split
        for domain in sorted(set(row_domains[split_mask].tolist())):
            mask = split_mask & (row_domains == domain)
            if mask.any():
                by_domain[f"{split}:{domain}"] = _classification_report(
                    labels[mask],
                    scores[mask],
                    selected_threshold,
                )

    scaler = clf.named_steps["standardscaler"]
    logreg = clf.named_steps["logisticregression"]
    result = {
        "version": version,
        "segmentation_version": str(seg_config["project"]["version"]),
        "segmentation_checkpoint_path": str(checkpoint_path),
        "manifest_path": str(resolve_project_path(project_root, data_cfg["manifest_path"])),
        "n_images": int(len(items)),
        "split_counts": split_counts,
        "feature_names": _feature_names([0.05, 0.1, 0.2, 0.3, 0.5], [0.001, 0.01, 0.05]),
        "train_split": train_split,
        "train_domains": sorted({str(domain) for domain in row_domains[train_mask].tolist()}),
        "n_train_fit": int(train_mask.sum()),
        "threshold_split": threshold_split,
        "threshold_selection": threshold_selection,
        "classifier_candidates": classifier_candidates,
        "metrics_by_split": by_split,
        "metrics_by_domain": by_domain,
        "classifier": {
            "type": "StandardScaler+LogisticRegression",
            "coef": logreg.coef_.astype(float).tolist(),
            "intercept": logreg.intercept_.astype(float).tolist(),
            "scaler_mean": scaler.mean_.astype(float).tolist(),
            "scaler_scale": scaler.scale_.astype(float).tolist(),
        },
        "decision_note": (
            "Diagnostic only. This experiment does not change v31 deployment, "
            "configs/base.yaml, artifacts/checkpoints/best.pt, backend, or frontend."
        ),
    }

    output_dir = get_run_evaluation_dir(project_root, version)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{version}_metrics.json"
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Saved: {output_path}")
    print(json.dumps(result["metrics_by_split"], indent=2))
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fit/evaluate a calibrated classifier over v8b lesion evidence features."
    )
    parser.add_argument("--config", required=True, help="Path to v8b evidence classifier config.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
