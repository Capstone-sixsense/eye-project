"""Phase 4-G: v31 classifier + v8b lesion evidence late-fusion diagnostic."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

from drscreen.cli.lesion_evidence_classifier import (
    EvidenceItem,
    _choose_threshold,
    _classification_report,
    _extract_features,
    _feature_names,
    _load_segmenter,
    _read_items,
)
from drscreen.infer.service import InferenceSession
from drscreen.settings import get_run_evaluation_dir, load_app_config, resolve_project_path


class _ImageDataset(Dataset):
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


def _extract_v31_scores(
    session: InferenceSession,
    items: list[EvidenceItem],
    *,
    batch_size: int,
    num_workers: int,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, str | int]]]:
    """Extract v31 probabilities on already-preprocessed manifest images.

    The manifest used here points to ``processed/images/...``. Therefore this
    diagnostic intentionally uses ``session.eval_transform`` directly and does
    not apply the inference-time raw-image preprocessor a second time.
    """
    dataset = _ImageDataset(items, session.eval_transform)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=session.device.type == "cuda",
    )
    scores: list[np.ndarray] = []
    labels: list[int] = []
    rows: list[dict[str, str | int]] = []
    session.model.eval()
    with torch.no_grad():
        for batch in loader:
            x = batch["image"].to(session.device, non_blocking=True)
            logits = session.model(x)
            if isinstance(logits, tuple):
                logits = logits[0]
            logits = logits.reshape(logits.shape[0], -1)
            if logits.shape[1] == 1:
                probabilities = torch.sigmoid(logits[:, 0])
            else:
                probabilities = torch.softmax(logits, dim=1)[:, 1]
            probabilities_np = probabilities.detach().cpu().numpy().astype(np.float32)
            probabilities_np = np.clip(probabilities_np, 1e-6, 1.0 - 1e-6)
            logits_np = np.log(probabilities_np / (1.0 - probabilities_np)).astype(np.float32)
            scores.append(np.stack([probabilities_np, logits_np], axis=1))
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
    return np.concatenate(scores, axis=0), np.asarray(labels, dtype=np.int64), rows


def _build_feature_matrix(
    feature_set: str,
    *,
    v31_scores: np.ndarray,
    v8b_features: np.ndarray,
) -> tuple[np.ndarray, list[str]]:
    v31_names = ["v31_probability", "v31_logit"]
    v8b_names = _feature_names([0.05, 0.1, 0.2, 0.3, 0.5], [0.001, 0.01, 0.05])
    if feature_set == "v31_score_only":
        return v31_scores, v31_names
    if feature_set == "v8b_evidence_only":
        return v8b_features, v8b_names
    if feature_set == "late_fusion":
        return np.concatenate([v31_scores, v8b_features], axis=1), [*v31_names, *v8b_names]
    raise ValueError(f"Unsupported feature_set: {feature_set}")


def _fit_select_evaluate(
    features: np.ndarray,
    labels: np.ndarray,
    rows: list[dict[str, str | int]],
    *,
    train_split: str,
    threshold_split: str,
    train_domains: list[str] | None,
    eval_splits: list[str],
    c_values: list[float],
    thresholds: np.ndarray,
    seed: int,
    class_weight: str,
) -> dict:
    row_splits = np.asarray([str(row["split"]) for row in rows])
    row_domains = np.asarray([str(row["domain"]) for row in rows])
    train_mask = row_splits == train_split
    if train_domains:
        train_mask &= np.isin(row_domains, [str(domain) for domain in train_domains])
    threshold_mask = row_splits == threshold_split
    if not train_mask.any():
        raise ValueError("No rows selected for fusion training.")
    if not threshold_mask.any():
        raise ValueError("No rows selected for threshold calibration.")

    best_model = None
    best_scores = None
    best_selection = None
    best_rank: tuple[float, float] | None = None
    candidates: list[dict] = []
    for c_value in c_values:
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=float(c_value),
                max_iter=5000,
                class_weight=class_weight,
                solver="lbfgs",
                random_state=seed,
            ),
        )
        model.fit(features[train_mask], labels[train_mask])
        scores = model.predict_proba(features)[:, 1]
        selection = _choose_threshold(labels[threshold_mask], scores[threshold_mask], thresholds)
        val_report = _classification_report(
            labels[threshold_mask],
            scores[threshold_mask],
            float(selection["best"]["threshold"]),
        )
        rank = (
            selection["best"]["balanced_accuracy"],
            val_report.get("auroc") or 0.0,
        )
        candidates.append(
            {
                "c_value": float(c_value),
                "threshold_selection": selection["best"],
                "threshold_split_metrics": val_report,
            }
        )
        if best_rank is None or rank > best_rank:
            best_model = model
            best_scores = scores
            best_selection = selection
            best_rank = rank

    assert best_model is not None
    assert best_scores is not None
    assert best_selection is not None
    selected_threshold = float(best_selection["best"]["threshold"])

    by_split: dict[str, dict] = {}
    by_domain: dict[str, dict] = {}
    for split in eval_splits:
        split_mask = row_splits == split
        if split_mask.any():
            by_split[split] = {
                "selected_threshold": _classification_report(
                    labels[split_mask],
                    best_scores[split_mask],
                    selected_threshold,
                ),
                "threshold_0_5": _classification_report(
                    labels[split_mask],
                    best_scores[split_mask],
                    0.5,
                ),
            }
            for domain in sorted(set(row_domains[split_mask].tolist())):
                mask = split_mask & (row_domains == domain)
                if mask.any():
                    by_domain[f"{split}:{domain}"] = _classification_report(
                        labels[mask],
                        best_scores[mask],
                        selected_threshold,
                    )

    logreg = best_model.named_steps["logisticregression"]
    return {
        "train_split": train_split,
        "train_domains": sorted(set(row_domains[train_mask].tolist())),
        "n_train_fit": int(train_mask.sum()),
        "threshold_split": threshold_split,
        "threshold_selection": best_selection,
        "classifier_candidates": candidates,
        "metrics_by_split": by_split,
        "metrics_by_domain": by_domain,
        "coef": logreg.coef_.astype(float).tolist(),
        "intercept": logreg.intercept_.astype(float).tolist(),
    }


def run(config_path: str) -> dict:
    config_path_obj = Path(config_path).resolve()
    project_root = config_path_obj.parents[1]
    config = load_app_config(config_path_obj)
    version = str(config["project"]["version"])
    data_cfg = config["data"]

    items = _read_items(config, project_root)
    classifier_session = InferenceSession.from_config_path(
        resolve_project_path(project_root, config["classifier"]["config_path"]),
        checkpoint_path=str(resolve_project_path(project_root, config["classifier"]["checkpoint_path"])),
    )

    seg_config_path = resolve_project_path(project_root, config["segmentation"]["config_path"])
    seg_config = load_app_config(seg_config_path)
    seg_checkpoint = resolve_project_path(project_root, config["segmentation"]["checkpoint_path"])
    segmenter = _load_segmenter(seg_config, project_root, seg_checkpoint)
    from drscreen.cli.lesion_evidence_classifier import _build_transform

    seg_transform = _build_transform(seg_config, project_root)
    batch_size = int(data_cfg.get("batch_size", 16))
    num_workers = int(data_cfg.get("num_workers", 0))

    v31_scores, labels, rows = _extract_v31_scores(
        classifier_session,
        items,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    v8b_features, v8b_labels, v8b_rows = _extract_features(
        segmenter,
        items,
        transform=seg_transform,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    if not np.array_equal(labels, v8b_labels):
        raise RuntimeError("v31 and v8b extraction labels are not aligned.")
    if [row["image_id"] for row in rows] != [row["image_id"] for row in v8b_rows]:
        raise RuntimeError("v31 and v8b extraction rows are not aligned.")

    fusion_cfg = config.get("fusion", {})
    threshold_cfg = fusion_cfg.get("thresholds", {})
    thresholds = np.linspace(
        float(threshold_cfg.get("min", 0.01)),
        float(threshold_cfg.get("max", 0.99)),
        int(threshold_cfg.get("steps", 99)),
    )
    c_values = [float(v) for v in fusion_cfg.get("c_values", [1.0])]
    feature_sets = [str(v) for v in fusion_cfg.get("feature_sets", ["late_fusion"])]
    train_variants = fusion_cfg.get("train_variants") or {"all_train": {"train_domains": None}}
    eval_splits = [str(v) for v in data_cfg.get("eval_splits", ["external_test"])]

    results: dict[str, dict] = {}
    for variant_name, variant_cfg in train_variants.items():
        train_domains = None
        if isinstance(variant_cfg, dict) and variant_cfg.get("train_domains") is not None:
            train_domains = [str(v) for v in variant_cfg["train_domains"]]
        for feature_set in feature_sets:
            matrix, feature_names = _build_feature_matrix(
                feature_set,
                v31_scores=v31_scores,
                v8b_features=v8b_features,
            )
            key = f"{variant_name}:{feature_set}"
            result = _fit_select_evaluate(
                matrix,
                labels,
                rows,
                train_split=str(data_cfg.get("train_split", "train")),
                threshold_split=str(data_cfg.get("threshold_split", "val")),
                train_domains=train_domains,
                eval_splits=eval_splits,
                c_values=c_values,
                thresholds=thresholds,
                seed=int(fusion_cfg.get("seed", 43)),
                class_weight=str(fusion_cfg.get("class_weight", "balanced")),
            )
            result["feature_set"] = feature_set
            result["feature_names"] = feature_names
            results[key] = result

    def _external_auroc(item: tuple[str, dict]) -> float:
        section = item[1]["metrics_by_split"].get("external_test", {})
        return float(section.get("selected_threshold", {}).get("auroc") or -1.0)

    best_key, best_result = max(results.items(), key=_external_auroc)
    output = {
        "version": version,
        "classifier_config": str(resolve_project_path(project_root, config["classifier"]["config_path"])),
        "classifier_checkpoint": str(resolve_project_path(project_root, config["classifier"]["checkpoint_path"])),
        "segmentation_config": str(seg_config_path),
        "segmentation_checkpoint": str(seg_checkpoint),
        "manifest_path": str(resolve_project_path(project_root, data_cfg["manifest_path"])),
        "n_images": int(len(items)),
        "results": results,
        "best_by_external_auroc": {
            "key": best_key,
            "metrics": best_result["metrics_by_split"].get("external_test"),
        },
        "decision_note": (
            "Diagnostic only. Late fusion does not change configs/base.yaml, "
            "artifacts/checkpoints/best.pt, backend, or frontend."
        ),
    }
    output_dir = get_run_evaluation_dir(project_root, version)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{version}_metrics.json"
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"Saved: {output_path}")
    print(json.dumps(output["best_by_external_auroc"], indent=2))
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate v31 + v8b late fusion.")
    parser.add_argument("--config", required=True, help="Path to late-fusion config.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
