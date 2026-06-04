"""Phase 4-F G1: DFR-style final-layer reweighting for v31.

This is a diagnostic track. It freezes the active v31 backbone/gated-pooling
path, extracts classifier pre-logit features from a group-balanced reweighting
set, fits a linear classifier, and writes a new checkpoint with only the final
classifier weights changed.

(한글 요약) DFR(Deep Feature Reweighting) 진단: backbone은 동결한 채 마지막 분류층만
그룹 균형 집합에서 다시 학습해, shortcut 의존을 줄일 수 있는지 본다. backbone 특징이
그대로라 DDR 외부 성능이 무너져(연구용) 배포 후보가 아니다.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from drscreen.infer.service import InferenceSession
from drscreen.models.build import get_classifier_module
from drscreen.settings import (
    get_run_artifact_dir,
    get_run_checkpoint_dir,
    resolve_project_path,
)
from drscreen.xai.iou import load_lesion_masks, load_maples_masks, union_mask


@dataclass(frozen=True)
class DFRItem:
    path: Path
    image_id: str
    label: int
    group: str
    domain: str
    source_path: Path | None = None


def _read_manifest_rows(project_root: Path, manifest: str) -> list[dict[str, str]]:
    manifest_path = resolve_project_path(project_root, manifest)
    data_root = project_root / "data/raw"
    rows: list[dict[str, str]] = []
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


def _idrid_train_lesion_items(project_root: Path) -> list[DFRItem]:
    image_dir = project_root / "data/raw/IDRiD/A. Segmentation/1. Original Images/a. Training Set"
    mask_dir = project_root / "data/raw/IDRiD/A. Segmentation/2. All Segmentation Groundtruths/a. Training Set"
    items: list[DFRItem] = []
    for image_path in sorted(image_dir.glob("*.jpg")) + sorted(image_dir.glob("*.png")):
        masks = load_lesion_masks(mask_dir, image_path.stem)
        lesion = union_mask(masks)
        if lesion is None or lesion.sum() == 0:
            continue
        items.append(
            DFRItem(
                path=image_path,
                image_id=image_path.stem,
                label=1,
                group="G_A_lesion_IDRiD",
                domain="IDRiD_SEG",
            )
        )
    return items


def _maples_train_lesion_items(project_root: Path, manifest_rows: list[dict[str, str]]) -> list[DFRItem]:
    items: list[DFRItem] = []
    ann_dir = project_root / "data/raw/MAPLES-DR/AdditionalData/annotations"
    for row in manifest_rows:
        if row.get("domain") != "MAPLES" or row.get("split") != "train":
            continue
        image_path = Path(row["_resolved_image_path"])
        masks = load_maples_masks(ann_dir, image_path.stem)
        lesion = union_mask(masks)
        if lesion is None or lesion.sum() == 0:
            continue
        items.append(
            DFRItem(
                path=image_path,
                image_id=str(row.get("image_id", image_path.stem)),
                label=1,
                group="G_B_lesion_MAPLES",
                domain="MAPLES",
            )
        )
    return items


def _manifest_normals(
    manifest_rows: list[dict[str, str]],
    *,
    domain: str,
    group: str,
) -> list[DFRItem]:
    items: list[DFRItem] = []
    for row in manifest_rows:
        if row.get("domain") != domain or row.get("split") != "train" or str(row.get("label")) != "0":
            continue
        image_path = Path(row["_resolved_image_path"])
        items.append(
            DFRItem(
                path=image_path,
                image_id=str(row.get("image_id", image_path.stem)),
                label=0,
                group=group,
                domain=domain,
            )
        )
    return items


def _rgb_stats(paths: list[Path], *, max_images: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = random.Random(seed)
    sample = list(paths)
    rng.shuffle(sample)
    pixels: list[np.ndarray] = []
    for path in sample[:max_images]:
        with Image.open(path) as image:
            arr = np.asarray(image.convert("RGB").resize((256, 256), Image.BILINEAR), dtype=np.float32)
        pixels.append(arr.reshape(-1, 3))
    if not pixels:
        return np.array([128.0, 128.0, 128.0], dtype=np.float32), np.ones(3, dtype=np.float32)
    stacked = np.concatenate(pixels, axis=0)
    return stacked.mean(axis=0), stacked.std(axis=0).clip(min=1.0)


def _reinhard_to_stats(image: Image.Image, target_mean: np.ndarray, target_std: np.ndarray) -> Image.Image:
    arr = np.asarray(image.convert("RGB"), dtype=np.float32)
    flat = arr.reshape(-1, 3)
    src_mean = flat.mean(axis=0)
    src_std = flat.std(axis=0).clip(min=1.0)
    out = (arr - src_mean.reshape(1, 1, 3)) / src_std.reshape(1, 1, 3)
    out = out * target_std.reshape(1, 1, 3) + target_mean.reshape(1, 1, 3)
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))


def _build_reweighting_items(
    project_root: Path,
    *,
    manifest: str,
    output_dir: Path,
    per_group: int,
    seed: int,
) -> tuple[list[DFRItem], dict]:
    manifest_rows = _read_manifest_rows(project_root, manifest)
    rng = random.Random(seed)

    group_a = _idrid_train_lesion_items(project_root)
    group_b = _maples_train_lesion_items(project_root, manifest_rows)
    group_c = _manifest_normals(manifest_rows, domain="IDRiD", group="G_C_normal_IDRiD")
    messidor_normals = _manifest_normals(
        manifest_rows,
        domain="Messidor",
        group="G_D_normal_MESSIDOR_to_MAPLES_synth_source",
    )

    for group in (group_a, group_b, group_c, messidor_normals):
        rng.shuffle(group)
    n = min(per_group, len(group_a), len(group_b), len(group_c), len(messidor_normals))
    if n <= 0:
        raise ValueError(
            "Cannot build DFR groups: "
            f"A={len(group_a)} B={len(group_b)} C={len(group_c)} Dsrc={len(messidor_normals)}"
        )

    selected_a = group_a[:n]
    selected_b = group_b[:n]
    selected_c = group_c[:n]
    selected_d_sources = messidor_normals[:n]

    maples_mean, maples_std = _rgb_stats([item.path for item in selected_b], max_images=min(n, 50), seed=seed)
    synth_dir = output_dir / "synthetic" / "MESSIDOR_to_MAPLES_synth"
    synth_dir.mkdir(parents=True, exist_ok=True)
    selected_d: list[DFRItem] = []
    for item in selected_d_sources:
        with Image.open(item.path) as image:
            synth = _reinhard_to_stats(image, maples_mean, maples_std)
        synth_path = synth_dir / f"{Path(item.image_id).stem}.png"
        synth.save(synth_path)
        selected_d.append(
            DFRItem(
                path=synth_path,
                image_id=item.image_id,
                label=0,
                group="G_D_normal_MESSIDOR_to_MAPLES_synth",
                domain="MESSIDOR_to_MAPLES_synth",
                source_path=item.path,
            )
        )

    items = selected_a + selected_b + selected_c + selected_d
    rng.shuffle(items)
    summary = {
        "per_group_requested": per_group,
        "per_group_used": n,
        "available_counts": {
            "G_A_lesion_IDRiD": len(group_a),
            "G_B_lesion_MAPLES": len(group_b),
            "G_C_normal_IDRiD": len(group_c),
            "G_D_normal_Messidor_source": len(messidor_normals),
        },
        "selected_counts": {
            "G_A_lesion_IDRiD": len(selected_a),
            "G_B_lesion_MAPLES": len(selected_b),
            "G_C_normal_IDRiD": len(selected_c),
            "G_D_normal_MESSIDOR_to_MAPLES_synth": len(selected_d),
        },
        "maples_style_target_mean_rgb": [float(v) for v in maples_mean],
        "maples_style_target_std_rgb": [float(v) for v in maples_std],
    }
    return items, summary


def _write_reweighting_manifest(output_path: Path, items: list[DFRItem]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "image_id",
                "image_path",
                "label",
                "group",
                "domain",
                "source_image_path",
            ],
        )
        writer.writeheader()
        for item in items:
            writer.writerow(
                {
                    "image_id": item.image_id,
                    "image_path": str(item.path),
                    "label": item.label,
                    "group": item.group,
                    "domain": item.domain,
                    "source_image_path": str(item.source_path or ""),
                }
            )


def _preprocess_image(session: InferenceSession, path: Path) -> torch.Tensor:
    with Image.open(path) as image:
        image = image.convert("RGB")
        if session.preprocessor is not None:
            image = session.preprocessor(image)
        return session.eval_transform(image)


def _extract_prelogits(
    session: InferenceSession,
    items: list[DFRItem],
    *,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    model = session.model
    model.eval()
    features: list[np.ndarray] = []
    labels: list[int] = []
    device = session.device

    for start in range(0, len(items), batch_size):
        batch = items[start : start + batch_size]
        x = torch.stack([_preprocess_image(session, item.path) for item in batch], dim=0).to(device)
        with torch.no_grad():
            if getattr(model, "use_gated_pooling", False):
                feat_map = model.backbone.forward_features(x)
                seg_logits = model._seg_forward(output_size=feat_map.shape[-2:])
                if getattr(model, "seg_channels", 1) > 1 and getattr(model, "lesion_weights", None) is not None:
                    weights = torch.softmax(model.lesion_weights, dim=0).view(1, -1, 1, 1)
                    gate = (torch.sigmoid(seg_logits) * weights).sum(dim=1, keepdim=True)
                else:
                    gate = torch.sigmoid(seg_logits)
                gate = gate / gate.mean(dim=(2, 3), keepdim=True).clamp_min(1e-6)
                prelogits = model.backbone.forward_head(feat_map * gate, pre_logits=True)
            else:
                backbone = getattr(model, "backbone", model)
                if hasattr(backbone, "forward_features") and hasattr(backbone, "forward_head"):
                    prelogits = backbone.forward_head(backbone.forward_features(x), pre_logits=True)
                else:
                    raise ValueError("DFR feature extraction currently expects a timm-style backbone.")
        features.append(prelogits.detach().cpu().numpy().astype(np.float32))
        labels.extend(item.label for item in batch)
    return np.concatenate(features, axis=0), np.asarray(labels, dtype=np.int64)


def _fit_linear_probe(features: np.ndarray, labels: np.ndarray, *, seed: int, c_value: float) -> tuple[object, dict]:
    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=c_value,
            max_iter=5000,
            class_weight="balanced",
            solver="lbfgs",
            random_state=seed,
        ),
    )
    clf.fit(features, labels)
    pred = clf.predict(features)
    proba = clf.predict_proba(features)[:, 1]
    metrics = {
        "train_auroc": float(roc_auc_score(labels, proba)),
        "train_accuracy": float(accuracy_score(labels, pred)),
        "train_macro_f1": float(f1_score(labels, pred, average="macro", zero_division=0)),
    }
    return clf, metrics


def _install_classifier_weights(session: InferenceSession, clf: object) -> None:
    scaler = clf.named_steps["standardscaler"]
    logreg = clf.named_steps["logisticregression"]
    coef = logreg.coef_.astype(np.float32)
    intercept = logreg.intercept_.astype(np.float32)
    scale = scaler.scale_.astype(np.float32)
    mean = scaler.mean_.astype(np.float32)

    effective_weight = coef / scale.reshape(1, -1)
    effective_bias = intercept - (coef * mean.reshape(1, -1) / scale.reshape(1, -1)).sum(axis=1)
    classifier = get_classifier_module(
        str(session.config["model"]["architecture"]),
        session.model,
    )
    if classifier.weight.shape != torch.Size(effective_weight.shape):
        raise ValueError(
            f"Classifier weight shape mismatch: model={tuple(classifier.weight.shape)} "
            f"dfr={effective_weight.shape}"
        )
    with torch.no_grad():
        classifier.weight.copy_(torch.from_numpy(effective_weight).to(classifier.weight.device))
        classifier.bias.copy_(torch.from_numpy(effective_bias).to(classifier.bias.device))


def run_dfr(
    *,
    config_path: str,
    checkpoint: str | None,
    version: str,
    manifest: str,
    per_group: int,
    batch_size: int,
    seed: int,
    c_value: float,
) -> dict:
    session = InferenceSession.from_config_path(config_path, checkpoint_path=checkpoint)
    project_root = session.project_root
    run_dir = get_run_artifact_dir(project_root, version)
    checkpoint_dir = get_run_checkpoint_dir(project_root, version)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    eval_dir = run_dir / "evaluations"
    eval_dir.mkdir(parents=True, exist_ok=True)

    items, group_summary = _build_reweighting_items(
        project_root,
        manifest=manifest,
        output_dir=run_dir,
        per_group=per_group,
        seed=seed,
    )
    reweight_manifest = run_dir / "reweighting_manifest.csv"
    _write_reweighting_manifest(reweight_manifest, items)

    features, labels = _extract_prelogits(session, items, batch_size=batch_size)
    clf, train_metrics = _fit_linear_probe(features, labels, seed=seed, c_value=c_value)
    _install_classifier_weights(session, clf)

    dfr_config = json.loads(json.dumps(session.config))
    dfr_config.setdefault("project", {})["version"] = version
    dfr_config.setdefault("train", {})["dfr_source_checkpoint"] = str(session.checkpoint_path)
    dfr_config["train"]["dfr_per_group"] = per_group
    dfr_config["train"]["dfr_c_value"] = c_value
    dfr_config["train"]["dfr_reweighting_manifest"] = str(reweight_manifest)
    payload = {
        "architecture": dfr_config["model"]["architecture"],
        "num_outputs": dfr_config["model"]["num_outputs"],
        "label_names": list(dfr_config["labels"]["names"]),
        "config": dfr_config,
        "model_state_dict": session.model.state_dict(),
        "dfr": {
            "source_version": str(session.config.get("project", {}).get("version")),
            "source_checkpoint": str(session.checkpoint_path),
            "group_summary": group_summary,
            "train_metrics": train_metrics,
        },
    }
    best_path = checkpoint_dir / "best.pt"
    last_path = checkpoint_dir / "last.pt"
    torch.save(payload, best_path)
    torch.save(payload, last_path)

    summary = {
        "version": version,
        "source_version": str(session.config.get("project", {}).get("version")),
        "source_checkpoint": str(session.checkpoint_path),
        "checkpoint_path": str(best_path),
        "last_checkpoint_path": str(last_path),
        "reweighting_manifest": str(reweight_manifest),
        "features_shape": list(features.shape),
        "group_summary": group_summary,
        "train_metrics": train_metrics,
        "note": "Diagnostic DFR run: only final classifier weights are changed; not a product candidate.",
    }
    summary_path = checkpoint_dir / "training_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved: {best_path}")
    print(f"Saved: {summary_path}")
    print(json.dumps({"train_metrics": train_metrics, "group_summary": group_summary}, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Phase 4-F G1 DFR-style final-layer relearning.")
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--checkpoint")
    parser.add_argument("--version", default="v31_dfr_v1")
    parser.add_argument("--manifest", default="data/processed/manifest_with_maples_r1plus.csv")
    parser.add_argument("--per-group", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--c-value", type=float, default=1.0)
    args = parser.parse_args()
    run_dfr(
        config_path=args.config,
        checkpoint=args.checkpoint,
        version=args.version,
        manifest=args.manifest,
        per_group=args.per_group,
        batch_size=args.batch_size,
        seed=args.seed,
        c_value=args.c_value,
    )


if __name__ == "__main__":
    main()
