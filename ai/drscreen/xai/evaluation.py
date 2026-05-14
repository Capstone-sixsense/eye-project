from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

from drscreen.infer.service import InferenceSession
from drscreen.settings import get_run_evaluation_dir
from drscreen.xai.gradcam import generate_gradcam, resolve_default_target_layer
from drscreen.xai.iou import (
    LESION_CODES,
    binarize_cam,
    compute_auc_iou,
    compute_auprc,
    compute_iou,
    load_lesion_masks,
    load_maples_masks,
    make_center_gaussian_cam,
    make_random_cam,
    make_retina_uniform_cam,
    normalize_cam_fov,
    pointing_game,
    union_mask,
)

_SPLIT_IMAGE_SUBDIR = {
    "train": "a. Training Set",
    "test": "b. Testing Set",
}


def _build_retina_mask(image: Image.Image) -> np.ndarray:
    rgb = np.asarray(image.convert("RGB"))
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    _, thresh = cv2.threshold(gray, 8, 255, cv2.THRESH_BINARY)
    min_dim = min(gray.shape[:2])
    ks = max(3, min(11, (min_dim // 50) * 2 + 1))
    kernel = np.ones((ks, ks), dtype=np.uint8)
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(cleaned)
    if n <= 1:
        return np.ones(gray.shape, dtype=np.uint8)
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return (labels == largest).astype(np.uint8)


def _eval_cam(
    cam: np.ndarray,
    retina_mask: np.ndarray,
    gt_masks: dict,
    gt_union,
    top_percents: list[float],
) -> dict:
    cam_norm = normalize_cam_fov(cam, retina_mask)

    result: dict = {
        "pointing_game": None,
        "auprc": None,
        "auc_iou": None,
        "thresholds": {},
    }

    if gt_union is not None:
        result["pointing_game"] = pointing_game(cam_norm, gt_union)
        result["auprc"] = compute_auprc(cam_norm, gt_union, retina_mask)
        result["auc_iou"] = compute_auc_iou(cam_norm, retina_mask, gt_union)

    for top_pct in top_percents:
        binary_cam = binarize_cam(cam_norm, retina_mask, top_percent=top_pct)
        key = f"top{int(top_pct * 100):02d}"
        per_code: dict[str, float | None] = {}
        for code in LESION_CODES:
            per_code[code] = compute_iou(binary_cam, gt_masks[code]) if code in gt_masks else None
        result["thresholds"][key] = {
            "iou_union": compute_iou(binary_cam, gt_union) if gt_union is not None else None,
            "iou_per_lesion": per_code,
        }

    return result


def _process_image(
    session: InferenceSession,
    image_path: Path,
    mask_base_dir: Path,
    gradcam_method: str,
    top_percents: list[float],
    target_layer=None,
    use_seg_head: bool = False,
    run_baselines: bool = False,
    method_override: str | None = None,
    use_mil_attention: bool = False,
    mask_loader=None,
) -> dict:
    image_stem = image_path.stem
    pil_image = Image.open(image_path).convert("RGB")
    image_tensor = session.eval_transform(pil_image).to(session.device)
    cam_h, cam_w = image_tensor.shape[-2], image_tensor.shape[-1]

    active_method = method_override if method_override is not None else gradcam_method

    if use_mil_attention:
        attn = session.model.get_attention_map(image_tensor.unsqueeze(0))
        cam = attn[0].cpu().numpy()
    elif use_seg_head:
        seg_prob = session.model.predict_seg(image_tensor.unsqueeze(0))
        cam_raw = seg_prob[0, 0].detach().cpu().numpy()
        if cam_raw.shape != (cam_h, cam_w):
            cam_raw = cv2.resize(cam_raw, (cam_w, cam_h), interpolation=cv2.INTER_LINEAR)
        cam = cam_raw
    else:
        gradcam = generate_gradcam(
            session.model,
            image_tensor.unsqueeze(0),
            method=active_method,
            target_layer=target_layer,
        )
        cam = gradcam.heatmap[0].detach().cpu().numpy()

    pil_resized = pil_image.resize((cam_w, cam_h), Image.BILINEAR)
    retina_mask = _build_retina_mask(pil_resized)
    if mask_loader is not None:
        gt_masks = mask_loader(image_stem, (cam_w, cam_h))
    else:
        gt_masks = load_lesion_masks(mask_base_dir, image_stem, target_size=(cam_w, cam_h))
    gt_union = union_mask(gt_masks)

    metrics = _eval_cam(cam, retina_mask, gt_masks, gt_union, top_percents)
    result: dict = {"image_id": image_stem, "masks_present": list(gt_masks.keys()), **metrics}

    if run_baselines and gt_union is not None:
        rng = np.random.default_rng(0)
        baseline_cams = {
            "random": make_random_cam(retina_mask, rng),
            "center_gaussian": make_center_gaussian_cam(retina_mask),
            "retina_uniform": make_retina_uniform_cam(retina_mask),
        }
        result["baselines"] = {
            name: _eval_cam(bcam, retina_mask, gt_masks, gt_union, top_percents)
            for name, bcam in baseline_cams.items()
        }

    return result


def _agg_scalar(per_image: list[dict], key: str) -> dict | None:
    vals = [r[key] for r in per_image if r.get(key) is not None]
    if not vals:
        return None
    return {"mean": float(np.mean(vals)), "std": float(np.std(vals)), "n": len(vals)}


def _aggregate(per_image: list[dict], top_percents: list[float]) -> dict:
    agg: dict = {
        "pointing_game": _agg_scalar(per_image, "pointing_game"),
        "auprc": _agg_scalar(per_image, "auprc"),
        "auc_iou": _agg_scalar(per_image, "auc_iou"),
        "thresholds": {},
    }

    for top_pct in top_percents:
        key = f"top{int(top_pct * 100):02d}"
        union_ious = [
            r["thresholds"][key]["iou_union"]
            for r in per_image
            if r["thresholds"][key]["iou_union"] is not None
        ]
        per_code_ious: dict[str, list[float]] = {c: [] for c in LESION_CODES}
        for r in per_image:
            for code in LESION_CODES:
                v = r["thresholds"][key]["iou_per_lesion"].get(code)
                if v is not None:
                    per_code_ious[code].append(v)
        agg["thresholds"][key] = {
            "mean_iou_union": float(np.mean(union_ious)) if union_ious else None,
            "n_images_with_gt": len(union_ious),
            "per_lesion": {
                code: {"mean_iou": float(np.mean(vals)) if vals else None, "n": len(vals)}
                for code, vals in per_code_ious.items()
            },
        }

    if per_image and "baselines" in per_image[0]:
        baseline_names = list(per_image[0]["baselines"].keys())
        agg["baselines"] = {}
        for name in baseline_names:
            baseline_records = [r["baselines"][name] for r in per_image if "baselines" in r]
            agg["baselines"][name] = {
                "pointing_game": _agg_scalar(baseline_records, "pointing_game"),
                "auprc": _agg_scalar(baseline_records, "auprc"),
                "auc_iou": _agg_scalar(baseline_records, "auc_iou"),
                "thresholds": {},
            }
            for top_pct in top_percents:
                key = f"top{int(top_pct * 100):02d}"
                b_union_ious = [
                    br["thresholds"][key]["iou_union"]
                    for br in baseline_records
                    if br["thresholds"][key]["iou_union"] is not None
                ]
                agg["baselines"][name]["thresholds"][key] = {
                    "mean_iou_union": float(np.mean(b_union_ious)) if b_union_ious else None,
                }

    return agg


def evaluate(
    config_path: str,
    split: str = "train",
    idrid_root: str | None = None,
    top_percents: list[float] | None = None,
    target_block: int | None = None,
    output_path: str | None = None,
    use_seg_head: bool = False,
    use_mil_attention: bool = False,
    run_baselines: bool = False,
    gate_sigma: float = 2.0,
) -> dict:
    if top_percents is None:
        top_percents = [0.10, 0.20, 0.30]

    project_root = Path(config_path).resolve().parents[1]
    idrid_root_path = Path(idrid_root) if idrid_root else project_root / "data" / "raw" / "IDRiD"

    split_subdir = _SPLIT_IMAGE_SUBDIR.get(split)
    if split_subdir is None:
        raise ValueError(f"split must be 'train' or 'test', got {split!r}")

    image_dir = idrid_root_path / "A. Segmentation" / "1. Original Images" / split_subdir
    mask_base_dir = idrid_root_path / "A. Segmentation" / "2. All Segmentation Groundtruths" / split_subdir

    if not image_dir.exists():
        raise FileNotFoundError(f"IDRiD image directory not found: {image_dir}")
    if not mask_base_dir.exists():
        raise FileNotFoundError(f"IDRiD mask directory not found: {mask_base_dir}")

    session = InferenceSession.from_config_path(config_path)
    session.preprocessor = None

    gradcam_method = session.config.get("infer", {}).get("gradcam_method", "gradcam")
    version = session.config.get("project", {}).get("version", "unknown")

    target_layer = None
    if use_mil_attention:
        block_label = "mil_attention"
    elif use_seg_head:
        block_label = "seg_head"
    else:
        block_label = "default"
    if not use_seg_head and not use_mil_attention and target_block is not None:
        blocks = getattr(session.model, "blocks", getattr(session.model, "features", None))
        if blocks is None:
            raise ValueError("Model has neither .blocks nor .features attribute")
        target_layer = blocks[target_block]
        block_label = f"block{target_block}"

    image_paths = sorted(image_dir.glob("*.jpg")) + sorted(image_dir.glob("*.png"))
    if not image_paths:
        raise FileNotFoundError(f"No images found in {image_dir}")

    print(f"Config    : {config_path}")
    print(f"Version   : {version}")
    print(f"Method    : {gradcam_method}")
    print(f"Layer     : {block_label}")
    print(f"Split     : {split}  ({len(image_paths)} images)")
    print(f"Thresholds: top {[int(p*100) for p in top_percents]}%")
    print(f"Baselines : {run_baselines}")
    print()

    per_image: list[dict] = []
    for i, image_path in enumerate(image_paths, 1):
        rec = _process_image(
            session, image_path, mask_base_dir, gradcam_method, top_percents, target_layer,
            use_seg_head=use_seg_head, use_mil_attention=use_mil_attention,
            run_baselines=run_baselines,
        )
        iou20 = rec["thresholds"].get("top20", {}).get("iou_union")
        print(
            f"  [{i:02d}/{len(image_paths)}] {rec['image_id']}"
            f"  iou={iou20:.4f}" if iou20 is not None else
            f"  [{i:02d}/{len(image_paths)}] {rec['image_id']}  iou=N/A",
            f"  auprc={rec['auprc']:.4f}" if rec["auprc"] is not None else "  auprc=N/A",
            f"  pg={rec['pointing_game']}",
        )
        per_image.append(rec)

    aggregate = _aggregate(per_image, top_percents)

    print()
    print("=== Aggregate ===")
    if aggregate["pointing_game"]:
        pg = aggregate["pointing_game"]
        print(f"Pointing game : {pg['mean']:.4f} ± {pg['std']:.4f}  (n={pg['n']})")
    if aggregate["auprc"]:
        ap = aggregate["auprc"]
        print(f"AUPRC         : {ap['mean']:.4f} ± {ap['std']:.4f}  (n={ap['n']})")
    if aggregate["auc_iou"]:
        ai = aggregate["auc_iou"]
        print(f"AUC-IoU       : {ai['mean']:.4f} ± {ai['std']:.4f}  (n={ai['n']})")
    for top_pct in top_percents:
        key = f"top{int(top_pct * 100):02d}"
        agg_t = aggregate["thresholds"][key]
        miou = agg_t["mean_iou_union"]
        miou_str = f"{miou:.4f}" if miou is not None else "N/A"
        print(f"IoU top{int(top_pct*100):02d}%    : {miou_str}  (n={agg_t['n_images_with_gt']})")

    if run_baselines and "baselines" in aggregate:
        print()
        print("=== Baselines ===")
        for name, bagg in aggregate["baselines"].items():
            ap_b = bagg["auprc"]["mean"] if bagg["auprc"] else None
            ai_b = bagg["auc_iou"]
            iou20_b = bagg["thresholds"].get("top20", {}).get("mean_iou_union")
            auc_iou_str = (
                f"  AUC-IoU={ai_b['mean']:.4f}±{ai_b['std']:.4f}" if ai_b else "  AUC-IoU=N/A"
            )
            print(
                f"  {name:20s}  AUPRC={ap_b:.4f}" if ap_b is not None else
                f"  {name:20s}  AUPRC=N/A",
                auc_iou_str,
                f"  IoU-top20={iou20_b:.4f}" if iou20_b is not None else "  IoU-top20=N/A",
            )

        model_auc = aggregate["auc_iou"]["mean"] if aggregate["auc_iou"] else None
        cg_bagg = aggregate["baselines"].get("center_gaussian")
        if model_auc is not None and cg_bagg and cg_bagg["auc_iou"]:
            cg = cg_bagg["auc_iou"]
            threshold = cg["mean"] + gate_sigma * cg["std"]
            gate = model_auc > threshold
            print()
            print(f"=== Phase-0 Gate (AUC-IoU > center_gaussian + {gate_sigma}σ) ===")
            print(f"  Model AUC-IoU  : {model_auc:.4f}")
            print(f"  Threshold      : {cg['mean']:.4f} + {gate_sigma}×{cg['std']:.4f} = {threshold:.4f}")
            print(f"  Gate           : {'PASS' if gate else 'FAIL'}")

    phase0_gate: dict | None = None
    cg_bagg = aggregate.get("baselines", {}).get("center_gaussian") if run_baselines else None
    model_auc = aggregate["auc_iou"]["mean"] if aggregate.get("auc_iou") else None
    if model_auc is not None and cg_bagg and cg_bagg.get("auc_iou"):
        cg = cg_bagg["auc_iou"]
        threshold = cg["mean"] + 2 * cg["std"]
        phase0_gate = {
            "model_auc_iou": model_auc,
            "center_gaussian_mean": cg["mean"],
            "center_gaussian_std": cg["std"],
            "threshold": threshold,
            "pass": bool(model_auc > threshold),
        }

    output = {
        "version": version,
        "checkpoint_path": str(session.checkpoint_path),
        "gradcam_method": gradcam_method,
        "target_block": block_label,
        "split": split,
        "n_images": len(per_image),
        "top_percents": top_percents,
        "phase0_gate": phase0_gate,
        "aggregate": aggregate,
        "per_image": per_image,
    }

    if output_path is None:
        eval_dir = get_run_evaluation_dir(project_root, str(version))
        eval_dir.mkdir(parents=True, exist_ok=True)
        output_path = str(eval_dir / f"xai_iou_{version}_{block_label}_{split}.json")

    Path(output_path).write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"\nSaved: {output_path}")
    return output


def compare_xai_methods(
    config_path: str,
    methods: list[str],
    split: str = "test",
    idrid_root: str | None = None,
    top_percents: list[float] | None = None,
    run_baselines: bool = False,
    output_path: str | None = None,
) -> dict:
    if top_percents is None:
        top_percents = [0.10, 0.20, 0.30]

    project_root = Path(config_path).resolve().parents[1]
    idrid_root_path = Path(idrid_root) if idrid_root else project_root / "data" / "raw" / "IDRiD"

    split_subdir = _SPLIT_IMAGE_SUBDIR.get(split)
    if split_subdir is None:
        raise ValueError(f"split must be 'train' or 'test', got {split!r}")

    image_dir = idrid_root_path / "A. Segmentation" / "1. Original Images" / split_subdir
    mask_base_dir = idrid_root_path / "A. Segmentation" / "2. All Segmentation Groundtruths" / split_subdir
    if not image_dir.exists():
        raise FileNotFoundError(f"IDRiD image directory not found: {image_dir}")

    session = InferenceSession.from_config_path(config_path)
    session.preprocessor = None

    version = session.config.get("project", {}).get("version", "unknown")
    default_method = session.config.get("infer", {}).get("gradcam_method", "gradcam")
    target_layer = resolve_default_target_layer(session.model)

    image_paths = sorted(image_dir.glob("*.jpg")) + sorted(image_dir.glob("*.png"))
    if not image_paths:
        raise FileNotFoundError(f"No images found in {image_dir}")

    print(f"Config  : {config_path}")
    print(f"Version : {version}")
    print(f"Methods : {methods}")
    print(f"Split   : {split}  ({len(image_paths)} images)")
    print()

    results_by_method: dict[str, dict] = {}
    for method in methods:
        print(f"--- {method} ---")
        per_image: list[dict] = []
        for i, image_path in enumerate(image_paths, 1):
            rec = _process_image(
                session, image_path, mask_base_dir, default_method, top_percents,
                target_layer=target_layer,
                run_baselines=(run_baselines and method == methods[0]),
                method_override=method,
            )
            iou20 = rec["thresholds"].get("top20", {}).get("iou_union")
            print(
                f"  [{i:02d}/{len(image_paths)}] {rec['image_id']}"
                f"  iou={iou20:.4f}" if iou20 is not None else
                f"  [{i:02d}/{len(image_paths)}] {rec['image_id']}  iou=N/A",
                f"  auprc={rec['auprc']:.4f}" if rec["auprc"] is not None else "  auprc=N/A",
            )
            per_image.append(rec)
        results_by_method[method] = _aggregate(per_image, top_percents)
        print()

    print("=== Comparison ===")
    header = f"{'Method':<14}  {'AUPRC':>7}  {'AUC-IoU':>8}  {'IoU-10%':>8}  {'IoU-20%':>8}  {'IoU-30%':>8}  {'PG':>7}"
    print(header)
    print("-" * len(header))
    for method, agg in results_by_method.items():
        auprc = agg["auprc"]["mean"] if agg["auprc"] else None
        auc_iou = agg["auc_iou"]["mean"] if agg["auc_iou"] else None
        iou10 = agg["thresholds"].get("top10", {}).get("mean_iou_union")
        iou20 = agg["thresholds"].get("top20", {}).get("mean_iou_union")
        iou30 = agg["thresholds"].get("top30", {}).get("mean_iou_union")
        pg = agg["pointing_game"]["mean"] if agg["pointing_game"] else None

        def _f(v):
            return f"{v:.4f}" if v is not None else "  N/A"

        print(f"{method:<14}  {_f(auprc):>7}  {_f(auc_iou):>8}  {_f(iou10):>8}  {_f(iou20):>8}  {_f(iou30):>8}  {_f(pg):>7}")

    output = {
        "version": version,
        "split": split,
        "methods": methods,
        "top_percents": top_percents,
        "results_by_method": results_by_method,
    }

    if output_path is None:
        eval_dir = get_run_evaluation_dir(project_root, str(version))
        eval_dir.mkdir(parents=True, exist_ok=True)
        method_tag = "_".join(methods)
        output_path = str(eval_dir / f"xai_compare_{version}_{method_tag}_{split}.json")

    Path(output_path).write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"\nSaved: {output_path}")
    return output


def evaluate_maples(
    config_path: str,
    split: str = "test",
    maples_root: str | None = None,
    messidor_images_dir: str | None = None,
    top_percents: list[float] | None = None,
    target_block: int | None = None,
    output_path: str | None = None,
    run_baselines: bool = False,
    gate_sigma: float = 2.0,
) -> dict:
    """XAI evaluation against MAPLES-DR lesion masks on MESSIDOR images."""
    import yaml

    if top_percents is None:
        top_percents = [0.10, 0.20, 0.30]

    project_root = Path(config_path).resolve().parents[1]
    maples_root_path = (
        Path(maples_root) if maples_root
        else project_root / "data" / "raw" / "MAPLES-DR" / "AdditionalData"
    )
    annotations_dir = maples_root_path / "annotations"
    messidor_dir = (
        Path(messidor_images_dir) if messidor_images_dir
        else project_root / "data" / "raw" / "messidor" / "images"
    )

    with open(maples_root_path / "dataset_record.yaml") as f:
        record = yaml.safe_load(f)
    if split not in ("train", "test"):
        raise ValueError(f"split must be 'train' or 'test', got {split!r}")
    stems: list[str] = record[split]

    image_paths: list[Path] = []
    for stem in stems:
        for ext in (".tif", ".jpg", ".png"):
            p = messidor_dir / f"{stem}{ext}"
            if p.exists():
                image_paths.append(p)
                break
    if not image_paths:
        raise FileNotFoundError(f"No MESSIDOR images found in {messidor_dir}")

    session = InferenceSession.from_config_path(config_path)
    session.preprocessor = None
    gradcam_method = session.config.get("infer", {}).get("gradcam_method", "gradcam")
    version = session.config.get("project", {}).get("version", "unknown")

    target_layer = None
    block_label = "default"
    if target_block is not None:
        blocks = getattr(session.model, "blocks", getattr(session.model, "features", None))
        if blocks is None:
            raise ValueError("Model has neither .blocks nor .features attribute")
        target_layer = blocks[target_block]
        block_label = f"block{target_block}"

    def _mask_loader(image_stem: str, target_size: tuple[int, int]) -> dict:
        return load_maples_masks(annotations_dir, image_stem, target_size)

    print(f"Config    : {config_path}")
    print(f"Version   : {version}")
    print(f"Method    : {gradcam_method}")
    print(f"Layer     : {block_label}")
    print(f"Dataset   : MAPLES-DR ({split}, {len(image_paths)} images)")
    print(f"Thresholds: top {[int(p*100) for p in top_percents]}%")
    print(f"Baselines : {run_baselines}")
    print()

    per_image: list[dict] = []
    for i, image_path in enumerate(image_paths, 1):
        rec = _process_image(
            session, image_path, None, gradcam_method, top_percents, target_layer,
            run_baselines=run_baselines, mask_loader=_mask_loader,
        )
        iou20 = rec["thresholds"].get("top20", {}).get("iou_union")
        print(
            f"  [{i:02d}/{len(image_paths)}] {rec['image_id']}"
            f"  iou={iou20:.4f}" if iou20 is not None else
            f"  [{i:02d}/{len(image_paths)}] {rec['image_id']}  iou=N/A",
            f"  auprc={rec['auprc']:.4f}" if rec["auprc"] is not None else "  auprc=N/A",
            f"  pg={rec['pointing_game']}",
        )
        per_image.append(rec)

    aggregate = _aggregate(per_image, top_percents)

    print()
    print("=== Aggregate ===")
    if aggregate["pointing_game"]:
        pg = aggregate["pointing_game"]
        print(f"Pointing game : {pg['mean']:.4f} ± {pg['std']:.4f}  (n={pg['n']})")
    if aggregate["auprc"]:
        ap = aggregate["auprc"]
        print(f"AUPRC         : {ap['mean']:.4f} ± {ap['std']:.4f}  (n={ap['n']})")
    if aggregate["auc_iou"]:
        ai = aggregate["auc_iou"]
        print(f"AUC-IoU       : {ai['mean']:.4f} ± {ai['std']:.4f}  (n={ai['n']})")
    for top_pct in top_percents:
        key = f"top{int(top_pct * 100):02d}"
        agg_t = aggregate["thresholds"][key]
        miou = agg_t["mean_iou_union"]
        miou_str = f"{miou:.4f}" if miou is not None else "N/A"
        print(f"IoU top{int(top_pct*100):02d}%    : {miou_str}  (n={agg_t['n_images_with_gt']})")

    if run_baselines and "baselines" in aggregate:
        print()
        print("=== Baselines ===")
        for name, bagg in aggregate["baselines"].items():
            ap_b = bagg["auprc"]["mean"] if bagg["auprc"] else None
            ai_b = bagg["auc_iou"]
            iou20_b = bagg["thresholds"].get("top20", {}).get("mean_iou_union")
            auc_iou_str = (
                f"  AUC-IoU={ai_b['mean']:.4f}±{ai_b['std']:.4f}" if ai_b else "  AUC-IoU=N/A"
            )
            print(
                f"  {name:20s}  AUPRC={ap_b:.4f}" if ap_b is not None else
                f"  {name:20s}  AUPRC=N/A",
                auc_iou_str,
                f"  IoU-top20={iou20_b:.4f}" if iou20_b is not None else "  IoU-top20=N/A",
            )

        model_auc = aggregate["auc_iou"]["mean"] if aggregate["auc_iou"] else None
        cg_bagg = aggregate["baselines"].get("center_gaussian")
        if model_auc is not None and cg_bagg and cg_bagg["auc_iou"]:
            cg = cg_bagg["auc_iou"]
            threshold = cg["mean"] + gate_sigma * cg["std"]
            gate = model_auc > threshold
            print()
            print(f"=== Phase-0 Gate (AUC-IoU > center_gaussian + {gate_sigma}σ) ===")
            print(f"  Model AUC-IoU  : {model_auc:.4f}")
            print(f"  Threshold      : {cg['mean']:.4f} + {gate_sigma}×{cg['std']:.4f} = {threshold:.4f}")
            print(f"  Gate           : {'PASS' if gate else 'FAIL'}")

    phase0_gate: dict | None = None
    cg_bagg = aggregate.get("baselines", {}).get("center_gaussian") if run_baselines else None
    model_auc = aggregate["auc_iou"]["mean"] if aggregate.get("auc_iou") else None
    if model_auc is not None and cg_bagg and cg_bagg.get("auc_iou"):
        cg = cg_bagg["auc_iou"]
        threshold = cg["mean"] + 2 * cg["std"]
        phase0_gate = {
            "model_auc_iou": model_auc,
            "center_gaussian_mean": cg["mean"],
            "center_gaussian_std": cg["std"],
            "threshold": threshold,
            "pass": bool(model_auc > threshold),
        }

    output = {
        "version": version,
        "checkpoint_path": str(session.checkpoint_path),
        "gradcam_method": gradcam_method,
        "target_block": block_label,
        "split": split,
        "dataset": "maples",
        "n_images": len(per_image),
        "top_percents": top_percents,
        "phase0_gate": phase0_gate,
        "aggregate": aggregate,
        "per_image": per_image,
    }

    if output_path is None:
        eval_dir = get_run_evaluation_dir(project_root, str(version))
        eval_dir.mkdir(parents=True, exist_ok=True)
        output_path = str(eval_dir / f"xai_maples_{version}_{block_label}_{split}.json")

    Path(output_path).write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"\nSaved: {output_path}")
    return output
