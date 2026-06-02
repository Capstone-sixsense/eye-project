"""Audit v8b teacher mask quality before mask-conditioned augmentation."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from drscreen.data.transforms import (  # noqa: E402
    FundusPreprocess,
    preprocess_kwargs_from_config,
)
from drscreen.settings import load_app_config, resolve_project_path  # noqa: E402
from eval_seg_evidence import (  # noqa: E402
    _eval_preprocessing_enabled,
    _eval_transform,
    _idrid_items,
    _load_model,
    _maples_items,
    _mask_dict_to_eval_tensor,
)


def _provider_items(project_root: Path, provider: str, split: str):
    if provider == "idrid":
        return _idrid_items(project_root, split)
    if provider == "maples":
        return _maples_items(project_root, split)
    raise ValueError(f"Unsupported provider: {provider}")


def _union_mask(mask: np.ndarray) -> np.ndarray:
    return mask.max(axis=0) > 0


def _dilated_area_ratio(mask: np.ndarray, dilate_px: int) -> float:
    if dilate_px <= 0:
        return float(mask.mean())
    kernel_size = int(dilate_px) * 2 + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    dilated = cv2.dilate(mask.astype(np.uint8), kernel, iterations=1) > 0
    return float(dilated.mean())


def run_audit(
    *,
    config_path: Path,
    checkpoint: str | None,
    providers: list[str],
    split: str,
    threshold: float | None,
    recall_threshold: float,
    max_fail_ratio: float,
    dilate_px: int,
) -> dict:
    project_root = config_path.parents[1]
    base_path = config_path.parent / "base.yaml"
    config = load_app_config(config_path, base_path=base_path if base_path.exists() else None)
    version = str(config["project"].get("version", "seg_evidence"))
    checkpoint_path = (
        resolve_project_path(project_root, checkpoint)
        if checkpoint
        else project_root / "artifacts/runs/09_evidence_segmentation" / version / "checkpoints/best.pt"
    )
    lesion_threshold = (
        float(threshold)
        if threshold is not None
        else float(config.get("infer", {}).get("lesion_threshold", 0.5))
    )

    model = _load_model(config, project_root, checkpoint_path)
    device = next(model.parameters()).device
    transform = _eval_transform(config, project_root)
    data_cfg = config["data"]
    mask_preprocessor = (
        FundusPreprocess(
            output_size=int(data_cfg.get("image_size", 512)),
            **preprocess_kwargs_from_config(data_cfg),
        )
        if _eval_preprocessing_enabled(config, project_root)
        else None
    )

    provider_results: dict[str, dict] = {}
    all_pass = 0
    all_fail = 0
    for provider in providers:
        items = _provider_items(project_root, provider, split)
        per_image: list[dict] = []
        for image_path, masks in items:
            image = Image.open(image_path).convert("RGB")
            tensor = transform(image).unsqueeze(0).to(device)
            with torch.no_grad():
                logits = model(tensor)
                probs = torch.sigmoid(logits)[0].detach().cpu().numpy()
            _, h, w = probs.shape
            gt = _mask_dict_to_eval_tensor(
                masks,
                image=image,
                size=(w, h),
                preprocessor=mask_preprocessor,
            ).numpy()
            pred_union = _union_mask(probs >= lesion_threshold)
            gt_union = _union_mask(gt)
            gt_pixels = int(gt_union.sum())
            pred_pixels = int(pred_union.sum())
            tp = int(np.logical_and(pred_union, gt_union).sum())
            fn = int(np.logical_and(~pred_union, gt_union).sum())
            fp = int(np.logical_and(pred_union, ~gt_union).sum())
            recall = float(tp / gt_pixels) if gt_pixels else None
            fn_ratio = float(fn / gt_pixels) if gt_pixels else None
            sample_pass = bool(
                gt_pixels > 0
                and recall is not None
                and fn_ratio is not None
                and recall >= recall_threshold
                and fn_ratio <= (1.0 - recall_threshold)
            )
            per_image.append(
                {
                    "image_id": image_path.stem,
                    "gt_pixels": gt_pixels,
                    "pred_pixels": pred_pixels,
                    "true_positive_pixels": tp,
                    "false_negative_pixels": fn,
                    "false_positive_pixels": fp,
                    "lesion_recall": recall,
                    "false_negative_pixel_ratio": fn_ratio,
                    "dilate_px": dilate_px,
                    "dilated_gt_area_ratio": _dilated_area_ratio(gt_union, dilate_px),
                    "passes_a1_pre1": sample_pass,
                }
            )

        passed = sum(1 for row in per_image if row["passes_a1_pre1"])
        failed = len(per_image) - passed
        all_pass += passed
        all_fail += failed
        provider_results[provider] = {
            "split": split,
            "n_images": len(per_image),
            "passed": passed,
            "failed": failed,
            "fail_ratio": float(failed / len(per_image)) if per_image else None,
            "mean_lesion_recall": float(
                np.mean([row["lesion_recall"] for row in per_image if row["lesion_recall"] is not None])
            )
            if per_image
            else None,
            "mean_false_negative_pixel_ratio": float(
                np.mean(
                    [
                        row["false_negative_pixel_ratio"]
                        for row in per_image
                        if row["false_negative_pixel_ratio"] is not None
                    ]
                )
            )
            if per_image
            else None,
            "mean_dilated_gt_area_ratio": float(np.mean([row["dilated_gt_area_ratio"] for row in per_image]))
            if per_image
            else None,
            "per_image": per_image,
        }

    total = all_pass + all_fail
    overall_fail_ratio = float(all_fail / total) if total else None
    return {
        "version": version,
        "config_path": str(config_path),
        "checkpoint_path": str(checkpoint_path),
        "providers": providers,
        "split": split,
        "threshold": lesion_threshold,
        "recall_threshold": float(recall_threshold),
        "false_negative_pixel_ratio_threshold": float(1.0 - recall_threshold),
        "max_fail_ratio_for_a1": float(max_fail_ratio),
        "dilate_px": int(dilate_px),
        "overall": {
            "n_images": total,
            "passed": all_pass,
            "failed": all_fail,
            "fail_ratio": overall_fail_ratio,
            "a1_preconditions_pass": bool(
                overall_fail_ratio is not None and overall_fail_ratio < max_fail_ratio
            ),
        },
        "providers_result": provider_results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit v8b mask quality for A1 preconditions.")
    parser.add_argument("--config", default="configs/seg_evidence_v8b_ddrseg_tjdr_maplesfix.yaml")
    parser.add_argument("--checkpoint")
    parser.add_argument("--providers", nargs="+", default=["idrid", "maples"], choices=["idrid", "maples"])
    parser.add_argument("--split", default="test", choices=["train", "test"])
    parser.add_argument("--threshold", type=float)
    parser.add_argument("--recall-threshold", type=float, default=0.6)
    parser.add_argument("--max-fail-ratio", type=float, default=0.3)
    parser.add_argument("--dilate-px", type=int, default=7)
    parser.add_argument("--output", default=".omc/research/a1_v8b_mask_quality_audit.json")
    args = parser.parse_args()

    result = run_audit(
        config_path=Path(args.config).resolve(),
        checkpoint=args.checkpoint,
        providers=list(args.providers),
        split=str(args.split),
        threshold=args.threshold,
        recall_threshold=float(args.recall_threshold),
        max_fail_ratio=float(args.max_fail_ratio),
        dilate_px=int(args.dilate_px),
    )
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Saved: {output_path}")
    print(json.dumps(result["overall"], indent=2))
    for provider, payload in result["providers_result"].items():
        print(
            f"{provider}: fail_ratio={payload['fail_ratio']:.4f} "
            f"mean_recall={payload['mean_lesion_recall']:.4f}"
        )


if __name__ == "__main__":
    main()
