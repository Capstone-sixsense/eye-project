"""Screen color canonicalization candidates before changing preprocessing.

(한글 요약) 전처리 변경 전, 색 정규화 후보들을 저비용으로 비교하는 Phase 1 진단이다
(FundusPreprocess나 활성 아티팩트를 건드리지 않음).

This is a low-cost Phase 1 diagnostic for
``.omc/plans/preprocessing_color_canonicalization_plan.md``.  It intentionally
does not modify ``FundusPreprocess`` or any active artifacts.  Each candidate is
applied to raw images in the same order planned for Phase 2:

    raw image -> active geometry -> candidate color transform -> resize -> Ben Graham

The script reports:

- a domain probe over foreground RGB/LAB color moments;
- lesion color contrast preservation on IDRiD/MAPLES mask-valid images;
- P1-G2/P1-G3/P1-G4 gate decisions.
"""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from collections.abc import Callable

import cv2
import numpy as np
from PIL import Image as PILImage

from drscreen.cli.diagnose_shortcut_audit import (
    ImageItem,
    _fit_probe,
    _idrid_mask_items,
    _maples_mask_items,
    _read_manifest_items,
)
from drscreen.data.transforms import FundusPreprocess
from drscreen.settings import resolve_project_path
from drscreen.xai.iou import union_mask

ColorTransform = Callable[[np.ndarray, np.ndarray], np.ndarray]


@dataclass(frozen=True)
class VariantSpec:
    transform: ColorTransform
    stage: str
    description: str


def shades_of_gray(rgb: np.ndarray, foreground: np.ndarray, *, p: float = 6.0) -> np.ndarray:
    """Apply Minkowski p-norm shades-of-gray color constancy."""
    work = rgb.astype(np.float32)
    fg = foreground.astype(bool)
    if not fg.any():
        fg = np.ones(work.shape[:2], dtype=bool)

    pixels = work[fg]
    illuminant = np.power(np.mean(np.power(np.maximum(pixels, 1.0), p), axis=0), 1.0 / p)
    target = float(np.mean(illuminant))
    scale = target / np.maximum(illuminant, 1e-6)
    out = work * scale.reshape(1, 1, 3)
    return np.clip(out, 0, 255).astype(np.uint8)


def clahe_luminance(
    rgb: np.ndarray,
    _foreground: np.ndarray,
    *,
    clip: float = 2.0,
    grid: int = 8,
) -> np.ndarray:
    """Apply CLAHE to LAB luminance only, preserving a/b chroma channels."""
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    clahe = cv2.createCLAHE(clipLimit=float(clip), tileGridSize=(int(grid), int(grid)))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


def _identity(rgb: np.ndarray, _foreground: np.ndarray) -> np.ndarray:
    return rgb


def ben_graham_channel_standardize(
    rgb: np.ndarray,
    foreground: np.ndarray,
    *,
    target_mean: float = 128.0,
    target_std: float = 45.0,
) -> np.ndarray:
    """Standardize foreground RGB channel moments after Ben Graham normalization.

    This is the follow-up "안2" screen.  It is intentionally isolated to this
    diagnostic script because the plan only names the candidate, not a committed
    production formula.
    """
    work = rgb.astype(np.float32)
    fg = foreground.astype(bool)
    if not fg.any():
        fg = np.ones(work.shape[:2], dtype=bool)

    out = work.copy()
    for channel_index in range(3):
        channel = work[:, :, channel_index]
        values = channel[fg]
        mean = float(values.mean())
        std = float(values.std())
        if std <= 1e-6:
            continue
        out[:, :, channel_index] = (channel - mean) * (target_std / std) + target_mean
    out[~fg] = 0
    return np.clip(out, 0, 255).astype(np.uint8)


def _project_root() -> Path:
    cwd = Path.cwd()
    if (cwd / "drscreen").exists() and (cwd / "data").exists():
        return cwd
    if (cwd / "ai" / "drscreen").exists() and (cwd / "ai" / "data").exists():
        return cwd / "ai"
    return Path(__file__).resolve().parents[2]


def _make_preprocessor(args: argparse.Namespace) -> FundusPreprocess:
    return FundusPreprocess(
        output_size=None,
        preprocess_mode=args.preprocess_mode,
        crop_tol=args.crop_tol,
        ben_graham_weight=args.ben_graham_weight,
        ben_graham_offset=args.ben_graham_offset,
        target_short_fill=args.target_short_fill,
        max_total_x_trim=args.max_total_x_trim,
        max_total_y_trim=args.max_total_y_trim,
        saliency_shift=args.saliency_shift,
        saliency_weight=args.saliency_weight,
        saliency_candidates=args.saliency_candidates,
        safezoom_max_dim=args.safezoom_max_dim,
    )


def _foreground_mask(pre: FundusPreprocess, image: np.ndarray) -> np.ndarray:
    mask = pre._foreground_mask(image)
    if mask.any():
        return mask
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return gray > 7


def _process_raw_image(
    image_path: Path,
    *,
    pre: FundusPreprocess,
    image_size: int,
    variant: VariantSpec,
) -> np.ndarray:
    with PILImage.open(image_path) as image:
        arr = np.asarray(image.convert("RGB")).copy()
    arr = pre._content_crop(arr)
    foreground = _foreground_mask(pre, arr)
    if variant.stage == "pre_ben":
        arr = variant.transform(arr, foreground)
    arr = np.asarray(
        PILImage.fromarray(arr).resize((image_size, image_size), PILImage.BICUBIC)
    ).copy()
    arr = pre._ben_graham(arr)
    if variant.stage == "post_ben":
        foreground = _foreground_mask(pre, arr)
        arr = variant.transform(arr, foreground)
    return arr


def _color_moment_features(pre: FundusPreprocess, image: np.ndarray) -> np.ndarray:
    foreground = _foreground_mask(pre, image)
    if not foreground.any():
        foreground = np.ones(image.shape[:2], dtype=bool)

    rgb = image.astype(np.float32)
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB).astype(np.float32)
    features: list[float] = []
    for space in (rgb, lab):
        pixels = space[foreground]
        features.extend(np.mean(pixels, axis=0).tolist())
        features.extend(np.std(pixels, axis=0).tolist())
    return np.asarray(features, dtype=np.float32)


def _run_color_probe(
    items: list[ImageItem],
    *,
    pre: FundusPreprocess,
    image_size: int,
    variant: VariantSpec,
    seed: int,
) -> dict:
    domain_names = sorted({item.domain for item in items})
    domain_to_idx = {domain: idx for idx, domain in enumerate(domain_names)}
    features = np.stack(
        [
            _color_moment_features(
                pre,
                _process_raw_image(
                    item.path,
                    pre=pre,
                    image_size=image_size,
                    variant=variant,
                ),
            )
            for item in items
        ],
        axis=0,
    )
    y = np.asarray([domain_to_idx[item.domain] for item in items], dtype=np.int64)
    result = _fit_probe(features, y, seed=seed, class_names=domain_names)
    result.update(
        {
            "probe": "color_moment_domain_probe",
            "feature": "foreground_rgb_lab_mean_std",
            "n_features": int(features.shape[1]),
        }
    )
    return result


def _align_union_mask(
    item: ImageItem,
    *,
    pre: FundusPreprocess,
    image_size: int,
) -> np.ndarray | None:
    if not item.masks:
        return None
    lesion = union_mask(item.masks)
    if lesion is None or int(lesion.sum()) <= 0:
        return None
    with PILImage.open(item.path) as image:
        aligned = pre.apply_mask_geometry(lesion, image.convert("RGB"), output_size=image_size)
    if aligned.ndim == 3:
        aligned = aligned[..., 0]
    return aligned.astype(bool)


def _foreground_for_processed(pre: FundusPreprocess, image: np.ndarray) -> np.ndarray:
    mask = _foreground_mask(pre, image)
    if not mask.any():
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        mask = gray > 7
    return mask.astype(bool)


def _lesion_contrast(image: np.ndarray, lesion_mask: np.ndarray, foreground: np.ndarray) -> dict | None:
    lesion = lesion_mask.astype(bool) & foreground
    if int(lesion.sum()) < 5:
        return None

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    dilated = cv2.dilate(lesion.astype(np.uint8), kernel, iterations=1).astype(bool)
    ring = dilated & ~lesion & foreground
    if int(ring.sum()) < 20:
        return None

    rgb = image.astype(np.float32)
    red_axis = rgb[:, :, 0] - 0.5 * (rgb[:, :, 1] + rgb[:, :, 2])
    yellow_axis = 0.5 * (rgb[:, :, 0] + rgb[:, :, 1]) - rgb[:, :, 2]

    red_delta = float(abs(red_axis[lesion].mean() - red_axis[ring].mean()))
    yellow_delta = float(abs(yellow_axis[lesion].mean() - yellow_axis[ring].mean()))
    return {
        "red_delta": red_delta,
        "yellow_delta": yellow_delta,
        "combined_delta": max(red_delta, yellow_delta),
        "lesion_pixels": int(lesion.sum()),
        "ring_pixels": int(ring.sum()),
    }


def _summary(values: list[float]) -> dict:
    if not values:
        return {"n": 0, "median": None, "mean": None, "p10": None, "p90": None}
    arr = np.asarray(values, dtype=np.float64)
    return {
        "n": int(arr.size),
        "median": float(np.median(arr)),
        "mean": float(np.mean(arr)),
        "p10": float(np.percentile(arr, 10)),
        "p90": float(np.percentile(arr, 90)),
    }


def _run_lesion_color_audit(
    items: list[ImageItem],
    *,
    pre: FundusPreprocess,
    image_size: int,
    variants: dict[str, VariantSpec],
) -> dict:
    per_variant_ratios: dict[str, list[float]] = {name: [] for name in variants if name != "baseline"}
    per_variant_domain: dict[str, dict[str, list[float]]] = {
        name: defaultdict(list) for name in variants if name != "baseline"
    }
    skipped: dict[str, int] = defaultdict(int)
    used_items = 0

    for item in items:
        lesion_mask = _align_union_mask(item, pre=pre, image_size=image_size)
        if lesion_mask is None:
            skipped["empty_mask"] += 1
            continue

        baseline_img = _process_raw_image(
            item.path,
            pre=pre,
            image_size=image_size,
            variant=variants["baseline"],
        )
        foreground = _foreground_for_processed(pre, baseline_img)
        baseline_contrast = _lesion_contrast(baseline_img, lesion_mask, foreground)
        if baseline_contrast is None or baseline_contrast["combined_delta"] <= 1e-6:
            skipped["no_baseline_contrast"] += 1
            continue
        used_items += 1

        for name, variant in variants.items():
            if name == "baseline":
                continue
            candidate_img = _process_raw_image(
                item.path,
                pre=pre,
                image_size=image_size,
                variant=variant,
            )
            candidate_foreground = _foreground_for_processed(pre, candidate_img)
            candidate_contrast = _lesion_contrast(candidate_img, lesion_mask, candidate_foreground)
            if candidate_contrast is None:
                skipped[f"{name}_no_candidate_contrast"] += 1
                continue
            ratio = candidate_contrast["combined_delta"] / max(
                baseline_contrast["combined_delta"],
                1e-6,
            )
            per_variant_ratios[name].append(float(ratio))
            per_variant_domain[name][item.domain].append(float(ratio))

    return {
        "metric": "candidate_combined_red_or_yellow_delta_divided_by_baseline",
        "items_total": int(len(items)),
        "items_used": int(used_items),
        "skipped": dict(skipped),
        "variants": {
            name: {
                "overall": _summary(values),
                "by_domain": {
                    domain: _summary(domain_values)
                    for domain, domain_values in sorted(per_variant_domain[name].items())
                },
            }
            for name, values in per_variant_ratios.items()
        },
    }


def _select_candidate(results: dict, *, min_auc_drop: float, min_color_preservation: float) -> dict:
    baseline_auc = results["variants"]["baseline"]["domain_probe"].get("macro_ovr_auroc")
    selected: list[dict] = []
    for name, variant in results["variants"].items():
        if name == "baseline":
            continue
        auc = variant["domain_probe"].get("macro_ovr_auroc")
        color_median = (
            results["lesion_color_preservation"]["variants"]
            .get(name, {})
            .get("overall", {})
            .get("median")
        )
        auc_drop = None if auc is None or baseline_auc is None else float(baseline_auc - auc)
        p1_g2 = auc_drop is not None and auc_drop >= min_auc_drop
        p1_g3 = color_median is not None and color_median >= min_color_preservation
        variant["gates"] = {
            "P1_G2_domain_separability_reduced": p1_g2,
            "P1_G2_auc_drop": auc_drop,
            "P1_G3_lesion_color_preserved": p1_g3,
            "P1_G3_median_preservation_ratio": color_median,
        }
        if p1_g2 and p1_g3:
            selected.append(
                {
                    "name": name,
                    "auc_drop": auc_drop,
                    "median_preservation_ratio": color_median,
                }
            )

    selected.sort(key=lambda row: row["auc_drop"], reverse=True)
    fallback = (
        "STOP_PHASE2_NO_COLOR_CANONICALIZATION_CANDIDATE"
        if "ben_graham_channel_standardized" in results["variants"]
        else "STOP_PHASE2_RECORD_BEN_GRAHAM_CHANNEL_STANDARDIZATION_FOLLOWUP"
    )
    return {
        "P1_G4_pass": bool(selected),
        "selected_candidate": selected[0] if selected else None,
        "all_passing_candidates": selected,
        "fallback_recommendation": None if selected else fallback,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="data/processed/manifest_with_maples.csv")
    parser.add_argument("--output", default=".omc/research/preprocessing_color/color_screen_v1.json")
    parser.add_argument("--max-per-domain", type=int, default=120)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--preprocess-mode", default="circular")
    parser.add_argument("--crop-tol", type=int, default=7)
    parser.add_argument("--ben-graham-weight", type=float, default=4.0)
    parser.add_argument("--ben-graham-offset", type=float, default=128.0)
    parser.add_argument("--target-short-fill", type=float, default=0.86)
    parser.add_argument("--max-total-x-trim", type=float, default=0.08)
    parser.add_argument("--max-total-y-trim", type=float, default=0.08)
    parser.add_argument("--saliency-shift", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--saliency-weight", type=float, default=1.2)
    parser.add_argument("--saliency-candidates", type=int, default=5)
    parser.add_argument("--safezoom-max-dim", type=int, default=1024)
    parser.add_argument("--shades-p", type=float, default=6.0)
    parser.add_argument("--clahe-clip", type=float, default=2.0)
    parser.add_argument("--clahe-grid", type=int, default=8)
    parser.add_argument("--channel-standardize-mean", type=float, default=128.0)
    parser.add_argument("--channel-standardize-std", type=float, default=45.0)
    parser.add_argument("--min-auc-drop", type=float, default=0.05)
    parser.add_argument("--min-color-preservation", type=float, default=0.70)
    parser.add_argument("--lesion-max-items", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = _project_root()
    manifest_path = resolve_project_path(project_root, args.manifest)
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Raw MAPLES-inclusive manifest not found: {manifest_path}. "
            "Do not substitute manifest_preprocessed.csv for this screen."
        )

    pre = _make_preprocessor(args)
    variants: dict[str, VariantSpec] = {
        "baseline": VariantSpec(
            transform=_identity,
            stage="pre_ben",
            description="Current circular geometry + resize + Ben Graham baseline.",
        ),
        "shades_of_gray": VariantSpec(
            transform=lambda rgb, fg: shades_of_gray(rgb, fg, p=args.shades_p),
            stage="pre_ben",
            description="Minkowski p-norm color constancy before resize and Ben Graham.",
        ),
        "clahe_l": VariantSpec(
            transform=lambda rgb, fg: clahe_luminance(
                rgb,
                fg,
                clip=args.clahe_clip,
                grid=args.clahe_grid,
            ),
            stage="pre_ben",
            description="LAB luminance CLAHE before resize and Ben Graham.",
        ),
        "ben_graham_channel_standardized": VariantSpec(
            transform=lambda rgb, fg: ben_graham_channel_standardize(
                rgb,
                fg,
                target_mean=args.channel_standardize_mean,
                target_std=args.channel_standardize_std,
            ),
            stage="post_ben",
            description=(
                "Follow-up option 2: foreground RGB channel mean/std standardization "
                "after Ben Graham output."
            ),
        ),
    }

    items = _read_manifest_items(
        project_root,
        manifest_path,
        max_per_domain=args.max_per_domain,
        seed=args.seed,
    )
    domain_counts: dict[str, int] = defaultdict(int)
    for item in items:
        domain_counts[item.domain] += 1

    results = {
        "script": "drscreen.cli.diagnose_color_canonicalization",
        "project_root": str(project_root),
        "manifest_path": str(manifest_path),
        "parameters": {
            key: value
            for key, value in vars(args).items()
            if key not in {"output"}
        },
        "sample": {
            "n_items": int(len(items)),
            "domain_counts": dict(sorted(domain_counts.items())),
        },
        "pipeline_order": [
            "raw image",
            f"active geometry ({args.preprocess_mode})",
            "candidate pre-Ben color_norm when stage=pre_ben",
            f"resize({args.image_size})",
            "Ben Graham",
            "candidate channel standardization when stage=post_ben",
        ],
        "variants": {
            name: {
                "stage": variant.stage,
                "description": variant.description,
            }
            for name, variant in variants.items()
        },
    }

    for name, variant in variants.items():
        results["variants"][name].update(
            {
            "domain_probe": _run_color_probe(
                items,
                pre=pre,
                image_size=args.image_size,
                variant=variant,
                seed=args.seed,
            )
            }
        )

    lesion_items = (
        _idrid_mask_items(project_root, "train")
        + _idrid_mask_items(project_root, "test")
        + _maples_mask_items(project_root, "train")
        + _maples_mask_items(project_root, "test")
    )
    if args.lesion_max_items and len(lesion_items) > args.lesion_max_items:
        rng = random.Random(args.seed)
        rng.shuffle(lesion_items)
        lesion_items = lesion_items[: args.lesion_max_items]

    lesion_domain_counts: dict[str, int] = defaultdict(int)
    for item in lesion_items:
        lesion_domain_counts[item.domain] += 1

    results["lesion_color_preservation"] = {
        "sample": {
            "n_items": int(len(lesion_items)),
            "domain_counts": dict(sorted(lesion_domain_counts.items())),
        },
        **_run_lesion_color_audit(
            lesion_items,
            pre=pre,
            image_size=args.image_size,
            variants=variants,
        ),
    }
    results["gate_decision"] = _select_candidate(
        results,
        min_auc_drop=args.min_auc_drop,
        min_color_preservation=args.min_color_preservation,
    )

    output_path = resolve_project_path(project_root, args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2, ensure_ascii=False)
        handle.write("\n")

    print(json.dumps(results["gate_decision"], indent=2, ensure_ascii=False))
    print(f"wrote {output_path}")


if __name__ == "__main__":
    main()
