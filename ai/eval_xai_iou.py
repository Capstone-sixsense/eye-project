"""XAI quantitative validation: Layer-CAM / HiResCAM / Eigen-CAM vs lesion masks.

(한글 요약) IDRiD 마스크 대비 CAM 위치정확도를 정량 평가하는 CLI(drscreen.xai.evaluation 래퍼).
방법/타깃 블록/베이스라인/Phase-0 게이트를 옵션으로 지정한다.

Generates a CAM for each evaluation image, normalizes the heatmap within the
retina FOV (Choe et al., CVPR 2020), binarizes at the top-N% threshold, and
computes:
  - IoU (top-N% binarization) against GT lesion masks (MA / HE / EX / SE)
  - Pixel AUPRC (continuous score vs binary GT mask)
  - AUC-IoU (mean IoU across full threshold sweep)
  - Pointing Game score

Phase 4-A additions
-------------------
- ``--mask-provider {idrid,maples}``  : route to IDRiD or MAPLES-DR lesion masks.
- ``--target-blocks N [N ...]``       : multi-block CAM fusion across blocks.
- ``--block-weights w [w ...]``       : per-block weights for fusion (default uniform).
- ``--method``                        : single XAI method (overrides config).
- ``--cam-postprocess {none,morph,guided}``
- ``--tta {none,flip4,rot8}``         : test-time augmentation averaging.

Usage
-----
    # baseline reproduction
    python eval_xai_iou.py --config configs/base.yaml --split test --target-block 4

    # multi-block fusion on IDRiD
    python eval_xai_iou.py --config configs/base.yaml --split test \\
        --target-blocks 2 3 4 --method layercam --cam-postprocess guided --tta flip4

    # MAPLES-DR clean-cohort eval
    python eval_xai_iou.py --config configs/base.yaml --split test \\
        --mask-provider maples --target-blocks 2 3 4 --method hirescam

    # 5-method comparison (single block, MAPLES not supported in compare mode)
    python eval_xai_iou.py --config configs/base.yaml --split test \\
        --methods gradcam layercam hirescam gradcam++ eigencam

Output
------
    artifacts/runs/{primary_group}/{version}/evaluations/xai_iou_{version}_{method}_{block_label}_{split}[_suffix].json
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from drscreen.xai.evaluation import (
    compare_xai_methods,
    evaluate,
    evaluate_maples,
)

METHOD_CHOICES = [
    "gradcam",
    "layercam",
    "hirescam",
    "gradcam++",
    "scorecam",
    "eigencam",
    "ig",
    "occlusion",
    "rise",
    "bagnet",
]
PERTURBATION_METHODS = {"occlusion", "rise"}
DIRECT_EVIDENCE_METHODS = {"bagnet"}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="XAI IoU evaluation against IDRiD or MAPLES-DR lesion masks"
    )
    parser.add_argument("--config", required=True, help="Path to model config YAML")
    parser.add_argument(
        "--mask-provider", default="idrid", choices=["idrid", "maples"],
        help="Which lesion mask dataset to evaluate against (default: idrid)"
    )
    parser.add_argument(
        "--split", default="train", choices=["train", "test"],
        help="Segmentation split (default: train for idrid, test recommended for maples)"
    )
    parser.add_argument("--idrid-root", help="Override IDRiD root directory")
    parser.add_argument("--maples-root", help="Override MAPLES-DR AdditionalData directory")
    parser.add_argument(
        "--messidor-images-dir",
        help="Override MESSIDOR images directory (used with --mask-provider maples)",
    )
    parser.add_argument(
        "--top-percents", nargs="+", type=float, default=[0.10, 0.20, 0.30],
        help="Top-N%% thresholds for CAM binarization (default: 0.10 0.20 0.30)"
    )
    parser.add_argument(
        "--target-block", type=int, default=None,
        help="Single block index into model.blocks for CAM target layer "
             "(legacy; prefer --target-blocks)."
    )
    parser.add_argument(
        "--target-blocks", nargs="+", type=int, default=None,
        help="One or more block indices for CAM. When more than one is given, "
             "per-block CAMs are normalized then weighted-summed (multi-block fusion). "
             "Not supported with --methods comparison mode."
    )
    parser.add_argument(
        "--block-weights", nargs="+", type=float, default=None,
        help="Weights for --target-blocks fusion (default: uniform). "
             "Must match --target-blocks length."
    )
    parser.add_argument(
        "--method", default=None,
        choices=METHOD_CHOICES,
        help="Single XAI method (overrides config infer.gradcam_method). "
             "Use --methods for comparison mode."
    )
    parser.add_argument(
        "--grid-size", type=int, default=16,
        help="Occlusion grid size per axis. 16 means 16x16 = 256 masked forwards."
    )
    parser.add_argument(
        "--occlusion-batch-size", type=int, default=16,
        help="Batch size for occlusion masked forwards."
    )
    parser.add_argument(
        "--occlusion-patch-value", type=float, default=0.0,
        help="Replacement value in normalized tensor space for occlusion patches."
    )
    parser.add_argument(
        "--rise-num-masks", type=int, default=4000,
        help="Number of random masks for RISE."
    )
    parser.add_argument(
        "--rise-mask-resolution", type=int, default=7,
        help="Low-resolution mask side length for RISE."
    )
    parser.add_argument(
        "--rise-keep-prob", type=float, default=0.5,
        help="Mask keep probability for RISE."
    )
    parser.add_argument(
        "--rise-batch-size", type=int, default=32,
        help="Batch size for RISE masked forwards."
    )
    parser.add_argument(
        "--rise-seed", type=int, default=0,
        help="Random seed for RISE mask generation."
    )
    parser.add_argument(
        "--cam-postprocess", default="none", choices=["none", "morph", "guided"],
        help="Post-process the CAM after extraction. "
             "'morph' = morphological closing (15x15). "
             "'guided' = guided filter (radius=8, eps=1e-2) using RGB as guide."
    )
    parser.add_argument(
        "--tta", default="none", choices=["none", "flip4", "rot8"],
        help="Test-time augmentation: average CAMs over flip4 (I/hflip/vflip/hvflip) "
             "or rot8 (4 rotations x flip2)."
    )
    parser.add_argument("--output", help="Output JSON path")
    parser.add_argument(
        "--use-seg-head", action="store_true",
        help="Use auxiliary seg_head sigmoid output as heatmap instead of CAM"
    )
    parser.add_argument(
        "--use-mil-attention", action="store_true",
        help="Use MIL attention map as heatmap instead of CAM"
    )
    parser.add_argument(
        "--baselines", action="store_true",
        help="Also evaluate random / center-Gaussian / retina-uniform baseline heatmaps"
    )
    parser.add_argument(
        "--gate-sigma", type=float, default=2.0,
        help="Phase-0 gate threshold multiplier: AUC-IoU > center_gaussian + N*sigma "
             "(default: 2.0; use 1.0 for progressive milestone)"
    )
    parser.add_argument(
        "--methods", nargs="+",
        choices=METHOD_CHOICES,
        help="Compare multiple XAI methods side by side (single-block only). "
             "Mutually exclusive with --target-blocks (multi-block fusion)."
    )
    parser.add_argument(
        "--add-faithfulness", action="store_true",
        help="Compute Petsiuk-style deletion/insertion AUC for each attribution."
    )
    parser.add_argument(
        "--faithfulness-steps", type=int, default=100,
        help="Number of deletion/insertion steps when --add-faithfulness is set."
    )
    parser.add_argument(
        "--mask-optic-disc", action="store_true",
        help="Zero out optic disc region in CAM before evaluation (anatomy-guided masking)"
    )
    parser.add_argument(
        "--od-dilation", type=int, default=0,
        help="Pixels to dilate the optic disc mask (default: 0)"
    )
    args = parser.parse_args()

    active_method = args.method.lower() if args.method else None
    is_perturbation = active_method in PERTURBATION_METHODS
    is_direct_evidence = active_method in DIRECT_EVIDENCE_METHODS

    if args.block_weights is not None:
        if args.target_blocks is None or len(args.block_weights) != len(args.target_blocks):
            parser.error(
                "--block-weights length must match --target-blocks "
                f"(got {len(args.block_weights)} weights for "
                f"{0 if args.target_blocks is None else len(args.target_blocks)} blocks)"
            )

    if args.methods:
        if any(m in PERTURBATION_METHODS | DIRECT_EVIDENCE_METHODS for m in args.methods):
            parser.error("--methods comparison mode does not support occlusion/rise/bagnet")
        if args.use_seg_head:
            parser.error("--methods comparison mode cannot be combined with --use-seg-head")
        if args.use_mil_attention:
            parser.error("--methods comparison mode cannot be combined with --use-mil-attention")
        if args.mask_provider != "idrid":
            parser.error("--methods comparison mode supports --mask-provider idrid only")
        if args.target_blocks and len(args.target_blocks) > 1:
            parser.error(
                "--methods comparison mode does not support multi-block fusion; "
                "use a single --target-block instead"
            )
        compare_xai_methods(
            config_path=args.config,
            methods=args.methods,
            split=args.split,
            idrid_root=args.idrid_root,
            top_percents=args.top_percents,
            run_baselines=args.baselines,
            output_path=args.output,
        )
        return

    if is_perturbation:
        if args.target_block is not None or args.target_blocks is not None:
            parser.error("--method occlusion/rise cannot be combined with target blocks")
        if args.tta != "none":
            parser.error("--method occlusion/rise cannot be combined with --tta")
        if args.use_seg_head:
            parser.error("--method occlusion/rise cannot be combined with --use-seg-head")
        if args.use_mil_attention:
            parser.error("--method occlusion/rise cannot be combined with --use-mil-attention")

    if is_direct_evidence:
        if args.target_block is not None or args.target_blocks is not None:
            parser.error("--method bagnet cannot be combined with target blocks")
        if args.tta != "none":
            parser.error("--method bagnet cannot be combined with --tta")
        if args.use_seg_head:
            parser.error("--method bagnet cannot be combined with --use-seg-head")
        if args.use_mil_attention:
            parser.error("--method bagnet cannot be combined with --use-mil-attention")

    perturbation_options = {
        "grid_size": args.grid_size,
        "patch_value": args.occlusion_patch_value,
        "batch_size": args.occlusion_batch_size,
        "rise_num_masks": args.rise_num_masks,
        "rise_mask_resolution": args.rise_mask_resolution,
        "rise_keep_prob": args.rise_keep_prob,
        "rise_batch_size": args.rise_batch_size,
        "rise_seed": args.rise_seed,
    }

    common_kwargs = dict(
        config_path=args.config,
        split=args.split,
        top_percents=args.top_percents,
        target_block=args.target_block,
        target_blocks=args.target_blocks,
        target_layer_weights=args.block_weights,
        method_override=args.method,
        postprocess=args.cam_postprocess,
        tta=args.tta,
        output_path=args.output,
        run_baselines=args.baselines,
        gate_sigma=args.gate_sigma,
        mask_optic_disc=args.mask_optic_disc,
        od_dilation_px=args.od_dilation,
        perturbation_options=perturbation_options,
        add_faithfulness=args.add_faithfulness,
        faithfulness_steps=args.faithfulness_steps,
    )

    if args.mask_provider == "maples":
        evaluate_maples(
            maples_root=args.maples_root,
            messidor_images_dir=args.messidor_images_dir,
            use_seg_head=args.use_seg_head,
            **common_kwargs,
        )
    else:
        evaluate(
            idrid_root=args.idrid_root,
            use_seg_head=args.use_seg_head,
            use_mil_attention=args.use_mil_attention,
            **common_kwargs,
        )


if __name__ == "__main__":
    main()
