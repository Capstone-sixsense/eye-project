"""XAI quantitative validation: Layer-CAM vs IDRiD lesion masks.

Generates Layer-CAM (or Grad-CAM) for each IDRiD segmentation image,
normalizes the heatmap within the retina FOV (Choe et al., CVPR 2020),
binarizes at the top-N% threshold, and computes:
  - IoU (top-N% binarization) against GT lesion masks (MA / HE / EX / SE)
  - Pixel AUPRC (continuous score vs binary GT mask)
  - AUC-IoU (mean IoU across full threshold sweep)
  - Pointing Game score

Usage
-----
    python eval_xai_iou.py --config configs/v24_multitask.yaml --split test

    # with baseline comparison (random / center Gaussian / retina uniform)
    python eval_xai_iou.py --config configs/v24_multitask.yaml --split test --baselines

    # use auxiliary seg_head output instead of Layer-CAM
    python eval_xai_iou.py --config configs/v24_multitask.yaml --split test --use-seg-head

Output
------
    artifacts/runs/{primary_group}/{version}/evaluations/xai_iou_{version}_{block_label}_{split}.json
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from drscreen.xai.evaluation import compare_xai_methods, evaluate


def main() -> None:
    parser = argparse.ArgumentParser(description="XAI IoU evaluation against IDRiD lesion masks")
    parser.add_argument("--config", required=True, help="Path to model config YAML")
    parser.add_argument(
        "--split", default="train", choices=["train", "test"],
        help="IDRiD segmentation split (default: train)"
    )
    parser.add_argument("--idrid-root", help="Override IDRiD root directory")
    parser.add_argument(
        "--top-percents", nargs="+", type=float, default=[0.10, 0.20, 0.30],
        help="Top-N%% thresholds for CAM binarization (default: 0.10 0.20 0.30)"
    )
    parser.add_argument(
        "--target-block", type=int, default=None,
        help="Index into model.blocks for CAM target layer (default: last block)."
    )
    parser.add_argument("--output", help="Output JSON path")
    parser.add_argument(
        "--use-seg-head", action="store_true",
        help="Use auxiliary seg_head sigmoid output as heatmap instead of Layer-CAM"
    )
    parser.add_argument(
        "--use-mil-attention", action="store_true",
        help="Use MIL attention map as heatmap instead of Grad-CAM"
    )
    parser.add_argument(
        "--baselines", action="store_true",
        help="Also evaluate random / center-Gaussian / retina-uniform baseline heatmaps"
    )
    parser.add_argument(
        "--gate-sigma", type=float, default=2.0,
        help="Phase-0 gate threshold multiplier: AUC-IoU > center_gaussian + N*sigma (default: 2.0)"
    )
    parser.add_argument(
        "--methods", nargs="+",
        choices=["gradcam", "layercam", "gradcam++", "scorecam", "ig"],
        help="Compare multiple XAI methods side by side (overrides single-method mode)"
    )
    args = parser.parse_args()

    if args.methods:
        compare_xai_methods(
            config_path=args.config,
            methods=args.methods,
            split=args.split,
            idrid_root=args.idrid_root,
            top_percents=args.top_percents,
            run_baselines=args.baselines,
            output_path=args.output,
        )
    else:
        evaluate(
            config_path=args.config,
            split=args.split,
            idrid_root=args.idrid_root,
            top_percents=args.top_percents,
            target_block=args.target_block,
            output_path=args.output,
            use_seg_head=args.use_seg_head,
            use_mil_attention=args.use_mil_attention,
            run_baselines=args.baselines,
            gate_sigma=args.gate_sigma,
        )


if __name__ == "__main__":
    main()
