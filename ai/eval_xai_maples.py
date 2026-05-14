"""XAI quantitative validation: Layer-CAM vs MAPLES-DR lesion masks on MESSIDOR images.

Generates Layer-CAM for each MESSIDOR image in the MAPLES-DR split,
and computes AUPRC / AUC-IoU / IoU top-N% / Pointing Game against
MAPLES-DR pixel-level lesion annotations (MA / HE / EX / CWS).

Usage
-----
    python eval_xai_maples.py --config configs/v31_no_se_gated.yaml --split test
    python eval_xai_maples.py --config configs/v35_warmstart_routing.yaml --split test
    python eval_xai_maples.py --config configs/v31_no_se_gated.yaml --split test --baselines

Output
------
    artifacts/runs/{primary_group}/{version}/evaluations/xai_maples_{version}_{block_label}_{split}.json
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from drscreen.xai.evaluation import evaluate_maples


def main() -> None:
    parser = argparse.ArgumentParser(
        description="XAI IoU evaluation against MAPLES-DR lesion masks"
    )
    parser.add_argument("--config", required=True, help="Path to model config YAML")
    parser.add_argument(
        "--split", default="test", choices=["train", "test"],
        help="MAPLES-DR split (default: test)"
    )
    parser.add_argument("--maples-root", help="Override MAPLES-DR AdditionalData directory")
    parser.add_argument("--messidor-images-dir", help="Override MESSIDOR images directory")
    parser.add_argument(
        "--top-percents", nargs="+", type=float, default=[0.10, 0.20, 0.30],
        help="Top-N%% thresholds for CAM binarization (default: 0.10 0.20 0.30)"
    )
    parser.add_argument(
        "--target-block", type=int, default=None,
        help="Index into model.blocks for CAM target layer (default: last block)"
    )
    parser.add_argument("--output", help="Output JSON path")
    parser.add_argument(
        "--baselines", action="store_true",
        help="Also evaluate random / center-Gaussian / retina-uniform baseline heatmaps"
    )
    parser.add_argument(
        "--gate-sigma", type=float, default=2.0,
        help="Phase-0 gate threshold multiplier (default: 2.0)"
    )
    args = parser.parse_args()

    evaluate_maples(
        config_path=args.config,
        split=args.split,
        maples_root=args.maples_root,
        messidor_images_dir=args.messidor_images_dir,
        top_percents=args.top_percents,
        target_block=args.target_block,
        output_path=args.output,
        run_baselines=args.baselines,
        gate_sigma=args.gate_sigma,
    )


if __name__ == "__main__":
    main()
