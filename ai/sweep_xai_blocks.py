"""Block sweep: run XAI IoU evaluation at each target block and compare.

For each block index (and the default last-block baseline), runs eval_xai_iou
and collects aggregate metrics. Prints a comparison table and saves a JSON summary.

Diagnostic interpretation:
  - block2/3 IoU high, last-block low → classifier/pooling shortcut
  - all blocks low                    → backbone lacks spatial lesion signal

Usage
-----
    # default sweep: last + blocks 2,3,4,5,6
    python sweep_xai_blocks.py --config configs/v24_multitask.yaml --split test

    # custom block range
    python sweep_xai_blocks.py --config configs/v24_multitask.yaml --split test --blocks 1 2 3 4 5 6 7

    # include baseline comparison on the last-block run
    python sweep_xai_blocks.py --config configs/v24_multitask.yaml --split test --baselines

Output
------
    artifacts/runs/{primary_group}/{version}/evaluations/block_sweep_{version}_{split}.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from drscreen.settings import get_run_evaluation_dir
from drscreen.xai.evaluation import evaluate


_DEFAULT_BLOCKS = [2, 3, 4, 5, 6]


def _fmt(v: float | None, width: int = 7) -> str:
    return f"{v:.4f}".rjust(width) if v is not None else "   N/A".rjust(width)


def _extract_metrics(result: dict, top_percents: list[float]) -> dict:
    agg = result["aggregate"]
    row: dict = {
        "pointing_game": agg["pointing_game"]["mean"] if agg["pointing_game"] else None,
        "auprc": agg["auprc"]["mean"] if agg["auprc"] else None,
        "auc_iou": agg["auc_iou"]["mean"] if agg["auc_iou"] else None,
    }
    for top_pct in top_percents:
        key = f"top{int(top_pct * 100):02d}"
        t = agg["thresholds"].get(key, {})
        row[f"iou_{key}"] = t.get("mean_iou_union")
    return row


def _print_table(rows: list[dict], top_percents: list[float]) -> None:
    iou_cols = [f"iou_top{int(p*100):02d}" for p in top_percents]
    header_parts = [f"{'Block':<12}", f"{'PG':>7}", f"{'AUPRC':>7}", f"{'AUC-IoU':>8}"]
    for col in iou_cols:
        pct = col.replace("iou_top", "top") + "%"
        header_parts.append(f"{pct:>9}")
    header = "  ".join(header_parts)
    print(header)
    print("-" * len(header))
    for row in rows:
        label = row["block_label"]
        pg = _fmt(row.get("pointing_game"), 7)
        ap = _fmt(row.get("auprc"), 7)
        ai = _fmt(row.get("auc_iou"), 8)
        iou_parts = [_fmt(row.get(col), 9) for col in iou_cols]
        parts = [f"{label:<12}", pg, ap, ai] + iou_parts
        print("  ".join(parts))


def sweep(
    config_path: str,
    blocks: list[int],
    split: str = "test",
    idrid_root: str | None = None,
    top_percents: list[float] | None = None,
    run_baselines: bool = False,
) -> dict:
    if top_percents is None:
        top_percents = [0.10, 0.20, 0.30]

    project_root = Path(config_path).resolve().parents[1]

    print(f"Config : {config_path}")
    print(f"Split  : {split}")
    print(f"Blocks : last + {blocks}")
    print()

    rows: list[dict] = []
    all_results: dict[str, dict] = {}

    # Run "last block" (default) first — also where baselines run if requested
    targets: list[tuple[str, int | None]] = [("last", None)] + [
        (f"block{b}", b) for b in blocks
    ]

    for label, block_idx in targets:
        print(f"{'='*50}")
        print(f"Running: {label}")
        print(f"{'='*50}")
        result = evaluate(
            config_path=config_path,
            split=split,
            idrid_root=idrid_root,
            top_percents=top_percents,
            target_block=block_idx,
            run_baselines=(run_baselines and label == "last"),
        )
        metrics = _extract_metrics(result, top_percents)
        rows.append({"block_label": label, **metrics})
        all_results[label] = result
        print()

    print()
    print("=" * 60)
    print("BLOCK SWEEP SUMMARY")
    print("=" * 60)
    _print_table(rows, top_percents)

    # Highlight best IoU top20 block
    key20 = "iou_top20"
    best_row = max(rows, key=lambda r: r.get(key20) or 0.0)
    print(f"\nBest IoU-top20: {best_row['block_label']} = {_fmt(best_row.get(key20))}")

    version = all_results["last"]["version"]
    output = {
        "version": version,
        "config_path": config_path,
        "split": split,
        "blocks_swept": [label for label, _ in targets],
        "top_percents": top_percents,
        "summary": rows,
        "results": all_results,
    }

    eval_dir = get_run_evaluation_dir(project_root, str(version))
    eval_dir.mkdir(parents=True, exist_ok=True)
    output_path = eval_dir / f"block_sweep_{version}_{split}.json"
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"\nSaved: {output_path}")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="XAI block sweep: compare IoU across target layers")
    parser.add_argument("--config", required=True, help="Path to model config YAML")
    parser.add_argument(
        "--split", default="test", choices=["train", "test"],
        help="IDRiD split (default: test)"
    )
    parser.add_argument("--idrid-root", help="Override IDRiD root directory")
    parser.add_argument(
        "--blocks", nargs="+", type=int, default=_DEFAULT_BLOCKS,
        help=f"Block indices to sweep (default: {_DEFAULT_BLOCKS})"
    )
    parser.add_argument(
        "--top-percents", nargs="+", type=float, default=[0.10, 0.20, 0.30],
        help="CAM binarization thresholds (default: 0.10 0.20 0.30)"
    )
    parser.add_argument(
        "--baselines", action="store_true",
        help="Include random/center-Gaussian/retina-uniform baselines on the last-block run"
    )
    args = parser.parse_args()

    sweep(
        config_path=args.config,
        blocks=args.blocks,
        split=args.split,
        idrid_root=args.idrid_root,
        top_percents=args.top_percents,
        run_baselines=args.baselines,
    )


if __name__ == "__main__":
    main()
