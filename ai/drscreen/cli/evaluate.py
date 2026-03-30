from __future__ import annotations

import argparse
from pathlib import Path
from pprint import pprint

from drscreen.settings import ensure_runtime_directories, load_app_config
from drscreen.train.runner import run_split_evaluation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fundus DR AI split evaluator.")
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    parser.add_argument("--split", help="Split name to evaluate. Defaults to config.data.test_split.")
    parser.add_argument(
        "--checkpoint",
        help="Checkpoint path. Defaults to config.infer.checkpoint_path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).resolve()
    project_root = config_path.parents[1]
    base_path = None
    candidate_base = config_path.parent / "base.yaml"
    if config_path.name != "base.yaml" and candidate_base.exists():
        base_path = candidate_base

    config = load_app_config(config_path, base_path=base_path)
    ensure_runtime_directories(config, project_root)

    summary = run_split_evaluation(
        config,
        config_path=config_path,
        project_root=project_root,
        split_name=args.split,
        checkpoint_path=Path(args.checkpoint).resolve() if args.checkpoint else None,
    )
    print("Evaluation complete")
    pprint(summary)


if __name__ == "__main__":
    main()
