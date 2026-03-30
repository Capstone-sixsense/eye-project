from __future__ import annotations

import argparse
from pathlib import Path
from pprint import pprint

from drscreen.settings import ensure_runtime_directories, load_app_config
from drscreen.train.runner import describe_training_setup, run_training


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fundus DR AI trainer.")
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    parser.add_argument("--dry-run", action="store_true", help="Validate config and paths only.")
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

    setup = describe_training_setup(config, config_path=config_path, project_root=project_root)

    print("Training setup")
    pprint(setup)
    pprint(config)

    if args.dry_run:
        return

    summary = run_training(config, config_path=config_path, project_root=project_root)
    print("Training complete")
    pprint(summary)


if __name__ == "__main__":
    main()
