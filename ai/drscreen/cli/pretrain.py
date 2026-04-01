from __future__ import annotations

import argparse
from pathlib import Path
from pprint import pprint

from drscreen.settings import load_app_config
from drscreen.ssl.runner import run_ssl_pretraining
from drscreen.utils.logging import get_logger


LOGGER = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SimCLR SSL pretraining for drscreen backbone.")
    parser.add_argument("--config", required=True, help="Path to SSL config YAML.")
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

    if args.dry_run:
        pprint(config)
        return

    summary = run_ssl_pretraining(config, config_path=config_path, project_root=project_root)
    LOGGER.info(
        "SSL pretraining complete. best_loss=%.4f backbone=%s",
        summary["best_loss"],
        summary["backbone_best_path"],
    )
    pprint({k: v for k, v in summary.items() if k != "history"})


if __name__ == "__main__":
    main()
