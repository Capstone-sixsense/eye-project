from __future__ import annotations

import argparse
from pathlib import Path

from drscreen.settings import load_app_config
from drscreen.ssl.trainer import run_ssl_pretraining


def main() -> None:
    parser = argparse.ArgumentParser(
        description="SimCLR SSL pretraining for RETFound backbone"
    )
    parser.add_argument("--config", required=True, help="Path to SSL config YAML")
    parser.add_argument(
        "--base-config",
        default=None,
        help="Optional base config YAML to merge under the SSL config",
    )
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    project_root = config_path.parents[1]
    base_path = Path(args.base_config).resolve() if args.base_config else None

    config = load_app_config(config_path, base_path=base_path)
    summary = run_ssl_pretraining(config, config_path=config_path, project_root=project_root)
    print(f"SSL pretraining complete. Encoder saved to: {summary['output_path']}")


if __name__ == "__main__":
    main()
