"""분류기 학습 CLI 진입점.

config(+ 같은 폴더 base.yaml)를 병합해 학습을 실행한다. --dry-run은 config/경로만 검증한다.
학습은 Python 3.14를 강제한다(배포/런타임 인터프리터로 실수 실행하는 것을 막기 위함).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from pprint import pprint

_REQUIRED_TRAINING_PYTHON = (3, 14)


def _enforce_training_python() -> None:
    # 학습 전용 Python 버전 가드. 다른 버전이면 명확한 안내와 함께 중단한다.
    if sys.version_info[:2] == _REQUIRED_TRAINING_PYTHON:
        return
    required = ".".join(str(part) for part in _REQUIRED_TRAINING_PYTHON)
    current = f"{sys.version_info.major}.{sys.version_info.minor}"
    executable = sys.executable
    raise RuntimeError(
        "Training must run on Python "
        f"{required}; current interpreter is Python {current}: {executable}. "
        f"Use `py -{required} -m drscreen.cli.train --config <config>`."
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fundus DR AI trainer.")
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    parser.add_argument("--dry-run", action="store_true", help="Validate config and paths only.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _enforce_training_python()

    from drscreen.settings import ensure_runtime_directories, load_app_config
    from drscreen.train.runner import describe_training_setup, run_training

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
