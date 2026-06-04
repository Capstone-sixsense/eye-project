"""독립 병변 분할기 학습 CLI 진입점(분류기 train.py와 별개).

config(+ base.yaml)를 병합해 seg_runner.run_segmentation_training을 실행한다.
분류 학습과 동일하게 Python 3.14를 강제한다. --dry-run은 데이터/모델 설정만 검증한다.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from pprint import pprint

_REQUIRED_TRAINING_PYTHON = (3, 14)


def _enforce_training_python() -> None:
    # 학습 전용 Python 버전 가드(배포 인터프리터로 실수 실행 방지).
    if sys.version_info[:2] == _REQUIRED_TRAINING_PYTHON:
        return
    required = ".".join(str(part) for part in _REQUIRED_TRAINING_PYTHON)
    current = f"{sys.version_info.major}.{sys.version_info.minor}"
    raise RuntimeError(
        "Segmentation training must run on Python "
        f"{required}; current interpreter is Python {current}: {sys.executable}. "
        f"Use `py -{required} -m drscreen.cli.train_seg --config <config>`."
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Standalone lesion evidence segmenter trainer.")
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    parser.add_argument("--dry-run", action="store_true", help="Validate data/model setup only.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _enforce_training_python()

    from drscreen.settings import ensure_runtime_directories, load_app_config
    from drscreen.train.seg_runner import (
        describe_segmentation_setup,
        run_segmentation_training,
    )

    config_path = Path(args.config).resolve()
    project_root = config_path.parents[1]
    base_path = None
    candidate_base = config_path.parent / "base.yaml"
    if config_path.name != "base.yaml" and candidate_base.exists():
        base_path = candidate_base
    config = load_app_config(config_path, base_path=base_path)
    ensure_runtime_directories(config, project_root)

    setup = describe_segmentation_setup(
        config,
        config_path=config_path,
        project_root=project_root,
    )
    print("Segmentation training setup")
    pprint(setup)
    if args.dry_run:
        return

    summary = run_segmentation_training(
        config,
        config_path=config_path,
        project_root=project_root,
    )
    print("Segmentation training complete")
    pprint(summary)


if __name__ == "__main__":
    main()
