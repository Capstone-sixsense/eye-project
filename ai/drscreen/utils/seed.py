"""재현성(reproducibility) 유틸: 시드 고정, 결정성 설정, 환경 스냅샷.

- set_seed: random/numpy/torch(+CUDA) 시드를 한 번에 고정.
- configure_determinism: cuDNN deterministic/benchmark, 결정적 알고리즘 사용 여부를 설정하고
  실제 적용 상태를 dict로 돌려준다(학습 요약에 기록해 재현 조건을 남긴다).
- environment_snapshot: Python/torch/CUDA/주요 패키지 버전과 GPU 정보를 수집한다.
  결과 차이를 추적할 때 '같은 환경이었는지' 확인하는 근거가 된다.
"""

from __future__ import annotations

import importlib.metadata
import os
import platform
import random
import sys
from pathlib import Path
from typing import Any
from collections.abc import Mapping

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def configure_determinism(train_config: Mapping[str, Any]) -> dict[str, Any]:
    deterministic = bool(train_config.get("deterministic", False))
    warn_only = bool(train_config.get("deterministic_warn_only", True))
    requested_cudnn_benchmark = bool(train_config.get("cudnn_benchmark", False))
    effective_cudnn_benchmark = False if deterministic else requested_cudnn_benchmark

    if deterministic and "CUBLAS_WORKSPACE_CONFIG" not in os.environ:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = effective_cudnn_benchmark
    try:
        torch.use_deterministic_algorithms(deterministic, warn_only=warn_only)
    except TypeError:
        torch.use_deterministic_algorithms(deterministic)

    return {
        "deterministic": deterministic,
        "deterministic_warn_only": warn_only,
        "requested_cudnn_benchmark": requested_cudnn_benchmark,
        "effective_cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "torch_deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled()),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
    }


def _package_version(package_name: str) -> str | None:
    try:
        return importlib.metadata.version(package_name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _local_pycache_tags(project_root: Path | None) -> list[str]:
    if project_root is None:
        return []
    drscreen_root = project_root / "drscreen"
    if not drscreen_root.exists():
        drscreen_root = project_root / "ai" / "drscreen"
    if not drscreen_root.exists():
        return []
    tags: set[str] = set()
    for cache_file in drscreen_root.rglob("*.pyc"):
        name = cache_file.name
        if ".cpython-" not in name:
            continue
        tag = name.split(".cpython-", 1)[1].split(".", 1)[0]
        if tag:
            tags.add(f"cpython-{tag}")
    return sorted(tags)


def environment_snapshot(project_root: Path | None = None) -> dict[str, Any]:
    gpu_names: list[str] = []
    if torch.cuda.is_available():
        gpu_names = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]

    return {
        "python_executable": sys.executable,
        "python_version": sys.version,
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "torch_cudnn_version": torch.backends.cudnn.version(),
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "cuda_device_names": gpu_names,
        "package_versions": {
            "albumentations": _package_version("albumentations"),
            "numpy": _package_version("numpy"),
            "opencv-python": _package_version("opencv-python"),
            "opencv-python-headless": _package_version("opencv-python-headless"),
            "pandas": _package_version("pandas"),
            "pillow": _package_version("pillow"),
            "timm": _package_version("timm"),
            "torch": _package_version("torch"),
            "torchvision": _package_version("torchvision"),
        },
        "local_pycache_python_tags": _local_pycache_tags(project_root),
    }
