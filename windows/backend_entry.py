"""Windows offline backend entrypoint.

Keeps backend source code unchanged by preparing the frozen-exe runtime here:
config/checkpoint paths, writable AppData directories, encryption key, and
QuickQual model path.
"""
from __future__ import annotations

import base64
import os
import secrets
import shutil
import sys
from pathlib import Path

import yaml


def _bundle_dir() -> Path:
    return Path(getattr(sys, "_MEIPASS", Path(sys.executable).parent)).resolve()


def _appdata_root() -> Path:
    base = os.environ.get("APPDATA") or str(Path.home())
    root = Path(base) / "EyeProject"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _ensure_utf8_stdio() -> None:
    for stream in (getattr(sys, "stdout", None), getattr(sys, "stderr", None)):
        if stream is None or not hasattr(stream, "reconfigure"):
            continue
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


def _ensure_encryption_key(app_root: Path) -> None:
    if os.environ.get("IMAGE_ENCRYPTION_KEY", "").strip():
        return
    key_path = app_root / ".key"
    if key_path.exists():
        key = key_path.read_bytes()
        if len(key) == 32:
            os.environ["IMAGE_ENCRYPTION_KEY"] = base64.b64encode(key).decode("ascii")
            return
    key = secrets.token_bytes(32)
    key_path.write_bytes(key)
    os.environ["IMAGE_ENCRYPTION_KEY"] = base64.b64encode(key).decode("ascii")


def _configure_cache_env(app_root: Path) -> None:
    cache_root = app_root / ".cache"
    hf_home = cache_root / "huggingface"
    torch_home = cache_root / "torch"
    hf_home.mkdir(parents=True, exist_ok=True)
    torch_home.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["HF_HOME"] = str(hf_home)
    os.environ["TORCH_HOME"] = str(torch_home)


def _copy_evaluations(bundle: Path, runtime_project: Path) -> None:
    src = bundle / "artifacts" / "evaluations"
    dst = runtime_project / "artifacts" / "evaluations"
    if not src.exists():
        return
    dst.mkdir(parents=True, exist_ok=True)
    for item in src.iterdir():
        if item.is_file():
            shutil.copy2(item, dst / item.name)


def _write_runtime_config(bundle: Path, app_root: Path) -> Path:
    runtime_project = app_root / "runtime"
    config_dir = runtime_project / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    _copy_evaluations(bundle, runtime_project)

    source_config = bundle / "configs" / "base.yaml"
    target_config = config_dir / "base.yaml"
    with source_config.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}

    infer = config.setdefault("infer", {})
    infer["checkpoint_path"] = str(bundle / "artifacts" / "checkpoints" / "best.pt")
    infer["prediction_dir"] = str(app_root / "artifacts" / "predictions")
    infer["heatmap_dir"] = str(app_root / "artifacts" / "heatmaps")

    train = config.setdefault("train", {})
    train["checkpoint_dir"] = str(app_root / "artifacts" / "runs")
    train["global_best_checkpoint_path"] = str(bundle / "artifacts" / "checkpoints" / "best.pt")

    with target_config.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, allow_unicode=True, sort_keys=False)
    return target_config


def _configure_runtime() -> Path:
    bundle = _bundle_dir()
    app_root = _appdata_root()
    _ensure_utf8_stdio()
    _ensure_encryption_key(app_root)
    _configure_cache_env(app_root)
    os.chdir(app_root)

    runtime_config = _write_runtime_config(bundle, app_root)
    os.environ["FUNDUS_CONFIG_PATH"] = str(runtime_config)
    os.environ.setdefault("QUICKQUAL_SVM_FILENAME", str(bundle / "models" / "quickqual_dn121_512.pkl"))
    return runtime_config


def main() -> None:
    _configure_runtime()
    import uvicorn
    from main import app

    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
