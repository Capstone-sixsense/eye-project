"""Runtime hook: prepare offline HuggingFace/Torch cache in AppData.

timm.create_model("densenet121.tv_in1k", pretrained=True) 는
HuggingFace Hub에서 가중치를 다운로드한다.

HF Hub 캐시 구조:
  blobs/{hash}                          ← 실제 가중치 파일 (31MB)
  refs/main                             ← 커밋 해시 참조
  snapshots/{commit_hash}/model.safetensors  ← blobs의 심볼릭 링크

Windows에서는 심볼릭 링크가 동작하지 않으므로, bundled blob을 사용자
AppData cache의 snapshot/model.safetensors로 복사해 HF Hub가 오프라인으로
찾을 수 있게 한다. 설치 폴더는 읽기 전용일 수 있으므로 쓰지 않는다.
"""
import os
import shutil
import sys
from pathlib import Path

BLOB_HASH   = "c894c6d9caa317a8ca1942986dee7a16a86c77734a4d691d2abe05389cfef358"
COMMIT_HASH = "f0d0f2698a02cb133b09d48396db6e1e46fe9f3b"
HF_MODEL_DIR = Path(".cache/huggingface/hub/models--timm--densenet121.tv_in1k")


def _appdata_root() -> Path:
    base = os.environ.get("APPDATA") or str(Path.home())
    root = Path(base) / "EyeProject"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _copy_if_needed(src: Path, dst: Path) -> bool:
    if not src.exists():
        return False
    if dst.exists() and dst.stat().st_size == src.stat().st_size:
        return True
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(str(src), str(dst))
    return True


if getattr(sys, "frozen", False):
    bundle_dir = Path(getattr(sys, "_MEIPASS", Path(sys.executable).parent)).resolve()
    app_root = _appdata_root()

    # HuggingFace/Torch cache must be writable even when install dir is not.
    hf_home = app_root / ".cache" / "huggingface"
    torch_home = app_root / ".cache" / "torch"
    hf_home.mkdir(parents=True, exist_ok=True)
    torch_home.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(hf_home)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TORCH_HOME"] = str(torch_home)

    bundled_model_dir = bundle_dir / HF_MODEL_DIR
    runtime_model_dir = hf_home / "hub" / "models--timm--densenet121.tv_in1k"

    bundled_blob = bundled_model_dir / "blobs" / BLOB_HASH
    runtime_blob = runtime_model_dir / "blobs" / BLOB_HASH
    _copy_if_needed(bundled_blob, runtime_blob)

    bundled_ref = bundled_model_dir / "refs" / "main"
    runtime_ref = runtime_model_dir / "refs" / "main"
    if not _copy_if_needed(bundled_ref, runtime_ref) and not runtime_ref.exists():
        runtime_ref.parent.mkdir(parents=True, exist_ok=True)
        runtime_ref.write_text(COMMIT_HASH, encoding="utf-8")

    # Replace HF snapshot symlink with a real AppData file.
    runtime_snapshot = runtime_model_dir / "snapshots" / COMMIT_HASH / "model.safetensors"
    snapshot_src = runtime_blob if runtime_blob.exists() else bundled_blob
    _copy_if_needed(snapshot_src, runtime_snapshot)
