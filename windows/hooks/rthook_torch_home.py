"""
Runtime hook: PyInstaller 실행 시 HuggingFace/Torch 캐시를 번들 내부로 설정.

timm.create_model("densenet121.tv_in1k", pretrained=True) 는
HuggingFace Hub에서 가중치를 다운로드한다.

HF Hub 캐시 구조:
  blobs/{hash}                          ← 실제 가중치 파일 (31MB)
  refs/main                             ← 커밋 해시 참조
  snapshots/{commit_hash}/model.safetensors  ← blobs의 심볼릭 링크

Windows에서는 심볼릭 링크가 동작하지 않으므로, blob을 snapshot 경로에
model.safetensors 이름으로 복사해 HF Hub가 오프라인으로 찾을 수 있게 한다.
"""
import os
import shutil
import sys
from pathlib import Path

BLOB_HASH   = "c894c6d9caa317a8ca1942986dee7a16a86c77734a4d691d2abe05389cfef358"
COMMIT_HASH = "f0d0f2698a02cb133b09d48396db6e1e46fe9f3b"
HF_MODEL_DIR = Path(".cache/huggingface/hub/models--timm--densenet121.tv_in1k")

if getattr(sys, "frozen", False):
    bundle_dir = Path(getattr(sys, "_MEIPASS", Path(sys.executable).parent))

    # HuggingFace 캐시 경로 설정
    hf_home = bundle_dir / ".cache" / "huggingface"
    os.environ.setdefault("HF_HOME", str(hf_home))
    os.environ.setdefault("HF_HUB_OFFLINE", "1")   # 네트워크 접근 차단
    os.environ.setdefault("TORCH_HOME", str(bundle_dir / ".cache" / "torch"))

    # blobs → snapshots/model.safetensors 복사 (심볼릭 링크 대체)
    blob_path      = bundle_dir / HF_MODEL_DIR / "blobs" / BLOB_HASH
    snapshot_dir   = bundle_dir / HF_MODEL_DIR / "snapshots" / COMMIT_HASH
    safetensors    = snapshot_dir / "model.safetensors"

    if blob_path.exists() and not safetensors.exists():
        snapshot_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(str(blob_path), str(safetensors))
