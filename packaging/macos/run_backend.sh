#!/usr/bin/env bash
# .app/Contents/Resources/run_backend.sh — 번들 백엔드 기동
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
DATA_DIR="${EYE_PROJECT_DATA_DIR:-$HOME/Library/Application Support/EyeProject}"

if [[ -f "$ROOT/bundled.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$ROOT/bundled.env"
  set +a
fi

mkdir -p "$DATA_DIR/storage" "$DATA_DIR/results" "$DATA_DIR/.cache/torch" "$DATA_DIR/.cache/huggingface" "$DATA_DIR/logs"

export PYTHONPATH="$ROOT/ai"
export FUNDUS_CONFIG_PATH="$ROOT/ai/configs/base.yaml"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"
export TORCH_NUM_THREADS="${TORCH_NUM_THREADS:-8}"
export TORCH_HOME="$DATA_DIR/.cache/torch"
export HF_HOME="$DATA_DIR/.cache/huggingface"
export QUICKQUAL_SVM_FILENAME="${QUICKQUAL_SVM_FILENAME:-quickqual_dn121_512.pkl}"
export FUNDUS_CHECK_ENABLED="${FUNDUS_CHECK_ENABLED:-true}"

BACKEND_DIR="$ROOT/backend"
mkdir -p "$BACKEND_DIR"

# cwd=backend 기준 상대 경로(storage, results)를 Application Support 로 연결
link_or_replace() {
  local name="$1"
  local target="$DATA_DIR/$name"
  local link_path="$BACKEND_DIR/$name"
  if [[ -L "$link_path" ]]; then
    rm "$link_path"
  elif [[ -d "$link_path" ]]; then
    echo "⚠️  $link_path 가 디렉터리입니다. Application Support 데이터로 이동 후 심볼릭 링크를 만듭니다."
    rm -rf "$link_path"
  fi
  ln -s "$target" "$link_path"
}

link_or_replace storage
link_or_replace results

cd "$BACKEND_DIR"
exec "$ROOT/python-venv/bin/uvicorn" main:app --host 127.0.0.1 --port 8000 \
  >>"$DATA_DIR/logs/backend.log" 2>&1
