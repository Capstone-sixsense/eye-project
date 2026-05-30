#!/usr/bin/env bash
# venv + 로컬 ai/backend 로 /health 스모크 테스트 (개발용, .app 빌드 전 확인)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VENV="$ROOT/packaging/macos/venv"
DATA_DIR="${EYE_PROJECT_DATA_DIR:-$TMPDIR/eye-project-smoke-$$}"

if [[ ! -x "$VENV/bin/uvicorn" ]]; then
  echo "❌ venv 없음. 먼저: ./packaging/macos/setup_venv.sh"
  exit 1
fi

mkdir -p "$DATA_DIR/storage" "$DATA_DIR/results" "$DATA_DIR/.cache/torch" "$DATA_DIR/.cache/huggingface" "$DATA_DIR/logs"

export PYTHONPATH="$ROOT/ai"
export FUNDUS_CONFIG_PATH="$ROOT/ai/configs/base.yaml"
export IMAGE_ENCRYPTION_KEY="${IMAGE_ENCRYPTION_KEY:-$(python3 -c 'import base64,os; print(base64.b64encode(os.urandom(32)).decode())')}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export TORCH_NUM_THREADS="${TORCH_NUM_THREADS:-4}"
export TORCH_HOME="$DATA_DIR/.cache/torch"
export HF_HOME="$DATA_DIR/.cache/huggingface"

cd "$ROOT/backend"
ln -sfn "$DATA_DIR/storage" storage
ln -sfn "$DATA_DIR/results" results

echo "🚀 uvicorn 시작 (모델 로드까지 최대 ~2분)..."
"$VENV/bin/uvicorn" main:app --host 127.0.0.1 --port 8000 &
PID=$!
trap 'kill "$PID" 2>/dev/null || true' EXIT

for i in $(seq 1 120); do
  if curl -sf "http://127.0.0.1:8000/health" >/dev/null 2>&1; then
    echo "✅ /health OK"
    curl -s "http://127.0.0.1:8000/health" | head -c 500
    echo
    exit 0
  fi
  sleep 1
done

echo "❌ 120초 내 /health 실패"
exit 1
