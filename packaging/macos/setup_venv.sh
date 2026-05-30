#!/usr/bin/env bash
# macOS .app 번들용 Python venv 생성 (최초 1회, 수 분~10분+ 소요).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VENV="$ROOT/packaging/macos/venv"
REQ="$ROOT/packaging/macos/backend_requirements_mac"

pick_python() {
  for candidate in python3.11 python3.12 python3.13 python3; do
    if command -v "$candidate" >/dev/null 2>&1; then
      local ver
      ver="$("$candidate" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
      local major minor
      major="${ver%%.*}"
      minor="${ver#*.}"
      if (( major > 3 || (major == 3 && minor >= 10) )); then
        echo "$candidate"
        return 0
      fi
    fi
  done
  return 1
}

PY="$(pick_python)" || {
  echo "❌ Python 3.10+ 가 필요합니다."
  echo "   macOS: brew install python@3.11"
  echo "   이후 PATH 에 python3.11 이 잡히는지 확인하세요."
  exit 1
}

echo "⚙️  Python: $($PY --version)"
echo "📦 venv: $VENV"

if [[ -d "$VENV" ]]; then
  echo "ℹ️  기존 venv 가 있습니다. 삭제 후 재생성하려면: rm -rf \"$VENV\""
else
  "$PY" -m venv "$VENV"
fi

# shellcheck disable=SC1091
source "$VENV/bin/activate"
python -m pip install --upgrade pip wheel
pip install -r "$REQ"

echo "✅ venv 준비 완료"
echo "   스모크 테스트: packaging/macos/smoke_backend.sh"
