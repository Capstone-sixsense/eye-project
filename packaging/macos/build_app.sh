#!/usr/bin/env bash
# Flutter macOS Release + 백엔드·AI·venv 를 .app Resources 에 번들
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VENV="$ROOT/packaging/macos/venv"
ENV_FILE="$ROOT/packaging/macos/bundled.env"
FLUTTER_APP="$ROOT/frontend/build/macos/Build/Products/Release/EyeProject.app"

if [[ ! -x "$VENV/bin/uvicorn" ]]; then
  echo "❌ venv 없음. 먼저 실행:"
  echo "   ./packaging/macos/setup_venv.sh"
  exit 1
fi

if [[ ! -f "$ROOT/ai/artifacts/checkpoints/best.pt" ]]; then
  echo "❌ ai/artifacts/checkpoints/best.pt 없음. git lfs pull 확인"
  exit 1
fi

if file "$ROOT/ai/artifacts/checkpoints/best.pt" | grep -q "ASCII text"; then
  echo "❌ best.pt 가 LFS 포인터입니다. git lfs pull 후 재시도"
  exit 1
fi

if [[ ! -f "$ENV_FILE" ]]; then
  KEY="$(python3 -c 'import base64,os; print(base64.b64encode(os.urandom(32)).decode())')"
  echo "IMAGE_ENCRYPTION_KEY=$KEY" >"$ENV_FILE"
  echo "🔐 bundled.env 생성 (재빌드 시 동일 키 유지 — 삭제하면 기존 암호화 이미지 복호화 불가)"
fi

echo "📱 Flutter macOS Release 빌드..."
cd "$ROOT/frontend"
flutter build macos --release --dart-define=API_BASE_URL=http://127.0.0.1:8000

RES="$FLUTTER_APP/Contents/Resources"
mkdir -p "$RES/backend" "$RES/ai"

echo "📦 Resources 번들 복사..."
rsync -a --delete \
  --exclude '__pycache__' \
  --exclude 'storage' \
  --exclude 'results' \
  --exclude '.cache' \
  --exclude '*.log' \
  --exclude '.env' \
  "$ROOT/backend/" "$RES/backend/"

rsync -a --delete \
  "$ROOT/ai/drscreen" \
  "$ROOT/ai/configs" \
  "$ROOT/ai/artifacts" \
  "$RES/ai/"

echo "🐍 Python venv 복사 (용량 큼, 수 분 소요)..."
rsync -a --delete \
  --exclude '__pycache__' \
  "$VENV/" "$RES/python-venv/"

install -m 755 "$ROOT/packaging/macos/run_backend.sh" "$RES/run_backend.sh"
cp "$ENV_FILE" "$RES/bundled.env"

if command -v codesign >/dev/null 2>&1; then
  echo "🔏 ad-hoc codesign..."
  codesign --force --deep --sign - "$FLUTTER_APP" || true
fi

echo ""
echo "✅ 빌드 완료: $FLUTTER_APP"
echo "   실행: open \"$FLUTTER_APP\""
du -sh "$FLUTTER_APP"
