#!/usr/bin/env bash
# POSIX sh 로 실행되면 bash 전용 문법(for ((…)))에서 깨지므로 bash 로 다시 띄웁니다.
if [ -z "${BASH_VERSION:-}" ]; then
    exec /usr/bin/env bash "$0" "$@"
fi
# eye-project 루트에서 실행하세요. (내부에서 스크립트 위치로 이동합니다.)
set -eo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

echo "⚙️  Eye-Project 환경 구축을 시작합니다..."

# Docker Compose: Mac Apple Silicon은 PyTorch CUDA 휠 이슈로 amd64 백엔드 오버레이 사용
eye_compose() {
    if [ "$(uname -s)" = Darwin ] && [ "$(uname -m)" = arm64 ]; then
        docker compose -f docker-compose.yml -f docker-compose-mac.yml "$@"
    else
        docker compose "$@"
    fi
}

open_browser() {
    local url="$1"
    if command -v open >/dev/null 2>&1; then
        open "$url"
    elif command -v xdg-open >/dev/null 2>&1; then
        xdg-open "$url"
    elif command -v cmd.exe >/dev/null 2>&1; then
        cmd.exe /c start "" "$url"
    else
        echo "브라우저에서 직접 열어 주세요: $url"
    fi
}

# 1. git 미추적 AI 폴더 생성
mkdir -p ai/data ai/storage ai/artifacts/checkpoints ai/artifacts/predictions ai/artifacts/heatmaps ai/artifacts/quickqual

# 2. Flutter 프로젝트가 없는 경우에만 자동 생성
if [ ! -f "./frontend/pubspec.yaml" ]; then
    echo '📦 Flutter 프로젝트를 생성 중입니다 (windows, linux)...'
    docker run --rm -v "$(pwd)/frontend:/app" -w /app \
        ghcr.io/cirruslabs/flutter:stable \
        flutter create . --platforms windows,linux --project-name eye_project
fi

# 3. 파일 권한 (Unix만 — Mac에서 $USER:$USER 가 그룹 오류 날 때 id -gn 사용)
case "$(uname -s)" in
    Darwin|Linux)
        echo "🔐 파일 권한을 설정합니다..."
        sudo chown -R "$(id -un):$(id -gn)" ./frontend
        ;;
    *)
        echo '🔐 Windows 환경: chown 생략 (Git Bash / CMD)'
        ;;
esac

# 4. 의존성 설치 확인
echo "📥 Flutter 패키지를 가져옵니다..."
docker run --rm -v "$(pwd)/frontend:/app" -w /app \
    ghcr.io/cirruslabs/flutter:stable \
    flutter pub get

echo "✅ Flutter 쪽 준비가 완료되었습니다."

# 5. Docker Compose + 브라우저
if [ "$(uname -s)" = Darwin ] && [ "$(uname -m)" = arm64 ]; then
    echo '🍎 Apple Silicon 감지: docker-compose-mac.yml (백엔드 linux/amd64)을 함께 사용합니다.'
fi
echo "🚀 Docker Compose 로 서비스를 띄웁니다..."
if ! eye_compose up --build -d; then
    echo "❌ docker compose 가 실패했습니다. Docker Desktop 실행 여부와 오류 메시지를 확인하세요."
    exit 1
fi

echo '⏳ 백엔드(8000) 준비 대기...'
BACKEND_OK=0
for ((i = 0; i < 120; i++)); do
    if curl -sf "http://127.0.0.1:8000/health" >/dev/null 2>&1; then
        BACKEND_OK=1
        break
    fi
    sleep 1
done
if [ "$BACKEND_OK" -ne 1 ]; then
    echo '⚠️  120초 안에 백엔드(8000)가 응답하지 않습니다.'
    eye_compose ps || true
    echo "   로그: Apple Silicon이면 -f docker-compose.yml -f docker-compose-mac.yml 로 실행한 compose 기준으로 backend 로그 확인"
fi

echo '⏳ 프론트(8080) 준비 대기… (최초 빌드는 수 분 걸릴 수 있습니다)'
FRONTEND_OK=0
for ((i = 0; i < 600; i++)); do
    if curl -sf -o /dev/null "http://127.0.0.1:8080/" 2>/dev/null; then
        FRONTEND_OK=1
        break
    fi
    sleep 1
done

URL="http://127.0.0.1:8080"
if [ "$FRONTEND_OK" -eq 1 ]; then
    echo "🌐 브라우저 열기: $URL"
    open_browser "$URL"
    echo "✅ 모든 준비가 완료되었습니다."
    if [ "$(uname -s)" = Darwin ] && [ "$(uname -m)" = arm64 ]; then
        echo "   종료: docker compose -f docker-compose.yml -f docker-compose-mac.yml down"
    else
        echo "   종료: docker compose down"
    fi
else
    echo "❌ 8080에 연결되지 않아 브라우저를 열지 않았습니다."
    echo "   직접 열기: $URL"
    eye_compose ps || true
    echo '   docker compose logs frontend | tail -n 120  (위와 동일 -f 조합으로 로그 확인)'
    exit 1
fi
