#!/bin/bash

echo "⚙️  Eye-Project 환경 구축을 시작합니다..."

# 1. git 미추적 AI 폴더 생성
mkdir -p ai/data ai/storage ai/artifacts/checkpoints ai/artifacts/predictions ai/artifacts/heatmaps ai/artifacts/quickqual

# 2. Flutter 프로젝트가 없는 경우에만 자동 생성
if [ ! -f "./frontend/pubspec.yaml" ]; then
    echo "📦 Flutter 프로젝트를 생성 중입니다 (windows, linux)..."
    docker run --rm -v $(pwd)/frontend:/app -w /app \
        ghcr.io/cirruslabs/flutter:stable \
        flutter create . --platforms windows,linux --project-name eye_project
fi

# 3. 파일 권한 조정 (Docker가 생성한 파일의 소유권을 현재 사용자로 변경)
echo "🔐 파일 권한을 설정합니다..."
sudo chown -R $USER:$USER ./frontend

# 4. 의존성 설치 확인 [cite: 6]
echo "📥 Flutter 패키지를 가져옵니다..."
docker run --rm -v $(pwd)/frontend:/app -w /app \
    ghcr.io/cirruslabs/flutter:stable \
    flutter pub get

echo "✅ 모든 준비가 완료되었습니다!"
echo "이제 'docker compose up --build'를 실행하여 서버를 가동하세요."