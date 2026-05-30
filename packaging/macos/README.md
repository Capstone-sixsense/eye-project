# macOS `.app` 빌드 가이드

Docker 없이 **Flutter macOS UI + 로컬 FastAPI + AI** 를 하나의 `.app` 으로 묶습니다.

## 한 번에 (최초)

```bash
cd eye-project

# 1) AI 체크포인트 (LFS)
git lfs pull

# 2) Python venv (PyTorch 포함, 5~15분+)
chmod +x packaging/macos/*.sh
./packaging/macos/setup_venv.sh

# 3) (선택) 백엔드만 스모크 테스트
./packaging/macos/smoke_backend.sh

# 4) .app 빌드 (Flutter + 번들, venv 복사로 수 분)
./packaging/macos/build_app.sh
```

산출물:

```text
frontend/build/macos/Build/Products/Release/EyeProject.app
```

실행:

```bash
open frontend/build/macos/Build/Products/Release/EyeProject.app
```

## 개발 중 UI만 (백엔드는 venv로 별도 실행)

```bash
# 터미널 1
./packaging/macos/smoke_backend.sh   # 또는 uvicorn 수동 실행

# 터미널 2
cd frontend
flutter run -d macos --dart-define=API_BASE_URL=http://127.0.0.1:8000
```

번들 백엔드가 없으면 AppDelegate 가 외부 `:8000` 을 짧게만 확인하고 넘어갑니다.

## 데이터·로그 위치

| 경로 | 내용 |
|------|------|
| `~/Library/Application Support/EyeProject/storage/` | 암호화 원본·DB |
| `~/Library/Application Support/EyeProject/results/` | 리포트 이미지 |
| `~/Library/Application Support/EyeProject/logs/backend.log` | uvicorn 로그 |
| `packaging/macos/bundled.env` | `IMAGE_ENCRYPTION_KEY` (재빌드 시 유지 권장) |

## 스크립트

| 파일 | 역할 |
|------|------|
| `setup_venv.sh` | mac CPU PyTorch venv 생성 |
| `smoke_backend.sh` | `/health` 스모크 테스트 |
| `build_app.sh` | Release `.app` + Resources 번들 |
| `run_backend.sh` | `.app` 내부에서 uvicorn 기동 |

## 용량·시간 참고

- venv + PyTorch + checkpoint: `.app` **약 1.5~2.5GB**
- 첫 기동 시 모델 로드: **~30–90초** (CPU)

## 문제 해결

- **`best.pt` LFS 포인터**: `git lfs pull`, `file ai/artifacts/checkpoints/best.pt` → Zip archive
- **백엔드 기동 실패**: `~/Library/Application Support/EyeProject/logs/backend.log`
- **Gatekeeper 차단**: `xattr -cr EyeProject.app` 또는 ad-hoc codesign (`build_app.sh` 가 시도함)
- **외부 배포**: Developer ID 서명 + notarize 필요 (별도 Apple Developer 계정)
