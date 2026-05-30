# eye-project · macOS 배포판

**최종 갱신: 2026-05-30**

안저(眼底) 이미지 기반 **당뇨병성 망막병증 보조 스크리닝** MVP의 **macOS 단독 실행판**입니다.  
Docker·브라우저 없이 `EyeProject.app` 하나로 UI, API, AI 추론이 함께 동작합니다.

> **의료 보조 용도** — 최종 판단은 반드시 의료 전문가가 합니다.

---

## 요구 사항


| 항목               | 내용                           |
| ---------------- | ---------------------------- |
| OS               | macOS (Apple Silicon 권장)     |
| Docker           | **불필요**                      |
| Python / Flutter | **실행만 할 때 불필요** (`.app`에 포함) |
| 디스크              | `.app` 약 **1.2GB**           |


---

## 실행 방법

### `.app` 파일만 받은 경우

1. `EyeProject.app`을 원하는 폴더에 둡니다.
2. **더블클릭**으로 실행합니다.
3. 첫 실행 시 AI 모델 로드로 **30~90초** 정도 걸릴 수 있습니다. 창이 뜬 뒤 이미지를 업로드해 분석하면 됩니다.

macOS 보안 경고가 뜨면:

- **우클릭 → 열기**로 한 번 실행하거나
- 터미널에서:

```bash
xattr -cr /path/to/EyeProject.app
open /path/to/EyeProject.app
```

### 저장소에서 직접 빌드한 경우

빌드 산출물 경로:

```text
frontend/build/macos/Build/Products/Release/EyeProject.app
```

실행:

```bash
open frontend/build/macos/Build/Products/Release/EyeProject.app
```

`.app`이 없으면 아래 **최초 빌드**를 먼저 진행합니다.

---

## 최초 빌드 (개발자)

저장소를 clone 한 뒤, 프로젝트 루트에서:

```bash
git lfs pull

chmod +x packaging/macos/*.sh
./packaging/macos/setup_venv.sh    # Python 3.11+ 필요, 최초 1회
./packaging/macos/build_app.sh
```

코드 수정 후 `.app`을 다시 만들 때는 `build_app.sh`만 실행하면 됩니다.

상세 빌드·개발 절차: `[packaging/macos/README.md](packaging/macos/README.md)`

---

## 종료

- 앱 창을 닫으면 백엔드(uvicorn)도 함께 종료됩니다.
- Dock에 남아 있으면 **EyeProject → 종료**로 완전히 닫습니다.

---

## 데이터·로그 위치

분석 이력과 암호화 이미지는 `.app` 밖 사용자 데이터 폴더에 저장됩니다.


| 경로                                                          | 내용        |
| ----------------------------------------------------------- | --------- |
| `~/Library/Application Support/EyeProject/storage/`         | DB·암호화 원본 |
| `~/Library/Application Support/EyeProject/results/`         | 리포트 이미지   |
| `~/Library/Application Support/EyeProject/logs/backend.log` | 백엔드 로그    |


---

## 문제 해결


| 증상                    | 확인                                                                     |
| --------------------- | ---------------------------------------------------------------------- |
| 앱이 바로 종료됨             | `~/Library/Application Support/EyeProject/logs/backend.log`            |
| “손상됨” / Gatekeeper 차단 | 위 **우클릭 → 열기** 또는 `xattr -cr`                                          |
| 포트 충돌                 | 다른 EyeProject/Docker 백엔드가 `:8000`을 쓰는지 확인 후 종료                         |
| 빌드 시 `best.pt` 오류     | `git lfs pull` 후 `file ai/artifacts/checkpoints/best.pt` → Zip archive |


---

## 다른 Mac으로 복사

`EyeProject.app` **폴더 통째로** 복사하면 됩니다.  
대상 Mac도 **Apple Silicon macOS**를 권장합니다.  
팀 외부·불특정 다수 배포 시에는 Apple Developer **서명·공증(notarize)** 이 필요합니다. (현재 미수행)