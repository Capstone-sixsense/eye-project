# Eye Project · Frontend

> **이 문서가 다루는 디렉터리(`eye-project/frontend/`)는 저장소의 프론트엔드 브랜치(Flutter 클라이언트)입니다.**  
> 백엔드·모델·인프라 코드는 이 폴더 밖에 있으며, 여기서는 **망막 이미지 업로드 → `/analyze` API 호출 → 결과 화면** 흐름만 담당합니다.

망막 사진을 선택해 서버로 보내고, AI 분석 결과(라벨, 이상 확률, 품질, 설명 이미지 URL 등)를 화면에 보여 주는 **Flutter** 앱입니다.

---

## 목차

1. [기술 스택](#기술-스택)
2. [폴더 구조 한눈에](#폴더-구조-한눈에)
3. [실행 방법](#실행-방법)
4. [Dart 소스 전체 가이드](#dart-소스-전체-가이드)
5. [화면·라우팅 흐름](#화면라우팅-흐름)
6. [테스트](#테스트)

---

## 기술 스택

| 구분 | 내용 |
|------|------|
| 프레임워크 | Flutter (Dart SDK `^3.11.3`) |
| HTTP | `http` — 멀티파트 업로드 및 JSON 응답 처리 |
| 파일 선택 | `file_picker` — 이미지 선택 및 바이트 로드 |
| UI | Material 3 (`MaterialApp`, `Scaffold`, `Theme`) |

---

## 폴더 구조 한눈에

```text
frontend/
├── lib/                      # 앱 Dart 소스 (핵심)
│   ├── main.dart             # 엔트리, 라우트 정의
│   ├── api/                  # HTTP 클라이언트
│   ├── config/               # API 베이스 URL 등
│   ├── constants/            # 오류 코드 상수
│   ├── models/               # 응답·인자 모델
│   └── screens/              # 화면 위젯
├── test/                     # 위젯 테스트
├── pubspec.yaml              # 패키지명·의존성
├── analysis_options.yaml     # 분석/린트
├── web/, linux/, windows/    # 플랫폼 러너 (Flutter 기본)
└── frontend_Dockerfile*      # 컨테이너 빌드용
```

---

## 플러터 단독 실행 방법

```bash
cd frontend   # 이 README가 있는 디렉터리
flutter pub get
flutter run
```

백엔드 주소가 기본(`http://127.0.0.1:8000`)과 다르면:

```bash
flutter run --dart-define=API_BASE_URL=http://호스트:포트
```

- Windows 등에서 `localhost`가 IPv6만 잡히면 Docker 포트와 어긋날 수 있어, 코드 기본값을 **`127.0.0.1`**로 두었습니다. (`api_config.dart` 주석 참고).

---

## Dart 소스 전체 가이드

아래는 **`lib/` 및 `test/`의 모든 `.dart` 파일**에 대한 역할과 코드 수준 설명입니다.

### `lib/main.dart`

| 항목 | 설명 |
|------|------|
| **역할** | 앱 진입점(`main`)과 **전역 라우팅**만 담당합니다. |
| **핵심 코드** | `MaterialApp`으로 `debugShowCheckedModeBanner: false`, `initialRoute: '/upload'`. |
| **라우트** | `/upload` → `UploadScreen`, `/history` → `HistoryScreen`, `/result` → `ResultScreen`. |
| **`/result`<br>인자** | `ModalRoute.settings.arguments`가 `ResultScreenArgs`이면 원본 바이트+`AnalyzeResponse`로 결과 화면 구성. **구버전 호환**으로 `Uint8List`만 넘기면 원본만 표시. 그 외에는 인자 없이 `ResultScreen()` 호출. |

**import**

| 모듈 | 설명 |
|------|------|
| `dart:typed_data` | 라우트 인자 타입 `Uint8List` 판별. |
| `package:flutter/material.dart` | `runApp`, `MaterialApp`, `ModalRoute`. |
| `models/result_screen_args.dart` | `/result`의 주 인자 타입. |
| `screens/history_screen.dart` 등 | 각 이름 라우트에 대응하는 화면 위젯. |

---

### `lib/config/api_config.dart`

| 항목 | 설명 |
|------|------|
| **역할** | 백엔드 **베이스 URL**과 서버가 돌려준 **상대 경로 → 절대 이미지 URL** 변환. |
| **`ApiConfig.baseUrl`** | `--dart-define=API_BASE_URL=...` 없으면 `http://127.0.0.1:8000`. |
| **`resolveAssetUrl`** | `results/...` 같은 경로를 `baseUrl` 기준으로 합쳐 네트워크 이미지 로드에 쓰입니다. 백슬래시를 슬래시로 통일하고 선행 `/`를 정리합니다. |

---

### `lib/constants/api_error_codes.dart`

| 항목 | 설명 |
|------|------|
| **역할** | 백엔드와 맞춘 **비즈니스 오류 코드** 상수만 보관합니다. |
| **상수** | `INPUT_CH_001` — 입력 채널 미지원(4채널·CMYK 등). `XAI_001` — 설명 이미지 생성 실패 등 XAI 관련. |
| **사용처** | `UploadScreen`에서 업로드 후 응답/예외의 코드와 비교, `ResultScreen`의 설명 패널 기본 코드 표시. |

---

### `lib/models/analyze_response.dart`

| 항목 | 설명 |
|------|------|
| **역할** | `POST /analyze` **성공 JSON**을 Dart 객체로 파싱하고, UI가 쓰는 **파생 getter**를 제공합니다. |
| **`QualitySummary`** | `quality` JSON 맵을 받아 `is_acceptable`, `is_low_quality`, `warning`, `grade`, `grade_confidence` 등 **여러 키 이름**을 흡수해 표시용 필드로 정규화합니다. |
| **`AnalyzeResponse` 필드** | `status`, `message`, `details`, `label`/`predicted_label`, `abnormal_probability`, `report_url`, `original_url`, `preprocessed`, `error_code`, `quality`, `explanation_url`/`explanation_image_url`/`heatmap_url`, `xai_error_code`/`xai_error` 등 API 확장에 맞춘 선택 필드. |
| **`fromJson`** | 위 키들을 읽어 `AnalyzeResponse` 생성. `preprocessed`는 int/bool/num 모두 처리. |
| **getter** | `resolvedExplanationPath` — XAI 실패 시 null, 아니면 설명 URL 우선 후 `report_url` 폴백. `isSuccess`/`isFail`, `canShowInferenceResults`(전처리 미통과 시 판정 숨김), `shouldShowExplanationFailure`(성공·추론 표시 가능인데 XAI 코드만 있는 경우). |

---

### `lib/models/result_screen_args.dart`

| 항목 | 설명 |
|------|------|
| **역할** | `/result`로 네비게이션할 때 넘기는 **불변 인자 묶음**. |
| **필드** | `originalImageBytes` — 사용자가 고른 이미지의 메모리 바이트. `analyzeResponse` — 방금 받은 API 응답 모델. |

---

### `lib/api/eye_api_client.dart`

| 항목 | 설명 |
|------|------|
| **역할** | 백엔드와의 **HTTP 통신** 전담. 멀티파트 필드명은 백엔드와 동일하게 **`image`**. |
| **`parseErrorCodeFromBody`** | 오류 응답 본문 JSON에서 `error_code`, `detail` 문자열/배열 등을 훑어 **표시·분기용 코드 문자열**을 추출합니다(실패 시 null). |
| **`EyeApiException`** | 비-2xx 또는 JSON 파싱 실패 시 `statusCode`, `body`, 선택적 `errorCode`를 담아 던집니다. |
| **`EyeApiClient`** | 생성 시 `http.Client` 주입 가능(테스트용). |
| **`analyze`** | `POST {baseUrl}/analyze`에 `MultipartFile.fromBytes('image', ...)`. |
| **타임아웃** | 전송 단계 기본 **20분**(`analyzeTimeout`) — Docker CPU 추론 대기용. 스트림→`Response` 변환은 **2분** 제한. |
| **성공 처리** | 2xx이고 본문이 `Map`이면 `AnalyzeResponse.fromJson`. |
| **실패 처리** | 로그 후 `parseErrorCodeFromBody`, `detail` 메시지 정리해 `EyeApiException` throw. |
| **`close`** | 내부 `http.Client` 종료. `UploadScreen.dispose`에서 호출. |

---

### `lib/screens/upload_screen.dart`

| 항목 | 설명 |
|------|------|
| **역할** | **이미지 선택 → 미리보기 → 업로드/분석 요청 → 결과 또는 오류 UI** 까지의 메인 워크플로. |
| **상태** | `fileName`, `fileBytes`, `_uploading`, `EyeApiClient` 인스턴스. |
| **`pickFile`** | `FilePicker`로 이미지 1개, `withData: true`. 웹 등에서 `bytes`가 null이면 스낵바로 안내. |
| **`_uploadAndAnalyze`** | 바이트·파일명 검증 후 `_uploading`, **닫을 수 없는** 진행 다이얼로그(CPU 모드 안내 문구 포함). 성공 시 `INPUT_CH_001`이면 다이얼로그로 안내 후 종료. 그 외 `ResultScreenArgs`로 `/result` push. |
| **예외** | `TimeoutException` → 스낵바. `EyeApiException` → 입력 채널 오류면 다이얼로그, 아니면 상태코드+본문 스낵바. 기타 → 스낵바에 예외와 `ApiConfig.baseUrl` 힌트. |
| **`finally`** | root `Navigator`로 진행 다이얼로그 pop, `_uploading` false. |
| **`dispose`** | `_api.close()`. |
| **`build`** | AppBar 제목 `Upload Retinal Image`, 현재 API URL, 파일 크기(KB), 썸네일, Select / Upload 버튼. 업로드 중 버튼 비활성 및 작은 `CircularProgressIndicator`. |

---

### `lib/screens/result_screen.dart`

| 항목 | 설명 |
|------|------|
| **역할** | 분석 **결과 시각화**: 원본 메모리 이미지, 설명 이미지(네트워크), 판정 카드, 이상 확률·품질 카드, 하단 액션. |
| **생성자** | `args` 우선, 없으면 `originalImageBytes`만(구 라우트). |
| **레이아웃** | `LayoutBuilder`로 **너비 720px 이상**이면 원본·설명을 가로 2열, 그 아래 판정/점수. 좁으면 세로 스택. |
| **`_ImageBox` / `_kResultImageMaxHeight`** | 원본 타일 최대 높이 280, 테두리·라운드 클립. |
| **`_JudgmentCard`** | 응답 없음/실패/전처리 미통과 시 안내 문구, 성공·추론 표시 가능 시 `label` 크게 표시. |
| **`_ScoreQualityCard`** | 이상 확률 %, 품질 등급·신뢰도·acceptable·warning. 품질 블록 없으면 legacy API 안내. |
| **`_ExplanationPanel`** | `shouldShowExplanationFailure`면 실패 문구+`xaiErrorCode`(없으면 `XAI_001`). URL 있으면 `Image.network`+로딩·에러(CORS 등) 처리. 없으면 placeholder 카드. |
| **`_InfoCard`** | 공통 카드 스타일(테두리·연한 배경·패딩). |
| **하단 버튼** | `다시 업로드` — `/upload`까지 스택 제거. `이력 보기` — `/history` push. |

---

### `lib/screens/history_screen.dart`

| 항목 | 설명 |
|------|------|
| **역할** | **분석 이력** 화면 자리만 잡은 **플레이스홀더**. |
| **현재 동작** | 중앙에 “분석 이력은 추후 연동 예정입니다.” 문구만 표시. 백엔드·로컬 스토리지 연동은 이후 작업으로 가정. |

---

### `test/widget_test.dart`

| 항목 | 설명 |
|------|------|
| **역할** | Flutter 기본 **위젯 테스트** — `UploadScreen`이 깨지지 않고 빌드되는지 검증. |
| **테스트 내용** | `MaterialApp(home: UploadScreen())`을 pump한 뒤 AppBar 제목 문자열 `Upload Retinal Image`가 하나 있는지 `expect`. |
| **import** | 패키지명은 `pubspec.yaml`의 `name: eye_project`에 맞춰 `package:eye_project/screens/upload_screen.dart` 사용. |

---

## 화면·라우팅 흐름

1. 앱 시작 → **`/upload`** (`UploadScreen`).
2. **Upload**로 분석 성공 → **`ResultScreenArgs`**를 `arguments`에 실어 **`/result`** (`ResultScreen`).
3. **`/result`**에서 **다시 업로드** → 스택을 비우고 **`/upload`** 로 복귀.
4. **`/result`**에서 **이력 보기** → **`/history`** (`HistoryScreen`, 현재 플레이스홀더).
5. **`/upload`**에서 별도 “이력” 진입은 `main.dart` 라우트만 정의되어 있고, 화면 버튼은 결과 화면 쪽에 있음.

---

## 테스트

```bash
cd frontend
flutter test
```

---

## 관련 파일 (Dart 외)

| 경로 | 용도 |
|------|------|
| `pubspec.yaml` | 패키지 메타데이터, `http` / `file_picker` / `cupertino_icons` 의존성 |
| `analysis_options.yaml` | `flutter_lints` 기반 정적 분석 규칙 |
| `web/index.html`, `web/manifest.json` | 웹 빌드 진입점 |
| `frontend_Dockerfile`, `docker-web.nginx.conf` | 컨테이너용 Web 릴리스 빌드 + nginx |

---
# eye-project

안저(眼底) 이미지 기반 당뇨병성 망막병증 보조 판별 AI 시스템. 이미지를 업로드하면 AI가 이상 여부를 분류하고, GradCAM 히트맵과 의료 리포트를 함께 제공한다. 본 시스템의 결과는 의료 전문가의 판단을 보조하는 용도로만 사용되어야 한다.

---

## 브랜치 구조

```
main
├── ai        # AI 팀 개발 브랜치
├── backend   # 백엔드 팀 개발 브랜치
└── frontend  # 프론트엔드 팀 개발 브랜치
```

각 컴포넌트 브랜치에서 작업한 후 `main`에 머지하는 방식으로 운영된다.

---

## 프로젝트 개요

| 항목 | 내용 |
|---|---|
| 목적 | 안저 이미지 기반 당뇨병성 망막병증 자동 스크리닝 |
| 출력 | 이상 여부 분류, 이상 확률, GradCAM 히트맵, 의료 리포트 |
| 배포 | Docker Compose 기반 로컬 실행형 클라이언트 |
| 설계 원칙 | 의료 시스템 특성상 속도보다 신뢰성과 정확성을 우선한다 |

---

## 프로젝트 목표

- 공개 안저 이미지 데이터셋을 활용하여 당뇨병성 망막병증 보조 판별 AI 모델을 학습한다.
- 사용자가 안저 이미지를 업로드하면 결과를 확인할 수 있는 클라이언트 기반 시스템을 구현한다.
- 예측 결과와 함께 신뢰도 점수 및 시각적 설명 정보를 제공하여 결과 해석 가능성을 높인다.
- 캡스톤디자인 과목 범위에 적합한 MVP를 완성하고, 실제 시연 가능한 형태로 구현한다.
- 향후 안저 촬영 장비 또는 스마트폰 기반 보조 촬영 장치와 연계 가능한 구조로 확장 가능성을 고려한다.

---

## 시스템 구조

```
[Flutter Web UI]  :8080
       │  HTTP
       ▼
[FastAPI Backend] :8000
       │
       ▼
[drscreen AI 패키지]
  - 이미지 전처리
  - EfficientNet-B5 추론
  - GradCAM 히트맵 생성
  - 의료 리포트 생성
```

---

## 기술 스택

| 컴포넌트 | 기술 |
|---|---|
| AI | PyTorch, EfficientNet-B5, FDA, SWAD, albumentations |
| Backend | FastAPI, Uvicorn, OpenCV, GradCAM, cleanvision |
| Frontend | Flutter (Dart) |
| 인프라 | Docker Compose |

---

## 시작하기

```bash
# 1. 저장소 클론
git clone https://github.com/Capstone-sixsense/eye-project.git
cd eye-project

# 2. 디렉토리·의존성 초기화 + Docker 기동 + 브라우저 자동 오픈
./setup.sh
```

**맥**

- **Docker Desktop** 실행 후, 터미널에서 저장소 **`eye-project` 루트**로 이동해 `./setup.sh` 한 번이면 됩니다.
- **Apple Silicon(M1/M2/M3)** 은 스크립트가 자동으로 `docker-compose-mac.yml` 을 붙입니다. (기본 백엔드 이미지는 ARM용 CUDA 휠이 맞지 않아 빌드가 실패할 수 있음 — 터미널에 나온 것과 같은 이유입니다.)
- 브라우저는 준비되면 `http://127.0.0.1:8080` 을 엽니다.

**Windows**

- **Docker Desktop**(WSL2 백엔드 권장)을 켠 뒤, **Git Bash**에서 저장소 `eye-project` 루트로 이동해 `./setup.sh` 를 실행합니다. (CMD만 쓰는 경우 Bash가 없으므로 Git Bash 설치가 필요합니다.)
- 브라우저 자동 열기는 Git Bash에서 `cmd.exe` 경로가 잡힐 때 `start` 로 열립니다.

**공통**

- `setup.sh` 는 `docker compose up --build -d` 까지 포함합니다. 다시 실행해도 compose 를 재기동하고, 8080이 뜨면 브라우저를 엽니다.
- **종료**: 맥 Apple Silicon은  
  `docker compose -f docker-compose.yml -f docker-compose-mac.yml down`  
  그 외는 `docker compose down`  
  (`setup.sh` 성공 시 터미널에도 같은 안내가 출력됩니다.)

---

## 주요 API 엔드포인트

| 메서드 | 경로 | 설명 |
|---|---|---|
| `GET` | `/health` | 모델 로드 상태 확인 |
| `POST` | `/predict` | 빠른 추론 (분류 결과만 반환) |
| `POST` | `/analyze` | 전체 분석 (품질 검사 + GradCAM + 리포트) |
| `GET` | `/storage/<path>` | 업로드 이미지 조회 |
| `GET` | `/results/<path>` | 생성된 리포트 조회 |

---

## 데이터셋 출처

학습 및 평가에 사용된 공개 데이터셋에 대한 저작권 표기.

| 데이터셋 | 출처 | 라이선스 |
|---|---|---|
| APTOS 2019 | [Kaggle — APTOS 2019 Blindness Detection](https://www.kaggle.com/c/aptos2019-blindness-detection) (Aravind Eye Hospital) | Kaggle Competition Terms |
| IDRiD | [IEEE DataPort — Indian Diabetic Retinopathy Image Dataset](https://ieee-dataport.org/open-access/indian-diabetic-retinopathy-image-dataset-idrid) | CC BY 4.0 |
| Messidor | [ADCIS — Messidor](https://www.adcis.net/en/third-party/messidor/) | 비상업적 연구 목적 |
| DDR | [GitHub — nkicsl/DDR-dataset](https://github.com/nkicsl/DDR-dataset) | 연구 목적 |

---

## 환경구축

1. 저장소 클론 후 `./setup.sh` 실행 (Flutter 준비 + `docker compose up --build -d` + 브라우저 열기)
2. Docker Desktop(또는 동등한 환경)이 실행 중인지 확인

