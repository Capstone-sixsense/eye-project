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
│   ├── screens/              # 화면 위젯
│   ├── state/                # 세션 상태 저장소
│   └── ui/                   # 공통 디자인 토큰/컴포넌트
├── test/                     # 위젯 테스트
├── pubspec.yaml              # 패키지명·의존성
├── analysis_options.yaml     # 분석/린트
├── web/, linux/, windows/    # 플랫폼 러너 (Flutter 기본)
└── frontend_Dockerfile*      # 컨테이너 빌드용
```

---

## 실행 방법 (`setup.sh` 권장)

```bash
cd eye-project
./setup.sh
```

실행 후 브라우저에서 `http://127.0.0.1:8080`으로 접속합니다.

종료는 실행 환경에 맞게 아래 중 하나를 사용합니다.

```bash
# 기본 compose
docker compose down

# Apple Silicon(Mac) 오버라이드 사용 시
docker compose -f docker-compose.yml -f docker-compose-mac.yml down
```

- 참고: Flutter 단독 실행도 가능하지만, 팀 공통 검증/시연은 `setup.sh` 흐름을 기준으로 합니다.

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
| `ui/medical_ui.dart` | 공통 디자인 토큰/컴포넌트를 전역 테마에 적용. |

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

### `lib/models/analysis_history_entry.dart`

| 항목 | 설명 |
|------|------|
| **역할** | 분석 이력 1건을 표현하는 데이터 모델입니다. |
| **필드** | `filename`, `originalImageBytes`, `response`, `createdAt`. |

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
| **`build`** | 앱바 `망막 이미지 분석`, 카드 기반 업로드 UI, 파일명/용량 배지, 미리보기, `이미지 선택`/`업로드 및 분석` 버튼 구성. |

---

### `lib/screens/result_screen.dart`

| 항목 | 설명 |
|------|------|
| **역할** | 분석 **결과 시각화**: 상단 고지문, 원본/설명 이미지, 판정 카드, 리포트 지표, 이상 확률 카드, 하단 액션. |
| **생성자** | `args` 우선, 없으면 `originalImageBytes`만(구 라우트). |
| **레이아웃** | 상단 고정 고지문 아래에 원본/설명 이미지를 먼저 배치하고, 이후 `분석 요약 → 성능 지표 → 이상 확률` 순서로 정보를 노출합니다. |
| **`_ImageBox` / `_kResultImageMaxHeight`** | 이미지 타일 최대 높이 300, 공통 카드 스타일로 높이/외곽을 통일. |
| **`_JudgmentCard`** | 응답 없음/실패/전처리 미통과 시 안내 문구, 성공·추론 표시 가능 시 `label` 크게 표시. |
| **`_ReportMetricsCard`** | `Accuracy`, `Precision`, `Sensitivity`, `Specificity`, `F1-score`를 배지 형태로 표시합니다. |
| **`_ProbabilityCard`** | 백엔드 `abnormal_probability`를 이상 확률 배지로 표시합니다. |
| **`_ExplanationPanel`** | `shouldShowExplanationFailure`면 실패 문구+`xaiErrorCode`(없으면 `XAI_001`). URL 있으면 `Image.network`+로딩·에러(CORS 등) 처리. 없으면 placeholder 카드. |
| **`_MedicalDisclaimerBanner`** | "본 결과는 의료적 확정 진단이 아닌 보조 판별 결과입니다." 고정 안내 배너를 표시합니다. |
| **`_InfoCard`** | 공통 카드 래퍼(`MedicalCard`). |
| **하단 버튼** | `다시 업로드` — `/upload`까지 스택 제거. `이력 보기` — `/history` push. |

---

### `lib/screens/history_screen.dart`

| 항목 | 설명 |
|------|------|
| **역할** | 세션 기반 **분석 이력 리스트**를 표시하고 결과 화면으로 재진입합니다. |
| **현재 동작** | `AnalysisHistoryStore.entries`를 카드 리스트로 렌더링, 상단 휴지통으로 전체 삭제, 항목 탭 시 `/result`로 원본 이미지+응답을 전달합니다. |

---

### `lib/state/analysis_history_store.dart`

| 항목 | 설명 |
|------|------|
| **역할** | 세션 메모리 기반 이력 저장소(static)입니다. |
| **API** | `entries`(읽기 전용), `add`, `clear`. |

---

### `test/widget_test.dart`

| 항목 | 설명 |
|------|------|
| **역할** | Flutter 기본 **위젯 테스트** — `UploadScreen`이 깨지지 않고 빌드되는지 검증. |
| **테스트 내용** | `MaterialApp(home: UploadScreen())`을 pump한 뒤 업로드 화면이 정상 렌더링되는지 확인합니다. |
| **import** | 패키지명은 `pubspec.yaml`의 `name: eye_project`에 맞춰 `package:eye_project/screens/upload_screen.dart` 사용. |

---

## 화면·라우팅 흐름

1. 앱 시작 → **`/upload`** (`UploadScreen`).
2. **Upload**로 분석 성공 → **`ResultScreenArgs`**를 `arguments`에 실어 **`/result`** (`ResultScreen`).
3. **`/result`**에서 **다시 업로드** → 스택을 비우고 **`/upload`** 로 복귀.
4. **`/result`**에서 **이력 보기** → **`/history`** (`HistoryScreen`, 세션 이력 리스트).
5. **`/history`**에서 항목 선택 시 해당 응답으로 **`/result`**를 다시 열어 재확인.

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