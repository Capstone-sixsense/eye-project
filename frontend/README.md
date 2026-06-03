# Eye Project · Frontend

**최종 갱신: 2026-05-30**

> `eye-project/frontend/` — **frontend** 브랜치 Flutter 클라이언트.  
> 망막 이미지 업로드 → 백엔드 분석 API → 결과·이력·PDF를 담당합니다. AI·FastAPI·Docker 설정은 저장소 루트를 참고하세요.

---

## 목차

1. [개요](#개요)
2. [기술 스택](#기술-스택)
3. [폴더 구조](#폴더-구조)
4. [실행 방법](#실행-방법)
5. [백엔드 API 연동](#백엔드-api-연동)
6. [분석·진행 UI](#분석진행-ui)
7. [화면·라우팅](#화면라우팅)
8. [모듈 요약](#모듈-요약)
9. [테스트](#테스트)
10. [배포·플랫폼](#배포플랫폼)

---

## 개요

- **대상**: 당뇨병성 망막병증 보조 스크리닝 MVP (의료 **보조** 용도, 최종 판단은 전문가).
- **UI**: Material 3 + `MedicalTokens` 기반 의료용 톤(한국어 문구).
- **주요 플로우**: 이미지 선택 → 비동기 `/analyze` → 진행 다이얼로그 → 결과 화면 → (선택) PDF 공유 / 이력 조회.
- **기본 API**: `http://127.0.0.1:8000` (`api_config.dart`, Docker Compose 백엔드와 맞춤).

---

## 기술 스택

| 구분 | 내용 |
|------|------|
| 프레임워크 | Flutter · Dart SDK `^3.11.3` |
| HTTP | `http` — 멀티파트 업로드, job 폴링, JSON |
| 파일 | `file_picker` — jpg/jpeg/png, 최대 10MB |
| PDF | `pdf`, `printing` — 결과 리포트 PDF 생성·공유 |
| UI | Material 3, 공통 위젯 `lib/ui/medical_ui.dart` |

---

## 폴더 구조

```text
frontend/
├── lib/
│   ├── main.dart                 # 테마·라우트 (/upload, /history, /result)
│   ├── api/
│   │   └── eye_api_client.dart   # REST 클라이언트
│   ├── config/
│   │   └── api_config.dart       # baseUrl, baseUri, 이미지 URL resolve
│   ├── constants/
│   │   └── api_error_codes.dart
│   ├── models/
│   │   ├── analyze_response.dart
│   │   ├── analyze_job_status.dart
│   │   ├── analysis_history_entry.dart
│   │   ├── report_metrics.dart
│   │   ├── result_screen_args.dart
│   │   └── server_log_entry.dart
│   ├── screens/
│   │   ├── upload_screen.dart    # 업로드·분석 시작
│   │   ├── result_screen.dart    # 판정·XAI·PDF
│   │   └── history_screen.dart   # 이력 대시보드·필터·삭제
│   ├── util/
│   │   └── format.dart           # 날짜·타임스탬프 표시 (이력·로그 공통)
│   └── ui/
│       ├── medical_ui.dart
│       ├── notice_dialog.dart
│       ├── analyze_progress_dialog.dart
│       ├── analyze_progress_controller.dart
│       ├── report_metrics_dialog.dart
│       ├── server_logs_dialog.dart
│       └── dialog_keyboard.dart
├── test/
├── web/, linux/, windows/         # 플랫폼 러너 (개발·Web Docker 빌드)
├── frontend_Dockerfile
├── docker-web.nginx.conf
└── pubspec.yaml
```

---

## 실행 방법

### 전체 스택 (권장, 저장소 루트)

```bash
cd eye-project
./setup.sh
# 또는: docker compose up -d
```

- UI(Web): `http://localhost:8080` (nginx + `flutter build web`)
- API: `http://localhost:8000`

### Flutter만 (로컬 UI 개발)

```bash
cd frontend
flutter pub get
flutter run -d chrome   # Web
# flutter run -d macos  # macOS 데스크톱 (Xcode 필요)
```

백엔드 주소 변경:

```bash
flutter run --dart-define=API_BASE_URL=http://127.0.0.1:8000
```

- `localhost` 대신 **`127.0.0.1`** 권장 (Windows·Docker IPv6 이슈 회피).

---

## 백엔드 API 연동

| 메서드 | 경로 | 프론트 사용처 |
|--------|------|----------------|
| `GET` | `/health` | (간접) 서버 기동 확인 |
| `GET` | `/deploy-metric` | 업로드 화면 「성능 지표 보기」 |
| `POST` | `/analyze` | 이미지 업로드 → **202** + `job_id` |
| `GET` | `/analyze/jobs/{job_id}` | 분석 진행 폴링 (`progress`, `phase`, `result`) |
| `GET` | `/history` | 이력 목록 (페이지네이션) |
| `DELETE` | `/history/{id}` | 이력 삭제 |
| `GET` | `/logs` | 업로드 화면 「서버 로그」 (레벨 필터·무한 스크롤) |
| `GET` | `/image/raw/{id}`, `/image/report/{id}` | `ApiConfig.resolveAssetUrl` |

- 멀티파트 필드명: **`image`** (백엔드와 동일).
- 분석 대기: `EyeApiClient.analyzeTimeout` 기본 **20분** (CPU 추론).
- 오류 본문: `detail.message`, `detail.code` (`not_fundus_image`, `low_image_quality` 등) 파싱 → `EyeApiException` / `showErrorNotice`.

---

## 분석·진행 UI

백엔드는 단계마다 `progress`·`phase`를 점프시키므로, 화면은 **표시용 진행률**을 별도로 보간합니다.

```text
POST /analyze (202, job_id)
    → GET /analyze/jobs/{id} (400ms 폴링)
    → AnalyzeProgressController (50ms 틱, phase별 상한·보간)
    → AnalyzeProgressDialog (N%, 단계 문구)
    → 서버 완료 후 visual 100% + 0.3초 정지
    → /result 이동
```

| `phase` (서버) | UI 문구 (예) |
|----------------|--------------|
| `upload` | 이미지 확인 중 |
| `fundus_check` | 안저 이미지 검증 중 |
| `quickqual` | 이미지 품질 평가 중 |
| `inference` | AI 분석 중 |
| `report` | 리포트 생성 중 |
| `done` | 완료 |

구현 파일: `analyze_progress_controller.dart`, `analyze_progress_dialog.dart`, `upload_screen.dart`.  
기본 단계 문구는 `AnalyzeProgressDialog.defaultMessage` 한 곳에서 관리합니다.

---

## 화면·라우팅

```text
/upload (UploadScreen)
  ├─ 이미지 선택·미리보기·10MB/확장자 검증
  ├─ 업로드 및 분석 → 진행 다이얼로그
  ├─ 서버 로그 (`GET /logs`, `server_logs_dialog.dart`)
  └─ 성능 지표 (`GET /deploy-metric`)

/result (ResultScreen)
  ├─ 원본·설명(heatmap) 이미지, 판정·확률·품질
  ├─ PDF 생성·공유 (printing)
  ├─ 다시 업로드 → /upload (스택 제거)
  └─ 이력 보기 → /history

/history (HistoryScreen)
  ├─ GET /history 목록·무한 스크롤
  ├─ 대시보드 / 목록 뷰, 판정·기간 필터
  ├─ 선택 삭제 (DELETE)
  └─ 항목 탭 → /result (저장 메타 + 네트워크 이미지)
```

`main.dart`의 `/result` 인자:

- **`ResultScreenArgs`** — `originalImageBytes` + `AnalyzeResponse` (일반 분석 직후).
- **`Uint8List`만** — 구 호환(원본만).
- 인자 없음 — 빈 결과 화면.

---

## 모듈 요약

| 경로 | 역할 |
|------|------|
| `api_config.dart` | `baseUrl`·`baseUri`(API 요청 베이스), `resolveAssetUrl` |
| `eye_api_client.dart` | analyze job, history, logs, deploy-metric, 오류 파싱 |
| `analyze_response.dart` | 분석·이력 JSON, `canShowInferenceResults`, XAI 경로 |
| `analyze_job_status.dart` | job 폴링 상태·`phaseLabel` |
| `analysis_history_entry.dart` | 이력 행 파싱 |
| `server_log_entry.dart` | 서버 로그 행 파싱·`phaseLabel` |
| `report_metrics.dart` | AUROC·민감도 등 배포 지표 |
| `util/format.dart` | `formatLocalDateTime`, `formatIsoTimestamp` |
| `medical_ui.dart` | 카드·버튼·배너·토큰 |
| `notice_dialog.dart` | `showNoticeDialog`·`showCodeNoticeDialog`·`showErrorNotice` |
| `report_metrics_dialog.dart` | 성능 지표 설명 UI |
| `server_logs_dialog.dart` | 서버 로그 조회·레벨 필터 UI |

---

## 테스트

```bash
cd frontend
flutter analyze
flutter test
```

- `test/widget_test.dart`는 업로드 화면 빌드 스모크 테스트입니다. AppBar 문구가 `망막 이미지 분석`으로 바뀐 경우 테스트 문자열을 맞춰 주세요.

---

## 배포·플랫폼

| 경로 | 용도 |
|------|------|
| `frontend_Dockerfile` | `flutter build web --release` + nginx (`:8080`) |
| `docker-web.nginx.conf` | SPA `try_files` |
| `web/` | Flutter Web 엔트리 (Compose 프론트) |
| `linux/`, `windows/` | 데스크톱 러너 (로컬 `flutter run`) |

단일 `.app`/`.exe`에 백엔드·모델까지 묶는 **오프라인 배포**는 이 README 범위 밖이며, 별도 패키징 절차가 필요합니다.

---

## 관련 문서

- 저장소 루트 `README.md` — Docker Compose, 브랜치 구조, API 목록
- `.cursor/rules/eye-project.mdc` — 팀 컨벤션·AI 연동 주의사항
