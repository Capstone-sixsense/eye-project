# Eye Project · Backend

**최종 갱신: 2026-06-11**

> `eye-project/backend/` — **backend** 브랜치 FastAPI 서버.  
> 망막 이미지 수신 → 안저 검증 → 품질 평가 → AI 추론 → 결과·이력 저장을 담당합니다. Flutter 클라이언트·AI 모델 설정은 저장소 루트를 참고하세요.

---

## 목차

1. [개요](#개요)
2. [기술 스택](#기술-스택)
3. [폴더 구조](#폴더-구조)
4. [실행 방법](#실행-방법)
5. [API 엔드포인트](#api-엔드포인트)
6. [분석 파이프라인](#분석-파이프라인)
7. [저장 구조 & 보안](#저장-구조--보안)
8. [모듈 요약](#모듈-요약)
9. [환경변수](#환경변수)
10. [배포·플랫폼](#배포플랫폼)

---

## 개요

- **대상**: 당뇨병성 망막병증 보조 스크리닝 MVP (의료 **보조** 용도, 최종 판단은 전문가).
- **주요 역할**: 이미지 수신 → 안저 휴리스틱 검증 → QuickQual 품질 평가 → AI 진단 추론 → 결과·이력 저장.
- **비동기 분석**: `POST /analyze` 202 응답 후 `GET /analyze/jobs/{job_id}` 폴링으로 진행 상태 확인.
- **암호화 저장**: 원본·결과 이미지 AES-256-GCM 암호화 보관 (`IMAGE_ENCRYPTION_KEY`).
- **기본 포트**: `8000` (Docker Compose 기준).

---

## 기술 스택

| 구분 | 내용 |
|------|------|
| 프레임워크 | FastAPI + Uvicorn (Python 3.11) |
| AI 추론 | drscreen `InferenceSession` (`/ai/configs/base.yaml`) |
| 품질 평가 | QuickQual — DenseNet121 features + sklearn SVM |
| 안저 검증 | `fundus_checker.py` — 휴리스틱 필터 (테두리·색조·형태) |
| 저장소 | SQLite (`storage/history.db`) + 암호화 파일 (`storage/`, `results/`) |
| 암호화 | AES-256-GCM (`cryptography` 라이브러리) |
| 이미지 처리 | Pillow, NumPy |
| 컨테이너 | Docker (`backend_Dockerfile`) + Docker Compose |

---

## 폴더 구조

```text
backend/
├── main.py                      # FastAPI 앱, 엔드포인트, 분석 job 관리
├── history.py                   # SQLite CRUD, 암호화 파일 저장/조회
├── crypto.py                    # AES-256-GCM 암복호화 유틸리티
├── fundus_checker.py            # 안저 이미지 휴리스틱 판별
├── make_result_img.py           # 분석 결과 이미지 PNG 직렬화
├── migration.py                 # 구형 .json.enc → SQLite 마이그레이션
├── models/
│   ├── quickqual_wrapper.py     # QuickQual 전처리 + 품질 평가
│   └── quickqual_dn121_512.pkl  # SVM 가중치 (DenseNet121 features)
├── QuickQual/                   # QuickQual 원본 레포 참조 코드
├── storage/
│   ├── history.db               # SQLite DB (메타데이터 + 로그)
│   └── raw_<id>.<ext>.enc       # 원본 이미지 (AES-256-GCM 암호화)
├── results/
│   └── <YYYY-MM-DD>/
│       └── report_<id>.png.enc  # 분석 결과 이미지 (암호화)
├── backend_requirements         # pip 의존성
└── backend_Dockerfile           # Docker 빌드 파일
```

---

## 실행 방법

### 전체 스택 (권장, 저장소 루트)

```bash
cd eye-project
docker compose up -d
```

- API: `http://localhost:8000`
- UI(Web): `http://localhost:8080`

### 백엔드만 (로컬 개발)

```bash
cd backend
pip install -r backend_requirements
IMAGE_ENCRYPTION_KEY=<base64-32bytes> uvicorn main:app --reload
```

암호화 키 생성:

```bash
python -c "import os, base64; print(base64.b64encode(os.urandom(32)).decode())"
```

> `.env` 파일에 `IMAGE_ENCRYPTION_KEY=<생성된 값>` 을 설정하면 Docker Compose가 자동으로 주입합니다.

---

## API 엔드포인트

| 메서드 | 경로 | 설명 |
|--------|------|------|
| `GET` | `/` | 서버 기동 확인 |
| `GET` | `/health` | 모델 준비 상태 (`diagnosis_model`, `quickqual`) |
| `GET` | `/deploy-metric` | 배포된 AI 모델 평가 지표 (AUROC 등) |
| `POST` | `/analyze` | 이미지 업로드 → **202** + `job_id` 반환 |
| `GET` | `/analyze/jobs/{job_id}` | 분석 진행 폴링 (`status`, `progress`, `phase`, `result`) |
| `GET` | `/logs` | 서버 로그 목록 (레벨·job_id 필터, 페이지네이션) |
| `GET` | `/history` | 분석 이력 목록 (최신순, 페이지네이션) |
| `GET` | `/history/{record_id}` | 특정 이력 단건 조회 |
| `DELETE` | `/history/{record_id}` | 이력 + 관련 파일 일괄 삭제 |
| `GET` | `/image/raw/{record_id}` | 원본 이미지 스트리밍 (복호화 후 응답) |
| `GET` | `/image/report/{record_id}` | 분석 결과 이미지 스트리밍 (PNG) |

- 멀티파트 필드명: **`image`** (프론트와 동일).
- 오류 본문: `{"detail": {"code": "...", "message": "..."}}` 형식 (`not_fundus_image`, `low_image_quality` 등).

---

## 분석 파이프라인

```text
POST /analyze (202, job_id)
    → upload      : 이미지 디코딩·유효성 확인
    → fundus_check: 안저 이미지 휴리스틱 검증 (비안저 → 422)
    → quickqual   : QuickQual 품질 평가 (bad → 422 / usable → 경고 포함)
    → inference   : drscreen InferenceSession.predict_image_bytes (AI 추론)
    → report      : GradCAM 오버레이 PNG 생성, raw + report 암호화 저장
    → done        : SQLite 메타데이터 저장, job result 세팅
```

| `phase` | 진행률 범위 | 설명 |
|---------|------------|------|
| `upload` | 0 → 5% | 이미지 디코딩 |
| `fundus_check` | 5 → 10% | 안저 검증 |
| `quickqual` | 10 → 30% | 품질 평가 + 전처리 |
| `inference` | 30 → 85% | AI 추론 |
| `report` | 85 → 90% | 결과 이미지 생성·저장 |
| `done` | 100% | 완료 |

- 분석 job은 완료 10분 후 메모리에서 만료됩니다 (`_expire_job`).

---

## 저장 구조 & 보안

| 경로 | 내용 |
|------|------|
| `storage/history.db` | SQLite — `records` 테이블(메타데이터 암호화 BLOB) + `logs` 테이블 |
| `storage/raw_<id>.<ext>.enc` | 원본 이미지 (AES-256-GCM 암호화) |
| `results/<YYYY-MM-DD>/report_<id>.png.enc` | 분석 결과 이미지 (AES-256-GCM 암호화) |

- `<id>` 형식: `YYYYMMDD_HHMMSS_mmm` (KST 기준 타임스탬프).
- 암호화 키: 환경변수 `IMAGE_ENCRYPTION_KEY` (base64 인코딩 32바이트).
- Associated Data로 `record_id`를 묶어 파일 swap 공격 방어.
- 로그는 30일 경과 시 자동 삭제.

---

## 모듈 요약

| 파일 | 역할 |
|------|------|
| `main.py` | FastAPI 앱·엔드포인트·lifespan(모델 로드)·job 관리 |
| `history.py` | SQLite CRUD (`records`, `logs`), 암호화 파일 저장/조회/삭제 |
| `crypto.py` | AES-256-GCM `encrypt_bytes`/`decrypt_bytes`, 파일 헬퍼 |
| `fundus_checker.py` | 안저 이미지 휴리스틱 판별 (테두리·종횡비·색조) |
| `make_result_img.py` | PIL 이미지 → PNG 바이트 직렬화 |
| `migration.py` | 구형 `.json.enc` 파일 → SQLite DB 일괄 마이그레이션 |
| `models/quickqual_wrapper.py` | `preprocess_fundus_image` + `QuickQualWrapper.preprocess_and_score` |

---

## 환경변수

| 변수 | 기본값 | 설명 |
|------|--------|------|
| `IMAGE_ENCRYPTION_KEY` | (필수) | base64 인코딩 32바이트 AES 키 |
| `FUNDUS_CONFIG_PATH` | `/ai/configs/base.yaml` | AI 모델 설정 파일 경로 |
| `FUNDUS_CHECKPOINT_PATH` | (선택) | AI 체크포인트 경로 오버라이드 |
| `FUNDUS_CHECK_ENABLED` | `true` | 안저 이미지 필터 활성화 여부 |
| `QUICKQUAL_SVM_FILENAME` | `quickqual_dn121_512.pkl` | QuickQual SVM 가중치 파일명 |
| `QUICKQUAL_BAD_THRESHOLD` | `0.7` | bad 확률 이 값 이상이면 품질 경고 (usable 등급) |
| `CORS_ALLOW_ORIGINS` | `localhost:8080,...` | CORS 허용 출처 (`,` 구분) |
| `TORCH_NUM_THREADS` | `8` | PyTorch 추론 스레드 수 |

---

## 배포·플랫폼

| 파일 | 용도 |
|------|------|
| `backend_Dockerfile` | Python 3.11-slim + 의존성 설치, `uvicorn` 기동 |
| `docker-compose.yml` (루트) | backend + frontend 통합 실행, hot-reload 개발 모드 |
| `storage/`, `results/` | 컨테이너 재시작 시에도 데이터 유지 (호스트 마운트) |

- `platform: linux/amd64` 고정 — Apple Silicon(arm64)에서도 PyTorch 바이너리 호환 보장.
- 프로덕션: `docker-compose.yml`의 `command` 블록 제거 → Dockerfile CMD(`uvicorn` 무reload) 사용.

---

## 관련 문서

- 저장소 루트 `README.md` — Docker Compose, 브랜치 구조, API 목록
- `ai/` 브랜치 `ai/README.md` — AI 모델 아키텍처·학습·추론 파이프라인
- `frontend/` 브랜치 `frontend/README.md` — Flutter 클라이언트 연동 가이드
