# eye-project

**최종 갱신: 2026-05-25**

안저(眼底) 이미지 기반 **당뇨병성 망막병증 보조 스크리닝** MVP입니다.  
이미지를 업로드하면 AI가 이상 여부를 추정하고, 설명(GradCAM·리포트)과 품질 정보를 함께 제공합니다.

> **의료 보조 용도** — 최종 판단은 반드시 의료 전문가가 합니다.  
> 설계·운영은 **정확성·재현성·설정 일관성**을 속도보다 우선합니다.

---

## 시스템 한눈에

```text
[Flutter Web UI]  :8080
        │  HTTP
        ▼
[FastAPI Backend] :8000
        │  import (PYTHONPATH=/ai)
        ▼
[drscreen · PyTorch]  configs · checkpoints · QuickQual
```

| 컴포넌트 | 역할 | 기술 |
|----------|------|------|
| **Frontend** | 업로드·진행·결과·이력·PDF | Flutter (Web / Docker nginx) |
| **Backend** | API, 이력·암호화 저장, QuickQual, 추론 호출 | FastAPI, Uvicorn |
| **AI** | 학습·추론·XAI·설정·체크포인트 | PyTorch, `drscreen` 패키지 |

---

## 브랜치 · 상세 문서

팀별 개발 브랜치에서 작업 후 `main`에 머지합니다. **자세한 내용은 각 브랜치의 README·문서를 보세요.**

| 브랜치 | 보는 것 | 문서로 이동 |
|--------|---------|-------------|
| [`frontend`](https://github.com/Capstone-sixsense/eye-project/tree/frontend) | Flutter 클라이언트, API 연동, 진행 UI | [`frontend/README.md`](https://github.com/Capstone-sixsense/eye-project/blob/frontend/frontend/README.md) |
| [`backend`](https://github.com/Capstone-sixsense/eye-project/tree/backend) | REST API, `storage/`·`results/`, Docker | `backend/` 디렉터리 · [`main.py`](https://github.com/Capstone-sixsense/eye-project/blob/backend/backend/main.py) |
| [`ai`](https://github.com/Capstone-sixsense/eye-project/tree/ai) | 모델·학습·추론·`configs/`·`artifacts/` | [`ai/AGENTS.md`](https://github.com/Capstone-sixsense/eye-project/blob/ai/ai/AGENTS.md) · `ai/docs/` |

로컬에서 브랜치 전환:

```bash
git fetch origin
git checkout frontend   # 또는 backend, ai
```

`main`에는 통합 실행용 `docker-compose.yml`·`setup.sh`가 있습니다.

---

## 빠른 시작 (통합 실행)

```bash
git clone https://github.com/Capstone-sixsense/eye-project.git
cd eye-project
./setup.sh
# 또는
docker compose up -d
```

| 서비스 | URL |
|--------|-----|
| UI | http://localhost:8080 |
| API | http://localhost:8000 |
| 헬스 | http://localhost:8000/health |

- AI 체크포인트는 Git LFS일 수 있습니다. 수백 바이트 포인터만 보이면 `git lfs install && git lfs pull`.
- 추론 기본 설정: `ai/configs/base.yaml` (Compose: `FUNDUS_CONFIG_PATH=/ai/configs/base.yaml`).

---

## 주요 API (요약)

| 메서드 | 경로 | 설명 |
|--------|------|------|
| `GET` | `/health` | 모델·QuickQual 준비 상태 |
| `POST` | `/analyze` | 분석 요청 (비동기 job, 202 + `job_id`) |
| `GET` | `/analyze/jobs/{id}` | 분석 진행·결과 |
| `GET` | `/history` | 분석 이력 |
| `GET` | `/deploy-metric` | 배포용 eval 지표 |
| `GET` | `/image/raw/{id}`, `/image/report/{id}` | 복호화 이미지 |

상세 스키마·오류 코드는 **backend** 브랜치 코드·프론트 `eye_api_client`와 맞춰 확인하세요.

---

## 저장소 레이아웃 (`main` 기준)

```text
eye-project/
├── ai/                 # drscreen 패키지, configs, artifacts
├── backend/            # FastAPI, QuickQual 래퍼
├── frontend/           # Flutter
├── docker-compose.yml
├── setup.sh
└── README.md           # 이 파일
```

---

## 데이터셋 (학습·평가 출처)

APTOS 2019, IDRiD, Messidor, DDR 등 공개 안저 데이터셋을 사용합니다. 라이선스·표기는 팀 문서 및 논문/시연 자료를 따릅니다.

---

## 기여·규칙

- 컴포넌트별 작업: 해당 브랜치에서 PR → `main` 머지.
- `ai/configs/base.yaml` 변경 시 Docker 추론·백엔드 응답 필드와 **함께** 검증.
- Cursor/에이전트 규칙: [`.cursor/rules/eye-project.mdc`](.cursor/rules/eye-project.mdc)

---

## 더 보기

- **프론트**: `git checkout frontend` → [`frontend/README.md`](frontend/README.md)
- **백엔드**: `git checkout backend` → `backend/main.py`, `docker-compose.yml`
- **AI**: `git checkout ai` → `ai/drscreen/`, `ai/configs/`, `ai/docs/`
