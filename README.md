# eye-project

당뇨망막병증(Diabetic Retinopathy) 스크리닝 시스템. 안저(眼底) 이미지를 업로드하면 AI가 이상 여부를 분류하고, GradCAM 히트맵과 의료 리포트를 함께 제공한다.

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
| 목적 | 안저 이미지 기반 당뇨망막병증 자동 스크리닝 |
| 출력 | 이상 여부 분류, 이상 확률, GradCAM 히트맵, 의료 리포트 |
| 배포 | Docker Compose (로컬 단일 명령 실행) |

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

# 2. 디렉토리 및 의존성 초기화
./setup.sh

# 3. 서비스 실행
docker compose up -d
```

실행 후 브라우저에서 `http://localhost:8080` 접속.

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

## 환경구축

1. git clone을 통해서 setup.sh 실행
2. docker hub를 이용 → 폴더 생성 후 docker-compose.yml 파일을 위치시킨 후 `docker compose up -d` 명령어 입력
