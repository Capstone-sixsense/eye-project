# eye-project

안저(眼底) 이미지 기반 당뇨병성 망막병증 보조 판별 AI 시스템. 이미지를 업로드하면 AI가 이상 여부를 분류하고, 병변 후보 영역 overlay와 의료 리포트를 함께 제공한다. 본 시스템의 결과는 의료 전문가의 판단을 보조하는 용도로만 사용되어야 한다.

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
| 출력 | 이상 여부 분류, 이상 확률, 병변 후보 영역 overlay, 의료 리포트 |
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
  - v31 classifier + v8b lesion segmenter 추론
  - numeric meta-classifier score fusion
  - 병변 후보 영역 overlay 생성
  - 의료 리포트 생성
```

---

## 기술 스택

| 컴포넌트 | 기술 |
|---|---|
| AI | PyTorch, EfficientNet-B5, ResNet50 U-Net lesion segmenter, numeric LogReg fusion, albumentations |
| Backend | FastAPI, Uvicorn, OpenCV, cleanvision |
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
| `POST` | `/analyze` | 전체 분석 (품질 검사 + 병변 evidence overlay + 리포트) |
| `GET` | `/storage/<path>` | 업로드 이미지 조회 |
| `GET` | `/results/<path>` | 생성된 리포트 조회 |

---

## 데이터셋 출처

학습 및 평가에 사용된 공개 데이터셋에 대한 저작권 및 credit 표기.
현재 배포 기준 `v31_v8b_fusion_v2`의 병변 feature는 `seg_evidence_v8b_ddrseg_tjdr_maplesfix` 모델에서 생성되며, 이 병변 feature 학습에는 `IDRiD`, `MAPLES-DR`, `TJDR`, `DDR_SEG`가 사용되었다.
데이터 원본은 저장소에 포함하지 않으며, 다운로드와 재사용은 각 원 배포처의 라이선스 및 이용 조건을 따른다.

| 데이터셋 | 이 프로젝트에서의 사용 | 출처 / 인용 | 라이선스 / 이용 조건 |
|---|---|---|---|
| APTOS 2019 | DR 분류 학습 보조 | [Kaggle — APTOS 2019 Blindness Detection](https://www.kaggle.com/c/aptos2019-blindness-detection) (Aravind Eye Hospital) | Kaggle Competition Terms |
| IDRiD | DR 분류, XAI 평가, v8b 병변 feature 학습(MA/HE/EX/SE mask) | [IDRiD Grand Challenge](https://idrid.grand-challenge.org/Data/) / [IEEE DataPort — Indian Diabetic Retinopathy Image Dataset](https://ieee-dataport.org/open-access/indian-diabetic-retinopathy-image-dataset-idrid) | CC BY 4.0 |
| Messidor | DR 분류 학습 및 MAPLES-DR 원본 fundus image | [ADCIS — Messidor](https://www.adcis.net/en/third-party/messidor/) | 비상업적 연구 목적 |
| MAPLES-DR | v8b 병변 feature 학습 및 평가(MA/HE/EX/CWS mask) | [LIV4D/MAPLES-DR](https://github.com/LIV4D/MAPLES-DR), [Scientific Data 2024](https://www.nature.com/articles/s41597-024-03739-6) | MAPLES-DR label/code repo: CC0-1.0. 논문: CC BY-NC-ND 4.0. 원본 fundus image는 Messidor 이용 조건을 따름 |
| TJDR | v8b 병변 feature 학습(MA/HE/EX/SE pixel-level mask) | [TJDR dataset page](https://www.juheapi.com/datasets/tjdr), [arXiv:2312.15389](https://arxiv.org/abs/2312.15389) | 공개 연구용 데이터셋. 명시 라이선스 파일은 로컬 데이터와 공개 페이지에서 확인되지 않았으므로 원 배포처 조건 및 논문 인용 필요 |
| DDR / DDR_SEG | DR 분류 평가 및 v8b 병변 feature 학습용 lesion segmentation subset | [GitHub — nkicsl/DDR-dataset](https://github.com/nkicsl/DDR-dataset) | 배포 repo: MIT License. README의 DDR 논문 인용 요구 및 원 배포처 조건 준수 |

TJDR 인용:

```bibtex
@article{mao2023tjdr,
  title={TJDR: A High-Quality Diabetic Retinopathy Pixel-Level Annotation Dataset},
  author={Mao, Jingxin and Ma, Xiaoyu and Bi, Yanlong and Zhang, Rongqing},
  journal={arXiv preprint arXiv:2312.15389},
  year={2023}
}
```

---

## 환경구축

1. git clone을 통해서 setup.sh 실행
2. docker hub를 이용 → 폴더 생성 후 docker-compose.yml 파일을 위치시킨 후 `docker compose up -d` 명령어 입력


##테스트시 result 폴더 비우기
rm -rf backend/storage/raw_* backend/results/2026-*

