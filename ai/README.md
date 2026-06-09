# Eye Project · AI (`drscreen`)

**최종 갱신: 2026-06-03 (Sprint 5 CLOSED)**

> `eye-project/ai/` — 안저(眼底) 이미지 한 장으로 당뇨병성 망막병증(DR)을 **Normal / Abnormal** 으로 스크리닝하는 `drscreen` 모듈.
> 분류 확률 + 병변 evidence overlay + 외부 평가 지표를 산출해 백엔드에 넘깁니다. 학습·추론·XAI·설정·체크포인트가 하나의 Python 패키지에 모여 있습니다.
> Flutter UI·FastAPI 백엔드·Docker 통합 실행은 저장소 **루트 `README.md`** 참고.

> **의료 보조 용도** — 최종 판단은 반드시 의료 전문가가 합니다.
> 병변 evidence는 "분류 결과 + 별도로 탐지된 병변 후보 영역"이며 분류의 인과적 근거가 아닙니다([§5](#5-병변-evidence--xai) shortcut audit 참고).

> 이 README는 요약입니다. **단일 진실 소스(SSOT)는 [`docs/AI_HANDOFF.md`](docs/AI_HANDOFF.md)** 이며, 문서가 코드/활성 설정(`configs/base.yaml`)과 충돌하면 **항상 코드·설정·아티팩트를 신뢰**합니다.

---

## 목차

1. [개요](#1-개요)
2. [활성 배포 모델](#2-활성-배포-모델)
3. [체크포인트 계보](#3-체크포인트-계보)
4. [런타임 흐름](#4-런타임-흐름)
5. [병변 evidence · XAI](#5-병변-evidence--xai)
6. [페이로드 계약 (백엔드 연동)](#6-페이로드-계약-백엔드-연동)
7. [설정 (`configs/base.yaml`)](#7-설정-configsbaseyaml)
8. [실행 방법](#8-실행-방법)
9. [아티팩트 · 실험 레지스트리](#9-아티팩트--실험-레지스트리)
10. [패키지 구조](#10-패키지-구조)
11. [데이터셋 출처](#11-데이터셋-출처)
12. [문서 맵](#12-문서-맵)
13. [스프린트 히스토리 하이라이트](#13-스프린트-히스토리-하이라이트)

---

## 1. 개요

- **태스크**: `binary_dr_screening` (Normal vs. Abnormal 이진 분류).
- **입력**: 단일 안저 이미지(RGB).
- **출력**: `abnormal_probability`(단일 logit, `num_outputs: 1`) + 병변 evidence overlay + compact 외부 지표.
- **핵심 설계 원칙**: 정확성·재현성·설정 일관성을 속도보다 우선. 모든 결정은 파일·테스트·아티팩트·로그로 검증.
- **현재 상태**: Sprint 5까지 종료(2026-06-03). 활성 배포는 `v31_v8b_fusion_quickqual_v2`.

---

## 2. 활성 배포 모델

**`v31_v8b_fusion_quickqual_v2`** (2026-06-03 승격, v31 collinearity refit).
`artifacts/checkpoints/best.pt` 하나에 분류기 + 병변 segmenter + 수치 meta-classifier가 묶인 composite 체크포인트입니다.

| 구성 요소 | 내용 | 소스 run |
|-----------|------|----------|
| **Base classifier** | EfficientNet-B5, true no-attention(SE/ECA 제거) + block4 gated pooling, Dice+BCE aux seg loss | `v31_no_se_gated_quickqual_v1` |
| **Lesion segmenter** | ResNet50 + U-Net, 4채널 병변 마스크(MA/HE/EX/SE) | `seg_evidence_v8b_quickqual_v1` |
| **Meta-classifier** | StandardScaler + LogisticRegression, feature **88개** = `v31_logit` 1개 + v8b 병변 feature | `v31_v8b_late_fusion_quickqual_v1` 계열 |

**Fusion 동작** (`drscreen/models/fusion.py`): `V31V8bFusion.predict_fusion_score()` 가 분류기로 v31 logit을, segmenter로 병변 맵을 계산하고, 병변 맵을 스칼라 feature로 환원한 뒤 저장된 numpy scaler/LogReg meta-classifier를 적용해 최종 `abnormal_probability` 를 만듭니다.

> `v2`는 `v1`의 redundant `v31_probability`(= `sigmoid(v31_logit)`, near-collinear) 컬럼을 제거한 refit입니다. 표준화 |coef|의 49.8%가 이 2개 feature에 분할돼 계수가 불안정했고, `v31_logit` 단일 표현으로 줄여(89→88 feature) 안정화 + AUROC 소폭 개선했습니다.

### 외부 평가 (DDR external_test, 20% calibration / 80% holdout)

| 지표 | 값 |
|------|-----|
| AUROC | **0.9360** |
| Threshold (배포) | **0.08563** (calibration split에서 직전 active 민감도 0.8234에 맞춰 선택, holdout leakage 방지) |
| Sensitivity | 0.8316 |
| Specificity | 0.9070 |
| Accuracy | 0.8693 |
| F1 | 0.8641 |

- Split 규모: calibration 2,504장 / holdout 10,018장.
- 지표 출처: `artifacts/evaluations/external_test_v31_v8b_fusion_quickqual_v2_best_metrics.json`
  (원본 `artifacts/runs/99_misc/v31_v8b_late_fusion_quickqual_v1_v31rep/evaluations/...`, key `classification_domains:late_fusion:v31_logit`).
- v1 대비 약하게 지배: AUROC +0.0019, Sens +0.008, Spec −0.0016.

### 롤백

| 단계 | 버전 | 백업 체크포인트 |
|------|------|------------------|
| 즉시 | `v31_v8b_fusion_quickqual_v1` (two-feature, AUROC 0.9341, thr 0.06) | `best_pre_collinearity_refit_20260603.pt.bak` |
| 차선 | `v31_v8b_fusion_features_hflip_v2` (circular, AUROC 0.9431, thr 0.3931) | `best_pre_quickqual_v1_20260529.pt.bak` |
| 심층 | `v31_v8b_fusion_v2` (circular, thr 0.38) | `best_pre_features_hflip_v2_20260527.pt.bak` |

롤백 절차: `.bak` 를 `artifacts/checkpoints/best.pt` 로 복사하고 `configs/base.yaml` 의 `project.version`·`infer.threshold`(circular 계열은 `preprocess_mode`/`tta_mode`/`manifest_path` 도)를 해당 버전 기준으로 되돌립니다.

---

## 3. 체크포인트 계보

`best.pt` 는 단일 학습의 산물이 아니라 긴 도메인 일반화·병변 근거 연구의 결과입니다. 주요 마일스톤(외부 테스트셋은 Sprint 3부터 Messidor 1,200장 → DDR 12,522장으로 교체):

```text
EfficientNet-B5 (ImageNet)
  → SSL SimCLR (APTOS+IDRiD+Messidor 5,378장 비레이블)
    → v4.1 supervised fine-tune            [val AUROC 0.9975]
      → v6_alpha_only (focal α=0.75)        [Messidor AUROC 0.8697]
        → v7_messidor_train (Messidor 편입) [DDR AUROC 0.8725]
          → v17_focal_g2 (FDA + focal γ=2.0)[DDR AUROC 0.8911]  ← Sprint 3 배포 best
            → v7_512_messidor_train (512px)  [DDR AUROC 0.9046]
              → v24_multitask (aux seg loss)
              → v28_no_attention → v30_gated_pooling [DDR AUROC 0.9137]
                → v31_no_se_gated (true no-attention + Dice+BCE) [DDR AUROC 0.9160]  ← fusion base classifier
                  + seg_evidence_v8b (MAPLES/TJDR/DDR_SEG 병변 마스크, MAPLES ROI 보정)
                    → v31_v8b_fusion_v2 (score-level fusion)        [DDR holdout 0.9403]
                    → v31_v8b_fusion_features_hflip_v2             [DDR holdout 0.9431]
                    → v31_v8b_fusion_quickqual_v1 (train-serve 일관성) [DDR holdout 0.9341]
                    → v31_v8b_fusion_quickqual_v2 (collinearity refit) [DDR holdout 0.9360]  ← ACTIVE
```

- v31은 단독 배포가 아니라 **fusion 내부 base classifier**로만 보존됩니다.
- `quickqual` 라인은 **peak AUROC가 아니라 train-serve 전처리 일관성**(백엔드 QuickQual geometry ↔ AI 학습 manifest)을 우선해 선택된 배포선입니다.
- 전체 계보·실패 분기(grounded classifier DFR/BagNet/CBM, decoder alignment v36~v39 등)는 [`docs/AI_HANDOFF.md` §3](docs/AI_HANDOFF.md) 와 [§9](#9-아티팩트--실험-레지스트리) 참고.

---

## 4. 런타임 흐름

### 학습 / 평가 경로 (offline-preprocessed)

- 일관성·속도를 위해 **오프라인 전처리된 매니페스트**를 사용합니다.
  - 분류 manifest: `data/processed/manifest_preprocessed_quickqual.csv`
  - fusion/evidence manifest: `data/processed/manifest_with_maples_tjdr_ddrseg_quickqual_preprocessed.csv`
- 전처리는 backend 호환 QuickQual crop/square-pad geometry + Ben Graham 정규화를 오프라인으로 적용한 상태입니다.
- `data.use_preprocessing: false` 는 의도적입니다 — 오프라인 전처리 매니페스트에 `true` 를 주면 전처리가 **이중 적용**됩니다.
- 마스크 정합: pixel-mask provider가 raw 병변 마스크를 manifest 이미지 geometry(crop/pad/resize)에 맞춥니다. 마스크에는 Ben Graham photometric 정규화를 적용하지 않습니다.

### 추론 경로 (`drscreen/infer/service.py` · `InferenceSession`)

1. 입력 이미지 RGB 변환.
2. **선택적 라이브 전처리** — 활성 배포는 `infer.use_preprocessing: true` 이지만 `infer.preprocess_mode: none`. 즉 **백엔드 QuickQual 태스크가 crop/square-pad 를 이미 적용**했다고 가정합니다.
3. Eval transform — Resize(512) → CenterCrop(512) → ToTensor → Normalize(ImageNet stats).
4. Forward — `V31V8bFusion` 이 v31 logit + v8b 병변 feature 계산 → meta-classifier → `abnormal_probability`.
5. Evidence / XAI — v8b 병변 segmentation evidence(`evidence_type: lesion_segmentation`). 실패 코드: CAM `XAI_001`, 병변 evidence `XAI_002`, grounded-classifier `XAI_003`.
6. Payload 조립 — 백엔드용 구조화 JSON.

### 전처리 geometry · footgun

- **백엔드 계약** (`backend/models/quickqual_wrapper.py:preprocess_fundus_image`): RGB-mean>15 bbox crop + square pad + 1024 resize. **Ben Graham은 백엔드가 적용하지 않음** → AI가 추론 시 1회 적용.
- **Double-BG footgun**: 이미 전처리된(`processed*`) 이미지에 serve 전처리를 다시 태우면 Ben Graham이 재적용됩니다. `FundusPreprocess.apply_ben_graham` 플래그 + `is_preprocessed_image_path` 헬퍼가 경고하도록 가드되어 있습니다.
- **Raw-input footgun**: 단독 CLI(`drscreen.cli.infer`)로 **원본 이미지**를 추론하면 `preprocess_mode: none` 이라 crop이 생략됩니다. raw 입력은 `preprocess_mode: quickqual` 설정 config를 사용하세요.
- AI 학습/평가/전처리는 품질 필터를 돌리지 않습니다. QuickQual은 **백엔드의 별도 태스크**이고, AI payload의 quality 필드는 호환용 placeholder(`None`)입니다.

---

## 5. 병변 evidence · XAI

- **활성 evidence**: v8b lesion segmentation (`seg_evidence_v8b_quickqual_v1`). 4채널 병변(MA/HE/EX/SE) 맵을 evidence overlay로 사용.
  - threshold-0.5 집계: MAPLES mDice 0.2492 / union IoU 0.1939, TJDR mDice 0.3493 / union IoU 0.3054, DDR_SEG mDice 0.3536 / union IoU 0.2619.
- **Shortcut audit (D5/D7)**: v31 base에서 block4 도메인 분리도(macro AUROC 0.9681)와 counterfactual style swap이 shortcut 의존을 시사했습니다. 따라서 **제품 문구는 "abnormal로 분류 + 별도로 탐지된 병변 후보 영역"** 으로 표현하고 "이 병변들 때문에 분류"라는 인과적 표현은 피합니다.
  - 단, 2026-06-03 meta-level counterfactual probe에서 **활성 quickqual base/meta는 이미 강하게 lesion-grounded**(matched_nonlesion/lesion: base 0.041, meta 0.046, shortcut_signal false)임이 확인됐습니다. 자주 인용된 D7 1.48x는 circular `v31_no_se_gated` proxy 수치이며 활성 모델에는 해당하지 않습니다.
- **IDRiD overlay caveat**: IDRiD seg-test mDice는 QuickQual 라인 전반에서 붕괴(0.10~0.22)했고 patient-level contamination 의심도 있어 **evidence 품질 판단 지표로 쓰지 않습니다**. MAPLES/TJDR/DDR_SEG로 판단하세요.
- **Phase-0 XAI gate**: CAM 연구 경로의 aspirational 기준(2σ=0.1089)은 전 모델 미달입니다. 활성 evidence는 CAM이 아니라 segmentation metric(`xai_seg_mdice`, `xai_seg_union_iou`, `xai_auc_iou`)으로 별도 평가합니다.

---

## 6. 페이로드 계약 (백엔드 연동)

`InferenceSession` 이 `prediction.payload` 로 백엔드에 전달하는 주요 필드:

| 필드 | 의미 |
|------|------|
| `predicted_index` / `predicted_label` | `0`(Normal) / `1`(Abnormal) 와 라벨 문자열 |
| `abnormal_probability` | 0.0 ~ 1.0 |
| `decision_threshold` | 최종 판정 threshold |
| `evidence_type` | `"lesion_segmentation"` / `"cam_research"` / `"grounded_classifier"`(연구용) |
| `lesion_summary` | 병변별 면적비·presence 요약(병변 evidence 활성 시), 없으면 `null` |
| `evidence_warning` | 예: `"LESION_EVIDENCE_LOW_CONFIDENCE"`, 없으면 `null` |
| `xai_error_code` / `xai_no_region` | `"XAI_001/002/003"` 또는 `null` / evidence 영역 없으면 `true` |
| `should_block` | AI측 하드 차단 플래그. 현재 모델은 `false` 유지 |
| `eval_metrics` | `external_test_<version>_best_metrics.json` 의 compact 외부 지표 |
| `quality*` | 백엔드 QuickQual 호환 placeholder (AI는 `null`) |

아티팩트 경로 필드: `checkpoint_path`, `prediction_path`, `heatmap_path`, `lesion_map_path`.
계약 검증은 `tests/regression/test_payload_contract.py`, `test_fusion_contract.py` 와 [`docs/AI_HANDOFF.md` §5](docs/AI_HANDOFF.md).

---

## 7. 설정 (`configs/base.yaml`)

```yaml
project.version:         v31_v8b_fusion_quickqual_v2
model.architecture:      v31_v8b_fusion
model.use_attention:     false          # attention_mode: none (SE/ECA/Spatial 제거)
model.use_aux_seg:       true           # aux_seg_block: 4
model.use_gated_pooling: true           # block4 lesion gate x classifier pooling
model.segmenter_encoder: resnet50       # segmenter_out_channels: 4
data.image_size:         512            # resize_size: 512
data.manifest_path:      data/processed/manifest_preprocessed_quickqual.csv
data.use_preprocessing:  false          # 오프라인 전처리 매니페스트 → 의도적 false
infer.checkpoint_path:   artifacts/checkpoints/best.pt   # 고정 alias
infer.use_preprocessing: true
infer.preprocess_mode:   none           # 백엔드가 QuickQual crop 적용 가정
infer.threshold:         0.08563088401268978
infer.use_meta_classifier: true
infer.evidence_type:     lesion_segmentation
infer.lesion_threshold:  0.5
```

- **Threshold 정책**: 추론은 선택된 run 아티팩트의 `optimal_threshold`(`external_test_<version>_best_metrics.json`)를 우선 사용하고, `infer.threshold` 는 배포 fallback입니다(활성 아티팩트의 DDR optimal threshold와 정렬).

---

## 8. 실행 방법

> **런타임 정책**: **학습만 Python 3.14** 를 사용합니다(학습 CLI가 다른 버전을 거부). 배포/추론 런타임(백엔드·Docker)은 변경하지 않습니다. 자동화에서 맨몸 `python -m drscreen.cli.train ...` 은 PATH가 다른 인터프리터로 잡힐 수 있어 피합니다.

```powershell
# 분류기 학습 (래퍼가 Python 3.14 강제)
.\train.ps1 -Config configs/v9_fda.yaml
py -3.14 -m drscreen.cli.train --config configs/v9_fda.yaml

# 병변 segmenter 학습 (mask-valid 행만, val mDice로 best 선택)
py -3.14 -m drscreen.cli.train_seg --config configs/seg_evidence_v8b_quickqual_v1.yaml
```

```bash
# 평가 (DDR external test)
python -m drscreen.cli.evaluate --config configs/base.yaml --split external_test

# 추론 — 백엔드 전처리(QuickQual crop) 가정 (preprocess_mode: none)
python -m drscreen.cli.infer --config configs/base.yaml --image path/to/cropped.png

# 회귀 테스트 (payload·fusion·preprocess geometry 계약)
py -3.14 -m pytest tests/regression -q
```

- 분류기 선택 지표: 기본 `val_auroc`(`min_checkpoint_sensitivity` 0.80 충족 epoch 중 최고 validation AUROC). 도메인 일반화 run은 `external_calibration_auroc` 선택 사용 가능(global-best 승격 생략).

---

## 9. 아티팩트 · 실험 레지스트리

### 저장 레이아웃

```text
artifacts/
├── checkpoints/best.pt              # 활성 배포 alias (고정 경로)
├── checkpoints/*.pt.bak             # 롤백 백업
├── runs/<primary_group>/<run_id>/
│   ├── checkpoints/                 # best.pt, last.pt, training_summary.json
│   ├── evaluations/                 # 분류 metric + XAI JSON
│   └── logs/
└── evaluations/                     # external_test_<version>_best_metrics.json (runtime alias)
```

- **고정 alias 정책**: `infer.checkpoint_path` 와 `train.global_best_checkpoint_path` 는 모두 `artifacts/checkpoints/best.pt`. 새 버전 배포 시 **경로를 바꾸지 말고** 해당 버전 체크포인트를 이 파일에 복사/승격하고, `base.yaml` 의 `project.version`·model flags·`infer.threshold` 를 함께 맞춥니다. 승격은 수동입니다.

### 실험 그룹 ([`docs/EXPERIMENT_REGISTRY.md`](docs/EXPERIMENT_REGISTRY.md))

| Group | 주제 |
|-------|------|
| `00_baselines_and_early` | 초기 baseline / supervised |
| `01_ssl_lineage` | SSL 계보, SSL 오염 검증, focal 변형 |
| `02_domain_generalization` | Messidor 편입, FDA, SWAD, IBN, CORAL |
| `03_resolution_layercam` | 512px 학습, Layer-CAM |
| `04_lesion_supervision` | aux 병변 마스크 감독 |
| `05_xai_attention_ablation` | attention ablation, block sweep |
| `06_xai_classifier_routing` | lesion gate routing |
| `07_lesion_evidence` | v31, per-lesion routing(v32~v35), shortcut audit |
| `08_xai_decoder_alignment` | U-Net decoder, CAM alignment(v36~v39) |
| `09_evidence_segmentation` | classifier-독립 병변 segmentation(seg_evidence_*) |
| `10_grounded_classifier` | shortcut-free 진단(DFR/BagNet/CBM), v31+v8b fusion |
| `99_misc` | 재현/스테이징/legacy. QuickQual 활성 소스 일부가 여기 위치 |

> 활성 배포는 모든 소스(`base.yaml`·`docs/AI_HANDOFF.md`·`docs/EXPERIMENT_REGISTRY.md`)에서 `quickqual_v2`(thr 0.0856)로 일치합니다.

---

## 10. 패키지 구조

```text
ai/
├── drscreen/                  # 메인 패키지
│   ├── cli/                   # train, train_seg, evaluate, infer, pipeline, build_fusion_checkpoint, diagnose_*
│   ├── data/                  # datasets, transforms, manifest_builder, mask_providers, anatomy
│   ├── models/                # build, fusion(V31V8bFusion), seg_evidence, aux_seg, concept_bottleneck, sparse_bagnet
│   ├── infer/                 # service(InferenceSession), pipeline, payload, late_fusion_features
│   ├── train/                 # runner, seg_runner, engine, loss, metrics, checkpointing, evaluate
│   ├── xai/                   # gradcam, faithfulness, perturbation, iou, seg_metrics, evaluation
│   ├── utils/                 # checkpoint, logging, seed
│   └── settings.py            # 설정 병합 / primary-group 경로 resolve
├── configs/                   # 실험·배포 YAML (base.yaml = 활성 배포)
├── artifacts/                 # 체크포인트·평가·진단 (§9)
├── docs/                      # AI_HANDOFF(SSOT), DEVLOG, EXPERIMENT_REGISTRY, AI_SECTOR_MAP, SPRINT*
├── tests/regression/          # payload·fusion·preprocess geometry/guard 계약 테스트
├── archive/retfound/          # 폐기된 RETFound 실험 (참고용)
├── train.ps1                  # 학습 래퍼 (Python 3.14 강제)
└── AGENTS.md                  # 작업 전 읽을 가이드
```

상위 단계의 코드/아티팩트 섹터 분해는 [`docs/AI_SECTOR_MAP.md`](docs/AI_SECTOR_MAP.md) 참고(Sector 0 활성 런타임 ~ Sector 6 legacy/staging).

---

## 11. 데이터셋 출처

학습·평가에 사용된 공개 데이터셋의 저작권·credit. 데이터 원본은 저장소에 포함하지 않으며, 다운로드·재사용은 각 원 배포처의 라이선스를 따릅니다.
활성 배포의 병변 evidence feature는 `seg_evidence_v8b_quickqual_v1` 계열 segmenter에서 생성되며, 병변 feature 학습 mask-valid 행 수는 **DDR_SEG 532 / TJDR 448 / MAPLES 122 / IDRiD 54** 입니다.

| 데이터셋 | 이 프로젝트에서의 사용 | 출처 / 인용 | 라이선스 |
|---|---|---|---|
| APTOS 2019 | DR 분류 학습 보조, SSL 사전학습 | [Kaggle — APTOS 2019](https://www.kaggle.com/c/aptos2019-blindness-detection) (Aravind Eye Hospital) | Kaggle Competition Terms |
| IDRiD | DR 분류, XAI 평가, v8b 병변 feature(MA/HE/EX/SE) | [IDRiD Grand Challenge](https://idrid.grand-challenge.org/Data/) / [IEEE DataPort](https://ieee-dataport.org/open-access/indian-diabetic-retinopathy-image-dataset-idrid) | CC BY 4.0 |
| Messidor | DR 분류 학습, SSL, MAPLES 원본 fundus | [ADCIS — Messidor](https://www.adcis.net/en/third-party/messidor/) | 비상업적 연구 목적 |
| MAPLES-DR | v8b 병변 feature 학습·평가(MA/HE/EX/CWS) | [LIV4D/MAPLES-DR](https://github.com/LIV4D/MAPLES-DR), [Scientific Data 2024](https://www.nature.com/articles/s41597-024-03739-6) | label/code: CC0-1.0 · 논문: CC BY-NC-ND 4.0 · 원본 fundus는 Messidor 조건 |
| TJDR | v8b 병변 feature 학습(MA/HE/EX/SE pixel mask) | [TJDR page](https://www.juheapi.com/datasets/tjdr), [arXiv:2312.15389](https://arxiv.org/abs/2312.15389) | 공개 연구용 · 명시 라이선스 미확인, 원 배포처 조건 및 논문 인용 필요 |
| DDR / DDR_SEG | DR 분류 외부 테스트(12,522장) 및 v8b 병변 segmentation subset | [GitHub — nkicsl/DDR-dataset](https://github.com/nkicsl/DDR-dataset) | 배포 repo: MIT · DDR 논문 인용 및 원 배포처 조건 준수 |

TJDR 인용:

```bibtex
@article{mao2023tjdr,
  title={TJDR: A High-Quality Diabetic Retinopathy Pixel-Level Annotation Dataset},
  author={Mao, Jingxin and Ma, Xiaoyu and Bi, Yanlong and Zhang, Rongqing},
  journal={arXiv preprint arXiv:2312.15389},
  year={2023}
}
```

> FGADR는 접근 절차가 무거워 기본 경로에서 제외하고, DDR lesion segmentation subset을 대체 데이터로 사용했습니다.

---

## 12. 문서 맵

| 문서 | 용도 |
|------|------|
| [`docs/AI_HANDOFF.md`](docs/AI_HANDOFF.md) | **단일 진실 소스**. 현재 아키텍처·설정·런타임·체크포인트 계보·payload 계약. 먼저 읽으세요. |
| [`docs/EXPERIMENT_REGISTRY.md`](docs/EXPERIMENT_REGISTRY.md) | 체크포인트·평가·XAI 결과의 canonical 분류 인덱스(run별 group·config·아티팩트·승격 여부) |
| [`docs/AI_SECTOR_MAP.md`](docs/AI_SECTOR_MAP.md) | 코드·아티팩트를 runtime/data/training/XAI/evidence/fusion 섹터로 분해한 리뷰 체크리스트 |
| [`docs/DEVLOG.md`](docs/DEVLOG.md) | 역사적 변경 로그 (*왜* 그렇게 됐는지 참고용, 현재 진실 아님) |
| [`docs/SPRINT*_Devlog.md`](docs/) | 스프린트별 회고 (Sprint 5 = CLOSED 2026-06-03). 예: [`docs/SPRINT5_Devlog.md`](docs/SPRINT5_Devlog.md) |
| [`docs/INDEX.md`](docs/INDEX.md) | docs 디렉토리 안내 |
| [`AGENTS.md`](AGENTS.md) | 이 모듈에서 작업하기 전 읽을 가이드 |
| 저장소 루트 `README.md` | 프로젝트 통합 개요, Docker Compose, 브랜치 맵, API 목록 |

> 문서가 코드/활성 설정과 충돌하면 **항상 코드·설정을 신뢰**합니다. 일부 historical 문서에는 원래 `fundus_dr_ai` 프로젝트의 경로·이름이 남아 있을 수 있습니다.

---

## 13. 스프린트 히스토리 하이라이트

`docs/SPRINT*_Devlog.md` 의 핵심만 요약합니다.

- **Sprint 2** — RETFound backbone 교체 실험 폐기(Messidor AUROC 0.66, EfficientNet-B5 0.87 미달). v4.1 SSL 계보 정정, SSL 오염 효과 실재 확인(Messidor 제외 시 0.8697→0.7262).
- **Sprint 3** — 외부 테스트셋 Messidor→DDR 교체. 도메인 일반화 시리즈(v7~v20). **focal γ=2.0 + FDA 의 `v17_focal_g2`** 가 DDR threshold bias를 0.06→0.42로 개선하며 AUROC 0.8911로 배포 best.
- **Sprint 4** — 512px backbone, aux 병변 감독, attention ablation, gated pooling을 거쳐 `v31_no_se_gated`(DDR 0.9160). XAI shortcut audit(D5/D7)로 CAM 인과 근거 한계 확인 → 독립 병변 segmentation evidence 경로로 전환.
- **Sprint 5 (CLOSED 2026-06-03)** — TJDR/DDR_SEG 통합 + MAPLES ROI 좌표 보정으로 `seg_evidence_v8b` 확보. v31+v8b late fusion 배포(fusion_v2 → features_hflip_v2 → quickqual_v1 → **quickqual_v2**). grounded classifier 재진입(DFR/BagNet/CBM)·safezoom/contentcrop 전처리·domain-overfit mitigation은 모두 진단만 하고 미승격. anatomy-aware evidence는 활성 모델이 이미 lesion-grounded임이 확인돼 evidence-based로 종결.

> 미승격으로 끝난 실험이 많습니다. 이는 정확성·재현성을 속도보다 우선하고, **DDR 분류 게이트 + 병변 localization 게이트 + sensitivity guard** 를 모두 통과한 후보만 배포로 승격했기 때문입니다.
