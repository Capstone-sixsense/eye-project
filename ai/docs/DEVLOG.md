# DEVLOG — fundus_dr_ai

> **IMPORTANT**: This document is for **historical reference only**. 
> For the current project state, architecture, and configuration, please refer to [AI_HANDOFF.md](./AI_HANDOFF.md).
> If this history conflicts with `AI_HANDOFF.md`, trust `AI_HANDOFF.md`.

프로그램 설계, 변경 이력, 수정 대기 항목, 개선 계획을 단일 문서로 관리한다.

---

## 2026-05-14 MAPLES-DR XAI eval (v31/v35) — clean-cohort 수치 확인

### 목적

IDRiD XAI 수치에 학습 도메인 편향이 포함됐을 가능성을 검증하기 위해,
완전히 별개 코호트(MESSIDOR → MAPLES-DR 어노테이션)에서 v31/v35 XAI를 측정했다.

### 구현 내용

- `drscreen/xai/iou.py`: `load_maples_masks()` 추가 (MAPLES-DR 어노테이션 로더)
- `drscreen/xai/evaluation.py`: `evaluate_maples()` 추가 + `_process_image`에 `mask_loader` 파라미터 추가
- `eval_xai_maples.py`: MAPLES-DR XAI eval CLI 진입점 신규 생성
- MESSIDOR 이미지 경로: `data/raw/messidor/images/{stem}.tif`
- MAPLES-DR split: test 60장 / train 138장 (`dataset_record.yaml` 기준)

### 결과 (test split, 60장, block4 Layer-CAM)

| Model | PG | AUPRC | AUC-IoU | IoU top20% |
|-------|:---:|:---:|:---:|:---:|
| v31 (IDRiD 참고) | 0.3704 | 0.1409 | 0.0496 | 0.0785 |
| v35 (IDRiD 참고) | 0.4074 | 0.1537 | 0.0525 | 0.0796 |
| **v31 (MAPLES-DR)** | **0.0500** | **0.0172** | **0.0051** | **0.0113** |
| **v35 (MAPLES-DR)** | **0.0500** | **0.0166** | **0.0053** | **0.0098** |

### 해석

**IDRiD XAI 수치는 과대평가였다.** MAPLES-DR clean-cohort 기준:
1. 전 지표 약 10× 하락 — 진짜 로컬라이제이션 일반화 능력은 IDRiD 수치가 시사한 것과 전혀 다름
2. v31 vs v35 차이 소멸 — v33~v35 XAI 개선(AUPRC +0.013)은 IDRiD-특화 패턴 강화였으며 OOD에서 재현 불가
3. PG 0.0500 (3/60) = 사실상 무작위 수준
4. IDRiD 수치를 XAI 개선 지표로 사용하는 실험 방향 재검토 필요

**결론**: 현 아키텍처(v31~v35)는 외부 코호트에서 의미있는 병변 로컬라이제이션을 보이지 않는다.
다음 단계: anatomy-guided CAM masking — optic disc 영역 CAM 제외로 false positive 억제 후 재평가.

---

## 2026-05-14 실험 계획 재수립 / MAPLES-DR 확보 / MAPLESMaskProvider 구현

### 실험 계획 재분류

이전 세션(v33 완료, v31 배포 승격)에서 이어, 전체 실험을 현재 코드·docs 기준으로 재분류했다.

| 분류 | 실험 |
|------|------|
| **COMPLETED** | v31 test XAI, v31 배포 승격, Phase-0 gate 파라미터화, v33 per-lesion routing, IDRiD contamination 문서화 |
| **DROP** | v27 MIL attention (random baseline 이하), v29 with_attention XAI (분류 열위, 방향 폐기), SparseBagNet, Phase 2A anatomy confounder |
| **PROCEED** | v34 (lambda=0.3 per-lesion routing), MAPLES-DR 기반 XAI eval/anatomy audit 평가 CLI·지표 wiring |

### MAPLES-DR 데이터 확보 확인

`data/raw/MAPLES-DR/` 로컬 데이터 구조 확인 완료:
- `AdditionalData/`: `dataset_record.yaml`, `MESSIDOR-ROIs.csv`, `diagnosis_infos.xls`, `biomarkers_annotation_infos.xls`, `annotations/{12종 biomarker}/`
- 마스크: train 138장 / test 60장, 1500×1500 PIL mode=1 binary
- `maples_dr.configure(maples_dr_path='data/raw/MAPLES-DR/AdditionalData', messidor_path='data/raw/messidor')` 정상 동작 확인

자동 다운로드(Figshare)는 0 bytes 실패 2회. 이미 로컬에 완비된 데이터를 직접 경로 지정으로 사용.

### v35 학습 및 평가 결과 / 4ch per-lesion routing 방향 종결

**학습**: val AUROC 0.9992 (epoch 9 = head3 + finetune6, early stop). v31 checkpoint warmstart 정상 작동 (missing=5 unexpected=0 — seg_head 4ch 파라미터만 fresh init).

**DDR external test**: AUROC **0.9081** (optimal thr=0.18, Sens=0.793, Spec=0.874, 12,522장).
v33(0.9131), v34(0.9129)보다 더 나빠짐. Warmstart 역효과 확인.

**XAI eval (test split, 27장)**:

| 지표 | v31 | v33 | v34 | v35 |
|------|-----|-----|-----|-----|
| PG | 0.3704 | 0.4074 | **0.5185** | 0.4074 |
| AUPRC | 0.1409 | 0.1478 | 0.1492 | **0.1537** ← AUPRC 기준 최고 |
| AUC-IoU | 0.0496 | **0.0557** | 0.0543 | 0.0525 |
| IoU top-20% | 0.0785 | **0.0799** | 0.0769 | 0.0796 |

**최종 결론: 4ch per-lesion routing의 DDR 회귀는 하이퍼파라미터로 해소 불가.**
- v33 (lambda=0.5, v7_512 warmstart): DDR 0.9131, AUPRC 0.1478
- v34 (lambda=0.3, v7_512 warmstart): DDR 0.9129, AUPRC 0.1492
- v35 (lambda=0.3, v31 warmstart, head_epochs=3, lr/2): DDR 0.9081, AUPRC 0.1537

XAI AUPRC는 v35에서 최고치(0.1537)이나, DDR 일반화는 시도할수록 오히려 악화. 4ch gating이 IDRiD/APTOS/Messidor 내부 분포에는 강하지만 DDR OOD에 취약한 feature를 학습하게 만드는 구조적 문제로 판단.

**방향 전환 결정**: 4ch per-lesion routing 실험 종결. v31 배포 유지. XAI 개선은 분류에 영향 없는 post-hoc 방법 탐색으로 전환한다. anatomy-guided CAM masking과 MAPLES-DR clean-cohort eval은 먼저 평가 CLI/metric wiring이 필요하다.

### v35 warmstart routing 실험 결정

v33/v34 모두 DDR AUROC ~0.9129-0.9131로 v31(0.9160) 대비 -0.003 고정 회귀. lambda 조정으로 해소 불가.

**가설 재수립**: 회귀 원인은 lambda가 아니라 **backbone 초기화 기준**. v33/v34는 v7_512(DDR AUROC 0.9046) checkpoint를 기반으로 4ch seg head + gated routing을 처음부터 학습. v31은 같은 기반이지만 1ch seg head → DDR 0.9160 달성. 4ch per-lesion 구조는 v7_512 backbone에서 시작할 때 분류 quality 상한이 낮을 수 있음.

**v35 전략**: v31 checkpoint(DDR 0.9160 달성 backbone)에서 warmstart → 4ch seg head + per-lesion routing 추가.
- backbone+classifier는 v31의 학습된 가중치 그대로 로드 (분류 quality 보존)
- 4ch seg_head + lesion_weights는 fresh init (아키텍처 불일치로 자동 제외)
- `head_epochs: 3`으로 seg head 선행 warmup
- `backbone_learning_rate: 0.00002` (v31 기반이므로 LR 절반으로 감소)
- `lambda_aux_seg: 0.3` 유지

성공 기준: DDR AUROC ≥ 0.9160 AND XAI AUPRC ≥ 0.1492.

### v34 학습 및 평가 결과

**config**: v33과 동일, `lambda_aux_seg: 0.5 → 0.3` 변경.

**학습**: val AUROC 0.9989 (epoch 3, early stop 7). v33(0.9980)보다 소폭 개선.

**DDR external test**: AUROC **0.9129** (optimal thr=0.51, Sens=0.772, Spec=0.908).
v33(0.9131)과 동등, v31(0.9160) 대비 -0.003 회귀 지속. lambda 감소로 분류 개선 불가 — 가설 기각.

**XAI eval (test split, 27장)**:

| 지표 | v31 | v33 | **v34** |
|------|-----|-----|---------|
| PG | 0.3704 | 0.4074 | **0.5185** ← PG 기준 최고 |
| AUPRC | 0.1409 | 0.1478 | **0.1492** |
| AUC-IoU | 0.0496 | **0.0557** | 0.0543 |
| IoU top-20% | 0.0785 | **0.0799** | 0.0769 |

PG 기준 최고. AUPRC는 당시 최고였으나 이후 v35에서 갱신됨. AUC-IoU·IoU top-20%는 v33 소폭 미달. 배포 기준(DDR ≥ v31 AND XAI ≥ v31) 중 분류 미달로 v31 유지.

**결론**: 4ch per-lesion routing의 DDR AUROC -0.003 회귀는 lambda 조정으로 해소 불가. 구조적 trade-off (4ch gating이 분류 feature를 희생). v35 방향: v31 checkpoint warmstart 또는 backbone_lr 추가 감소.

### MAPLESMaskProvider 구현

`drscreen/data/mask_providers.py`에 `MAPLESMaskProvider` 추가:
- Channel 순서: MA / HE / EX / SE(CWS) — `IDRiDPerLesionMaskProvider`와 동일
- MAPLES-DR 매핑: `Microaneurysms / Hemorrhages / Exudates / CottonWoolSpots`
- 입력: `domain="messidor"` 이미지만 처리, 그 외 `is_valid=False`
- resize: `cv2.INTER_NEAREST` (IDRiD loader와 동일)
- `LesionMaskProvider` Protocol 충족 확인

검증 결과: shape `[4, 512, 512] float32`, MA/HE/EX 채널 픽셀 정상, IDRiD→False, 미포함 MESSIDOR→False.

---

## 2026-05-14 Phase-0 XAI Gate 평가 / v31·v32 학습 및 XAI 평가 / 실험 계획 재수립

### 배경

이전 세션(v30 완료)에서 이어, Phase-0 XAI gate를 기준으로 현재 모델들의 병변 localization 능력을 정량 평가하고, v31·v32 학습 및 XAI eval을 수행했다.

### Phase-0 Gate 평가 결과

현재 문서 기준으로 Phase-0 gate는 IDRiD **test split(27장)** 기준으로 통일한다. center_gaussian baseline 대비 2σ gate는 AUC-IoU > 0.0366 + 2×0.0362 = **0.1089**다.

| 모델 | XAI 방법 | split | AUC-IoU | AUPRC | PG | Gate |
|------|----------|-------|---------|-------|-----|------|
| v28_no_attention | LayerCAM block4 | test | 0.0374 | 0.1253 | 0.4444 | **FAIL** |
| v30_gated_pooling | LayerCAM block4 | test | 0.0443 | 0.1311 | 0.3704 | **FAIL** |
| v31_no_se_gated | LayerCAM block4 | test | 0.0496 | 0.1409 | 0.3704 | **FAIL** |
| v33_per_lesion_routing | LayerCAM block4 | test | **0.0557** | 0.1478 | 0.4074 | **FAIL** |
| v34_calibrated_routing | LayerCAM block4 | test | 0.0543 | 0.1492 | **0.5185** | **FAIL** |
| v35_warmstart_routing | LayerCAM block4 | test | 0.0525 | **0.1537** | 0.4074 | **FAIL** |
| center_gaussian (baseline) | — | test | 0.0366±0.0362 | 0.0526 | — | thresh=0.1089 |

전 모델 Phase-0 gate FAIL. 2σ 기준이 현실적으로 달성 불가 수준임을 확인. gate 기준 재조정 필요.

### v27 MIL Attention 평가 (Phase 3)

v27 config에 `use_aux_seg: false`, `use_attention: true`, `attention_mode: eca_spatial` 추가 후 재평가.
AUC-IoU 0.0119 — random baseline(0.0260) 이하. MIL attention은 spatial localization에 구조적으로 부적합. **방향 폐기**.

### v31 학습 결과

- config: `attention_mode: none` (true no-attention), `use_gated_pooling: true`, `aux_seg_block: 4`, `seg_loss_type: dice_bce`, `lambda_aux_seg: 0.5`
- val AUROC: **0.9993** (epoch 5), early stop at epoch 8
- global best(0.9995) 미달로 best.pt 미승격
- train XAI (block4 LayerCAM): AUPRC 0.1174, AUC-IoU 0.0491, IoU top-20 0.0601, PG 0.3333
- **v28 대비 전 XAI 지표 우세** (PG 3배, AUPRC +45%)

### v32 학습 결과

- config: `attention_mode: none`, `use_gated_pooling: true`, `aux_seg_block: 4`, `aux_seg_channels: 4` (per-lesion MA/HE/EX/SE), `seg_loss_type: dice_bce`, `lambda_aux_seg: 0.5`
- val AUROC: **0.9992** (epoch 3), early stop at epoch 8
- train XAI (seg_head 직접 출력): AUPRC 0.0538, AUC-IoU 0.0208 — v28 대비 열위
- 상태: v32 artifact는 train split seg_head 직접 출력만 평가됐고, 제품 XAI 후보로 쓰기에는 낮음
- 개선 방향: 채널별 독립 gate → learnable weighted sum (v33). 현재 코드의 4채널 gated classifier는 이 경로를 사용하며, 단일 evidence map 생성용 `predict_seg_union()`만 `amax(dim=1)` union을 사용함.

### 실험 분류 재수립

| 분류 | 실험 |
|------|------|
| **DROP** | v27 MIL attention, v29 with_attention, SparseBagNet(Phase 4), Phase 2A anatomy confounder |
| **IMPROVE** | v32 seg_head 한계 → per-lesion weighted routing(v33), Phase-0 gate 기준 test split 통일 |
| **PROCEED** | v31/v32 test-split eval, DDR 분류 평가, v30 deployment 승격 검토, MAPLES-DR 확보 |

### 시각화

IDRiD_25 (v31 최고 IoU 이미지, IoU=0.2547)에 대한 5-panel heatmap vs GT lesion mask 시각화 생성.
`artifacts/heatmaps/heatmap_vs_gt_IDRiD_25_v31_no_se_gated.png`

관찰: LayerCAM이 좌상단 병변 영역을 포착하나 HE(Hemorrhage)/SE(Soft Exudate)는 거의 미포착.
원인: 분류 모델이 MA/EX 위주로 feature를 학습하며, post-hoc CAM은 분류에 기여한 특징만 시각화.

### 추가 구현 사항

- `DiceBCELoss` 구현 (`drscreen/train/loss.py`) 및 engine wiring (`engine.py:196-209`)
- `eval_xai_iou.py` Phase-0 gate 출력 및 σ-aware 집계 (`evaluation.py`)
- `visualize_heatmap_gt.py` 신규 (5-panel heatmap vs GT 시각화)
- `v27_mil_attention.yaml` config 수정 (architecture mismatch 수정)

### v31 DDR External Test 결과 (당일 추가)

v31/v32 중 train-split XAI 우세 모델(v31)에 대해 DDR 외부 데이터셋(12,522장) 분류 평가 실행.

| 모델 | DDR AUROC | Optimal Thr | Sens@Opt | Spec@Opt | Acc@Opt |
|------|-----------|-------------|----------|----------|---------|
| v28_no_attention | 0.8924 | 0.45 | 0.748 | 0.906 | — |
| v30_gated_pooling | 0.9137 | 0.31 | 0.784 | 0.901 | — |
| **v31_no_se_gated** | **0.9160** | **0.35** | **0.798** | **0.868** | **0.833** |

v31이 분류 AUROC 기준 최우선 배포 후보로 확인됨.
승격 결정은 v31 test-split XAI eval → v30 test AUPRC(0.1311)와 공정 비교 후 확정.

### v33 per-lesion routing 학습 및 평가 (당일 추가)

**구조 변경**: `drscreen/models/aux_seg.py`
- `lesion_weights: nn.Parameter(zeros(4))` 추가 — zeros 초기화 → softmax = 균등 [0.25×4]
- `_forward_gated_classifier`: 4ch 경로에서 `amax` 대신 per-channel sigmoid + softmax weighted sum

**학습 결과**:
- val AUROC: 0.9980 (epoch 3, early stop 7)
- DDR external_test AUROC: 0.9131 (v31 0.9160 대비 -0.003)

**XAI eval (test split, 27장)**:

| 지표 | v31 test | v33 test | diff |
|------|----------|----------|------|
| PG | 0.3704 | **0.4074** | +10% |
| AUPRC | 0.1409 | **0.1478** | +5% |
| AUC-IoU | 0.0496 | **0.0557** | +12% |
| IoU top-20% | 0.0785 | **0.0799** | +2% |

XAI 전 지표 개선 확인. per-lesion weighted sum이 단일 gate보다 병변 localization 품질을 향상시킴.
단, DDR AUROC -0.003 회귀로 배포 미승격 (v31 유지). Phase-0 gate FAIL 지속 (0.0557 < 0.1089).

---

## 2026-05-09 XAI Evidence Pivot / Attention Taxonomy

v30_gated_pooling은 DDR AUROC와 block4 CAM 지표를 개선했지만, 임상의에게 제품 근거로 제시할 수준의 병변 설명으로 보지는 않는다. 이후 제품 후보 XAI는 CAM 보정이 아니라 **lesion segmentation evidence**로 전환한다.

코드 기준 attention 분류를 명시했다.
- `attention_mode: eca_spatial`: `_EcaSpatialAttn` = ECA channel + CBAM spatial
- `attention_mode: eca`: 기존 `use_attention=false` 동작. EfficientNet SE 위치에 `EcaModule`은 남아 있음
- `attention_mode: none`: `IdentitySE`로 SE/ECA/Spatial 계열을 모두 제거하는 true no-attention 모드

신규 실험 config:
- `v31_no_se_gated`: `attention_mode: none`, `aux_seg_block: 4`, gated pooling 유지. v30과 gated-pooling 조건을 맞춘 true no-attention 대조군이며, v30 checkpoint continuation은 아니다.
- `v32_lesion_seg_evidence`: IDRiD MA/HE/EX/SE 4채널 mask provider와 lesion evidence payload 사용. MAPLES-DR은 이후 로컬 데이터가 확인되고 `MAPLESMaskProvider`가 추가됐지만, v32 학습 ingestion에는 포함되지 않았고 XAI eval/anatomy audit wiring은 별도 작업으로 남음.

Payload 확장:
- `evidence_type`: `cam_research` 또는 `lesion_segmentation`
- `lesion_map_path`
- `lesion_summary`: 병변별 area ratio / presence score
- `evidence_warning`

---

## 2026-05-06 Artifact Migration Note

기존 checkpoint/evaluation/XAI/log artifact는 연구 질문 기준으로 재분류했고, primary group 기준으로 물리 이동했다. Canonical 분류표는 [EXPERIMENT_REGISTRY.md](./EXPERIMENT_REGISTRY.md)를 따른다.

현재 completed run artifact의 canonical 위치:
- `artifacts/runs/<primary_group>/<run_id>/checkpoints/`
- `artifacts/runs/<primary_group>/<run_id>/evaluations/`
- `artifacts/runs/<primary_group>/<run_id>/logs/`

예외 정리: `artifacts/checkpoints/best.pt`는 active deployment checkpoint alias로 고정한다. 버전별 canonical checkpoint는 `artifacts/runs/<primary_group>/<run_id>/checkpoints/`에 보관하되, 배포 시에는 해당 checkpoint를 고정 alias에 배치한다. `v29_with_attention`은 이후 사용자 요청에 따라 checkpoint까지 `artifacts/runs/05_xai_attention_ablation/v29_with_attention/checkpoints/`로 이동했고, DDR external_test metric도 `artifacts/runs/05_xai_attention_ablation/v29_with_attention/evaluations/`에 있다.

주요 정정:
- `v24_multitask`는 이제 internal `test`와 DDR `external_test` artifact가 모두 존재한다.
- `v24_multitask` DDR external_test: AUROC 0.845189, optimal threshold 0.17.
- `v28_no_attention` DDR external_test: AUROC 0.892425, optimal threshold 0.45.
- v24/v28 matched block sweep 결과는 attention-enabled v24에서 shortcut-driven XAI mislocalization이 발생하고, no-attention v28에서 mid-level lesion attribution이 회복됨을 지지한다.
- `v29_with_attention`은 checkpoint와 DDR external_test metric이 모두 primary group 경로로 정리됐다. XAI artifact는 아직 없다.
- 배포 기준은 각 run artifact의 DDR `external_test` optimal threshold를 사용한다. Active config는 `v28_no_attention`으로 변경됐다.

---

## 2026-05-06 Sprint Numbering Correction

`SPRINT1_Devlog.md`는 Sprint 1을 v0.1.0~v0.7.0으로, `SPRINT2_Devlog.md`는 Sprint 2를 v7~v14로 고정한다. 사용자 제공 `Product Backlog.csv` 기준 현재 진행 중인 제품 스프린트는 Sprint 3이다. 따라서 AI 실험 기록도 v15부터 v29까지를 Sprint 3 하위 실험으로 정규화한다. AI 변경 이력 내부에 남아 있던 "Sprint 4/5/6" 표기는 제품 스프린트 번호가 아니라 내부 실험 파동을 잘못 승격한 표기였으므로 수정했다.

---

## 목차

1. [프로그램 개요](#1-프로그램-개요)
2. [시스템 설계](#2-시스템-설계)
3. [변경 이력](#3-변경-이력)
4. [수정 대기 항목](#4-수정-대기-항목)
5. [개선 계획](#5-개선-계획)

---

## 1. 프로그램 개요

단일 안저(fundus) 이미지를 입력받아 당뇨망막병증(DR) 유무를 이진 분류하는 의료 AI 스크리닝 보조 도구.

| 항목 | 내용 |
|---|---|
| 태스크 | Binary DR screening (normal vs abnormal) |
| 입력 | 단일 안저 이미지 (PNG / JPG / TIFF) |
| 출력 | 이상 확률, 예측 레이블, XAI 히트맵 경로/오버레이 |
| 학습 데이터 | Manifest에는 APTOS 2019 + IDRiD + Messidor 포함. Active base config는 `train_exclude_domains: []`로, 학습 후보에서 Messidor를 제외하지 않음 |
| 외부 테스트 | DDR |
| 핵심 모델 | EfficientNet-B5. `attention_mode`에 따라 `eca_spatial` / `eca` / `none`을 선택하며, v24부터 auxiliary segmentation head 지원 |

---

## 2. 시스템 설계

### 2.1 디렉토리 구조

```
fundus_dr_ai/
├── configs/            YAML 실험 설정 파일
├── data/
│   ├── raw/            원본 데이터셋 (APTOS, IDRiD, Messidor, DDR)
│   └── processed/      manifest CSV 파일들
├── artifacts/
│   ├── checkpoints/    active deployment checkpoint alias
│   ├── runs/           primary group/run_id 기준 checkpoint/evaluation/log
│   ├── heatmaps/       Grad-CAM/Layer-CAM 오버레이 PNG
│   ├── predictions/    추론 결과 JSON
│   └── quality/        과거 품질 보정 리포트/산출물
├── drscreen/
│   ├── cli/            CLI 엔트리포인트
│   ├── data/           데이터셋, 전처리, manifest 빌더
│   ├── models/         모델 팩토리, 프로파일, auxiliary segmentation head
│   ├── train/          학습 엔진, 메트릭, 러너
│   ├── infer/          추론 파이프라인, 서비스
│   ├── ssl/            SimCLR 자기지도 사전학습
│   ├── xai/            Grad-CAM
│   └── utils/          로깅, 시드
└── preprocess_images.py  오프라인 전처리 스크립트
```

### 2.2 추론 파이프라인

```
이미지 입력
  → RGB 변환
  → infer.use_preprocessing=true이면 FundusPreprocess(output_size=512): Circular Crop → Ben Graham → 512×512 리사이즈
  → eval_transform (Resize 512 → CenterCrop 512 → Normalize)
  → EfficientNet-B5 forward (`attention_mode`에 따라 ECA/Spatial/IdentitySE 결정)
  → Sigmoid → abnormal_probability
  → config/checkpoint threshold → 예측 레이블
  → Grad-CAM 또는 Layer-CAM 히트맵 생성 (실패 시 xai_error_code=XAI_001)
  → Prediction payload 반환
```

**품질 검사 경계**: 현재 AI 모듈은 학습·평가·전처리 단계에서 품질 필터를 적용하지 않는다. 추론 시 QuickQual은 backend 단의 별도 task로 분리되었고, `drscreen/infer/service.py`는 quality 관련 payload 필드를 호환 목적의 `None` 값으로만 채운다.

### 2.3 학습 파이프라인

두 단계(phase) 학습:

1. **Head-only phase** (3 epochs): backbone 동결, classifier만 학습. 높은 LR.
2. **Finetune phase** (15 epochs): 전체 모델 학습. backbone에 낮은 LR 적용.

스케줄러: Linear warmup (2 epochs) → Cosine annealing.

### 2.4 모델 프로파일 (EfficientNet-B5 기준)

| 항목 | 값 |
|---|---|
| 백본 | EfficientNet-B5 (timm, `se_layer`는 `attention_mode`로 결정) |
| 어텐션 모드 | `eca_spatial`: `_EcaSpatialAttn`, `eca`: `EcaModule`, `none`: `IdentitySE` |
| 입력 크기 | Active config 기준 512 × 512. `models/profiles.py`의 B5 native profile은 448 기준 참고값 |
| 파라미터 수 | config-dependent. historical ECA+Spatial 기준은 약 23.16M |
| batch_size | 12 |
| head_lr | 2e-4 |
| backbone_lr | 8e-5 |
| weight_decay | 1e-4 |
| gradient_clip_norm | 1.0 |
| optimizer | AdamW |
| scheduler | Cosine w/ warmup (2 epochs) |
| AMP | BF16 자동 감지 (RTX 5070 Ti 이상), FP16 fallback |
| 체크포인트 선택 | val_sensitivity ≥ 0.80 AND val_auroc > best |

### 2.5 데이터 분할 전략

| split | 역할 | 도메인 |
|---|---|---|
| `train` | 학습 후보 | APTOS + IDRiD + Messidor. Active base config는 `train_exclude_domains: []` 적용 |
| `val` | 체크포인트 선택 | APTOS + IDRiD (IDRiD shadow holdout 포함) |
| `test` | 내부 테스트 | APTOS + IDRiD |
| `external_test` | 외부 일반화 검증 | DDR |

IDRiD shadow holdout: val 분할에 IDRiD 샘플을 의도적으로 포함하여 도메인 편향 없이 체크포인트를 선택하는 전략.

### 2.6 품질 검사 정책

현재 `ai/drscreen` 코드에는 품질 검사 패키지가 없다. 학습/평가/오프라인 전처리는 전체 manifest를 대상으로 수행하며 blur/brightness/QuickQual 필터를 적용하지 않는다. 추론 품질 판정은 backend 단의 QuickQual task가 담당한다. AI payload의 `quality_warning`, `quality`, `quality_grade`, `quality_grade_confidence` 필드는 backend 호환을 위해 남아 있으나 현재 AI 모듈 내부에서는 `None`으로 채워진다.

### 2.7 Grad-CAM 구현

- Hook 방식: forward hook으로 activation, backward hook으로 gradient 캡처
- target layer: `infer.gradcam_target_block`로 지정. Active v28 기준 block4, legacy/default last-block은 `blocks.6`
- 후처리: 레티나 foreground masking → 파워법칙 강조 (threshold 0.45, gamma 0.8) → TURBO colormap → 원본에 alpha blend (82%)

### 2.8 지원 모델 목록

| 아키텍처 | 역할 |
|---|---|
| `efficientnet_b5` | 기본 production 모델 |
| `resnet50` | 안정적인 baseline |
| `convnext_tiny` | 현대 아키텍처 비교군 |

---

### 2.9 전체 시스템 아키텍처 (풀스택)

> 출처: `task.md` (시스템 구성도), `pjja.md` (프로젝트 제안서)

```
Flutter 클라이언트
  UI (GoRouter)
  상태 관리 (Riverpod)
  네트워크 (Dio)
  결과 시각화 (Syncfusion Gauges / Photo View)
    │ REST API 요청 (이미지 업로드)
    ▼
FastAPI 백엔드
  ├─ 전처리 (OpenCV / NumPy)
  │    Circular Crop → Ben Graham normalization → Resize 512×512
  ├─ 품질 검사 / QuickQual task
  │    AI 모듈과 분리된 backend task로 수행
  ├─ AI 추론 (PyTorch / EfficientNet-B5)
  │    Sigmoid → abnormal_probability
  │    threshold → 정상 / 이상 레이블
  └─ XAI (Grad-CAM / Layer-CAM, 자체 구현)
       CAM overlay 생성
       foreground mask → power-law 강조 → TURBO colormap
       실패 시 xai_error_code=XAI_001 반환
    │ JSON 응답 (결과 + 설명 이미지)
    ▼
Flutter 결과 화면
  원본 이미지 | 판별 결과 + 이상 확률 점수 | Grad-CAM 설명 이미지

저장소
  SQLite 또는 JSON (회원별 분석 이력, 최대 20건)
```

**프로젝트 설계 원칙** (pjja.md)

- 목적: 보조 판별 시스템. 의료적 확정 진단이 아님을 결과 화면에 명시.
- 우선순위: **신뢰성·정확성 > 처리 속도**.
- 초기 범위: 정상/이상 이진 분류(referable DR 선별).
- 확장 방향: 다중 등급 분류, 모바일 앱 연동, 스마트폰 기반 보조 촬영 장치.

---

### 2.10 스프린트 계획 개요

> 출처: 사용자 제공 `Product Backlog.csv` (`C:\Users\dg203\Desktop\Product Backlog.csv`, 2026-05-06 확인)

| Sprint | 핵심 기능 | 현재 상태 |
|---|---|---|
| Sprint 1 | 파일 선택/미리보기/입력 화면, 이미지 크기 조정, 품질 감지, 픽셀 정규화, 판별 요청/결과, 설명 이미지 제공 | Done 9 |
| Sprint 2 | 분석 요청 전 입력 확인, 채널/색상 통일, 전처리 완료 이미지 추론 전달, 분석 결과 화면 구성/통합, 설명 이미지 실패 처리, 이미지 스케일업 모델 테스트 | Done 5 / In Review 1 / In Progress 1 |
| Sprint 3 | 이상 확률 점수, 주요 관심 영역 강조, 결과 섹션 구분, 결과 화면 상태 전환, 원본-설명 이미지 비교, 보조 판별 안내, 이미지 암호화 보관 | Current — In Progress 2 / To DO 5 |
| Sprint 4 | 업로드 이미지 재선택, 지원 형식 검증, 판별 실패 예외 처리, 시각화 해석 안내, 결과 화면 오류 메시지, 분석 이력 비교, 새 이미지 재분석 흐름 | To DO 7 |
| Sprint 5 | 회원별 이미지 및 결과 저장 | To DO 1 |

**품질 검사 상태**

`Product Backlog.csv`의 Sprint 1 항목에는 QuickQual(DenseNet121+SVM), Laplacian variance 248.929, brightness mean 126.870 기준의 저품질 이미지 입력 감지가 Done으로 기록돼 있다. 현재 AI 코드 기준 정책은 “AI 학습/평가에는 품질 필터 없음, 추론 품질 판정은 backend QuickQual task에서 처리”이다.

**결과 화면 4개 요소** (Sprint 2~3)

원본 이미지 / 판별 결과 라벨 / 이상 확률 점수 (0–100%) / Grad-CAM 설명 이미지

---

## 3. 변경 이력

### [SPRINT 1] — (v0.1.0 ~ v6)

### v0.1.0 — 초기 스캐폴드 구축

**내용**

- 프로젝트 구조 설계 (src layout, pyproject.toml, CLI entrypoints)
- EfficientNet-B3 기반 이진 분류 모델 구현
- APTOS + IDRiD manifest 빌더 구현
- 두 단계 학습 파이프라인 (head-only → finetune)
- 품질 검사 모듈 (블러, 밝기) 구현
- Grad-CAM 히트맵 생성 구현
- Flask 로컬 데모 서버 구현
- 단위 테스트 25개 작성

**결과**

- 기본 학습/추론/평가 파이프라인 동작 확인

---

### v0.1.1 — IDRiD Shadow Validation 도입

**배경**

초기 manifest는 APTOS 비중이 압도적으로 높아 val 분할이 APTOS 도메인에 편중되었다. 이 경우 val loss 기반 체크포인트 선택이 IDRiD 도메인 성능을 제대로 반영하지 못한다.

**내용**

- `manifest_variants.py` 추가: IDRiD 샘플의 일부를 val 분할에 강제 배치하는 shadow holdout 전략 구현
- `manifest_val_plus_idrid_shadow.csv` 생성
- `build_shadow_val_manifest.py` CLI 추가
- `efficientnet_b3_idrid_shadow_val.yaml` config 추가

**결과**

- val 분할에 IDRiD 103개 포함 → 도메인 편향 완화

---

### v0.1.2 — Shadow Val 재실험 (최고 성능 달성)

**배경**

Shadow val 전략 도입 후 재현성 확인을 위해 동일 설정으로 재실험.

**내용**

- `efficientnet_b3_idrid_shadow_val_rerun.yaml` config 추가
- historical shadow-val rerun checkpoint dir 생성. 현재 migrated artifact set에는 이 산출물이 존재하지 않는다.

**결과 (best epoch 6)**

| 지표 | val | test |
|---|---|---|
| AUROC | 0.9955 | 0.9915 |
| F1 | 0.9750 | 0.9617 |
| Sensitivity | 0.9658 | 0.9576 |
| Specificity | 0.9806 | 0.9657 |
| Accuracy | — | 0.9616 |

---

### v0.2.0 — EfficientNet-B5 전환 + Messidor 외부 테스트 지원

**배경**

- EfficientNet-B3(300px)보다 더 높은 해상도(456px)에서 미세 병변(미세혈관류, 삼출물)을 더 잘 포착할 수 있는 B5로 backbone 업그레이드.
- 학습 데이터와 완전히 분리된 외부 데이터셋(Messidor)으로 일반화 성능을 검증할 필요.

**내용**

- `models/profiles.py`: B3 프로파일 제거, B5 프로파일 추가
- `models/build.py`: B5 build/classifier 분기로 교체
- `configs/base.yaml`: 기본 아키텍처, 입력 크기, batch, lr을 B5 기준으로 갱신
- `data/manifest_builder.py`:
  - `_build_messidor_rows()` 추가 (Messidor-2 CSV 포맷 지원)
  - `build_manifest_frame(include_messidor=True)` 옵션 추가
  - `ManifestSummary`에 `external_test_rows` 필드 추가
- `cli/build_manifest.py`: `--include-messidor` 플래그 추가
- `configs/efficientnet_b5_messidor_ext.yaml` 신규 추가
- B3 전용 config 파일 2개 삭제
- 관련 테스트 전부 B5 기준으로 갱신

**Messidor 데이터 배치 규약**

```
data/raw/Messidor/
  messidor_data.csv        # 컬럼: image_id, adjudicated_dr_grade
  images/
    *.jpg / *.tif / *.png
```

---

### v0.3.0 — timm 백본 전환 + ECA/공간 어텐션 통합 + BUG-01/02 수정

**배경**

- 논문(멀티미디어학회논문지 28권 2호, 2025) 검토 결과 ECA + 공간 어텐션(CBAM 방식)을 모든 MBConv 블록에 균등 적용했을 때 QWK 92.85, Accuracy 86.89%로 최고 성능 달성.
- torchvision EfficientNet-B5의 SE Block을 ECA로 교체하려면 timm이 `se_layer=EcaModule` 키워드를 지원해 코드 변경이 최소화됨.
- 체크포인트 선택 기준(BUG-01)이 val_loss였으나, 스크리닝 목적에서는 sensitivity ≥ 0.95 조건하에 AUROC를 극대화하는 것이 임상적으로 타당.
- GradCAM 대상 레이어가 `model.features` 기준이었으나 timm 구조는 `model.blocks`를 사용해 ValueError가 발생(BUG-02).

**내용**

- `models/profiles.py`
  - `ModelProfile`에 `use_attention: bool`, `gradcam_target_layer: str` 필드 추가.
  - efficientnet_b5 프로파일을 timm 기준으로 전면 하드코딩: 448×448, bicubic, 23,155,054 params (ECA + SpatialAttn).
  - `get_weights_enum`에서 efficientnet_b5 항목 제거 (timm으로 이관).

- `models/build.py`
  - efficientnet_b5: `timm.create_model('efficientnet_b5', se_layer=EcaModule, num_classes=num_outputs)` 로 빌드.
  - `_SpatialAttnWrapper`: 개별 MBConv 블록을 감싸 출력에 `SpatialAttn(kernel_size=7)`를 적용하는 래퍼 모듈.
  - `_inject_spatial_attention()`: `model.blocks`의 7개 그룹 × 전체 블록에 래퍼 주입. 논문 ablation 결과 모든 블록 균등 적용이 부분 적용보다 우수.
  - `build_model`에 `use_attention: bool = False` 파라미터 추가.

- `xai/gradcam.py`
  - `resolve_default_target_layer`: `model.blocks` fallback 추가 → timm B5에서 `blocks[-1]` 반환.
  - `generate_gradcam`: `target_layer_name: str | None` 파라미터 추가 → `model.get_submodule(name)`으로 임의 레이어 지정 가능 (BUG-02 해결).

- `train/runner.py`
  - `MIN_SENSITIVITY = 0.95` 상수 추가.
  - 체크포인트 선택 기준: `val_loss < best_val_loss` → `val_sensitivity >= MIN_SENSITIVITY and val_auroc > best_val_auroc` (BUG-01 해결).
  - `run_training`, `run_split_evaluation` 양쪽에서 `use_attention` config 값을 `build_model`에 전달.
  - LOGGER 출력에 `val_sensitivity`, `val_auroc` 추가.

- `configs/base.yaml`
  - `data.resize_size`, `data.image_size`: 456 → 448 (timm B5 기본 입력 크기).
  - `model.use_attention: true` 추가.

**검증 (로컬 smoke test)**

```
classifier: Linear(in_features=2048, out_features=1, bias=True)  # head 정상 교체
blocks.6[0] type: _SpatialAttnWrapper                             # 공간 어텐션 주입 확인
backbone params: 504 / head params: 2                             # 파라미터 분리 정상
without spatial attn: 23,151,154                                  # ECA 단독
with spatial attn:    23,155,054  (delta +3,900)                  # + SpatialAttn 가중치
```

---

### v0.3.1 — BF16 AMP + 적응형 전처리 + Messidor 외부 평가

**배경**

- RTX 5070 Ti (Blackwell, SM 12.0)에서 FP16 AMP 사용 시 초기 logit이 NaN으로 발산하는 현상 확인. Blackwell 아키텍처는 FP16 오버플로에 취약하므로 BF16으로 전환 필요.
- APTOS로 학습한 모델을 Messidor에 적용하면 AUROC 0.505로 도메인 시프트가 심각함. 추론 시 CLAHE만 적용해도 개선 없음(AUROC 0.504) → 학습/추론 전처리 일관성이 필수.
- Messidor-1 (1200장)을 external_test split으로 추가해 도메인 일반화 검증 기반 마련.

**내용**

- `train/engine.py`
  - `_amp_dtype(device)` 추가: `torch.cuda.is_bf16_supported()` → True면 bfloat16, 아니면 float16.
  - `train_one_epoch`, `evaluate_one_epoch` autocast dtype을 `_amp_dtype()` 결과로 교체.
  - BF16 사용 시 `GradScaler` 비활성화(`_amp_needs_scaler` 조건).

- `data/transforms.py`
  - `FundusPreprocess` 클래스 추가: `A.CLAHE(clip_limit=2.0, tile_grid_size=(8,8))` + `A.GaussianBlur(blur_limit=(3,3))`, PIL→PIL.
  - `build_train_transform`, `build_eval_transform`에 `use_preprocessing: bool` 파라미터 추가.

- `data/manifest_builder.py`
  - Messidor-1 XLS 어노테이션 파일(Annotation_Base11-34.xls) 12개 파싱 지원.
  - grade 0 → normal(546), grade 1-3 → abnormal(654) 매핑.
  - `data/processed/messidor_manifest.csv` 생성 (1200 rows, split=external_test).

- `configs/messidor_eval.yaml` 신규: Messidor external_test 전용 평가 설정.
- `configs/base.yaml`: `use_preprocessing: true`, `device: cuda` 반영.

**학습 결과 (use_preprocessing: false, 18 epochs)**

| split | AUROC | sensitivity |
|---|---|---|
| val best (epoch 11) | 0.9764 | 0.9588 |
| test | 0.898 | 0.949 |
| Messidor external_test | 0.505 | — |

→ `use_preprocessing: true`로 재학습 필요.

---

### v0.4.0 — eye-project 통합 + EyeQ 교체(QuickQual) + 전처리 파이프라인 교체(Ben Graham)

**배경**

- fundus_dr_ai를 eye-project FastAPI 백엔드에 통합하는 과정에서 여러 구조적 결함이 발견됨.
- EyeQ MCF-Net 가중치(`DenseNet121_v3_v1.tar`)를 공식 저장소(github.com/HzFu/EyeQ)에서 취득 불가(OneDrive 링크 단절). 대체재 탐색 필요.
- 품질 검사가 전처리된 이미지에서 실행되는 순서 버그 발견(BUG-04).
- CLAHE를 흑색 여백이 있는 상태에서 적용하면 히스토그램이 왜곡된다는 문제 인식. Ben Graham 방식이 더 적합함.

**내용**

**(1) eye-project FastAPI 통합**

- `eye-project/backend/main.py` 신규 작성: FastAPI lifespan에서 `InferenceSession` 로드, `POST /predict`, `GET /health` 엔드포인트 구현.
- `eye-project/backend/backend_requirements`: Flask·grad-cam 제거, scikit-learn 추가.
- `eye-project/docker-compose.yml`: `./ai:/ai` 볼륨 마운트 추가, 기존 data/storage 볼륨 제거.
- `eye-project/backend/backend_Dockerfile`: `ENV PYTHONPATH=/ai` 추가.

**(2) 디렉토리 재구조화**

- `backend/fundus_dr_ai/`, `backend/configs/`, `backend/data/`, `backend/storage/` → `ai/` 하위로 이동.
- `eye-project/.gitignore`에 `ai/` 추가. AI 패키지·데이터·아티팩트 전체를 git 미추적으로 관리.
- `eye-project/setup.sh`: `ai/data`, `ai/storage`, `ai/artifacts/{checkpoints,predictions,heatmaps,quickqual}` 자동 생성으로 변경.

**(3) 미사용 파일 삭제**

| 삭제 파일 | 이유 |
|---|---|
| `fundus_dr_ai/web/` (전체) | Flask 데모 서버 — FastAPI로 대체 |
| `fundus_dr_ai/cli/serve_demo.py` | Flask 데모 CLI — 삭제된 web/ 의존 |
| `fundus_dr_ai/quality/eyeq.py` | QuickQual로 전면 교체 |

**(4) EyeQ → QuickQual 교체**

- `fundus_dr_ai/quality/quickqual.py` 신규 작성.
  - DenseNet121 (ImageNet 가중치, torchvision 자동 캐시) + SVM (`.pkl` 30 MB, GitHub Releases).
  - 동일한 3-class 레이블: Good / Usable / Reject.
  - 성능: EyeQ accuracy 88.50%, AUC 0.9687 (MCF-Net 88.00%, 0.9588 대비 우위).
  - `from_config()`: `quickqual.weights_path` 미설정·파일 없을 때 None 반환, graceful skip.
- `configs/base.yaml`: `eyeq.weights_path` → `quickqual.weights_path: artifacts/quickqual/quickqual_dn121_512.pkl`.
- `infer/pipeline.py`, `infer/service.py`, `train/runner.py`: `EyeQAssessor` → `QuickQualAssessor` 교체.
- `strict=False` 문제 근본 해결: QuickQual은 아키텍처와 가중치가 일치하며, 가중치 없을 때 안전하게 비활성화.
- 참고: Engelmann et al., "QuickQual" (arXiv 2307.13646, 2023).

**(5) eyeq 네이밍 전면 제거**

| 이전 | 이후 |
|---|---|
| `eyeq_grade` | `quality_grade` |
| `eyeq_confidence` | `quality_grade_confidence` |
| `eyeq_assessor` | `quality_assessor` |
| `eyeq_result` | `quality_result` |

**(6) 품질 검사 순서 수정 (BUG-04 해결)**

- `infer/service.py:predict_pil_image()`: `raw_image = np.asarray(original_image)` 추출 위치를 `FundusPreprocess` 호출 **전**으로 이동.
- 수정 전 파이프라인: 전처리 → raw_image 추출 → 품질 검사 (전처리된 이미지 기준 측정, 오류).
- 수정 후 파이프라인: raw_image 추출 → 품질 검사 (원본 기준) → 전처리 → 추론.

**(7) FundusPreprocess 교체: CLAHE + GaussianBlur → Circular Crop + Ben Graham**

- `data/transforms.py:FundusPreprocess` 전면 재작성.
  - `albumentations` 의존성 제거, `cv2` 단독 사용.
  - **1단계 — Circular Crop**: 흑색 여백 제거 후 정사각형 패딩. CLAHE 히스토그램 왜곡 원천 차단.
  - **2단계 — Ben Graham 정규화**: `sigmaX = max(H, W) / 30` (해상도 적응형). 비균등 조명을 전역 제거하고 혈관·병변 구조를 부각.
  - 기존 CLAHE + GaussianBlur 대비 개선점: 도메인 간 조명 차이(APTOS↔Messidor) 정규화 강화, Circular Crop 선행으로 히스토그램 왜곡 제거.
  - `crop_tol=7`, `ben_graham_weight=4.0`, `ben_graham_offset=128.0` 기본값 (논문 기준).

**추론 파이프라인 (v0.4.0 이후)**

```
이미지 입력
  → RGB 변환
  → raw_image 추출 (품질 검사용 원본 보존)
  → FundusPreprocess: Circular Crop → Ben Graham (sigmaX adaptive)
  → 기본 품질 검사 (blur_score, mean_brightness — 원본 기준)
  → QuickQual 품질 검사 (DenseNet121+SVM: Good / Usable / Reject)
  → eval_transform (Resize 448 → CenterCrop 448 → Normalize)
  → EfficientNet-B5 (ECA + Spatial Attention) forward
  → Sigmoid → abnormal_probability
  → threshold 0.5 → 예측 레이블
  → Grad-CAM 히트맵 생성 (선택)
  → InferenceResult (quality_grade, quality_grade_confidence 포함) 반환
```

### v0.5.0 — 오프라인 전처리 + 학습 최적화 + 품질 임계값 캘리브레이션 + 체크포인트 버전 관리

**배경**

- v0.4.0 학습 실행 시 GPU 사용률 9%, epoch당 50분 이상 소요. RTX 5070 Ti (16GB VRAM) 대비 심각한 저활용.
- 원인 분석: `use_preprocessing: true` 상태에서 매 이미지마다 Ben Graham + Circular Crop을 학습 중 CPU 실시간 수행 → 데이터 로딩 병목. 원본 이미지 해상도 그대로 로드하는 문제 병존.
- 품질 임계값 `blur_score_min: 4.5`가 사실상 필터 기능 없음 확인 (전체 5378장 중 필터링 0건).
- 학습 재실행마다 `best.pt`가 덮어씌워져 버전 구분 불가.

**내용**

**(1) 오프라인 전처리 파이프라인 도입**

- `prepare_messidor.py` 신규 작성: Messidor XLS 12개 파싱 → `messidor_data.csv` 생성, 이미지를 `images/` 플랫 디렉토리에 hardlink.
- `preprocess_images.py` 신규 작성: manifest 전체 이미지에 Ben Graham + Circular Crop + 448×448 리사이즈를 일괄 적용, `data/raw/processed/images/`에 PNG 저장, `manifest_preprocessed.csv` 생성.
- `FundusPreprocess`에 `output_size: int | None` 파라미터 추가 — 전처리 내 리사이즈를 선택적으로 통합. 학습(오프라인)과 추론(온라인) 모두 동일한 `FundusPreprocess(output_size=448)` 경로를 거쳐 파이프라인 일관성 확보.
- `configs/base.yaml`: `data.use_preprocessing: false` (학습 시 오프라인 이미지 사용), `data.preprocess_size: 448` 추가.

**(2) 추론 전처리 독립 제어**

- `configs/base.yaml`: `infer.use_preprocessing: true` 추가 — 추론 시 원본 이미지에 실시간 전처리 적용.
- `infer/service.py`: `use_preprocessing`을 `data` 섹션 대신 `infer` 섹션에서 우선 읽도록 수정. `preprocess_size`를 `data` 섹션에서 읽어 `FundusPreprocess(output_size=preprocess_size)` 생성.

**(3) 학습 DataLoader 최적화**

- `batch_size: 8 → 32`: 오프라인 전처리 후 VRAM 여유(14GB) 활용.
- `num_workers: 4 → 0`: Windows `spawn` 방식 오버헤드 제거. 오프라인 전처리 완료 후 per-image CPU 부담 감소로 단일 프로세스가 더 효율적.
- 결과: finetune 단계 GPU 사용률 9% → 99%.

**(4) Gradient Checkpointing 도입**

- `models/build.py:build_model()`: `grad_checkpointing: bool = False` 파라미터 추가. `True`일 때 `model.set_grad_checkpointing(True)` 호출 (timm 내장).
- `train/runner.py`: `config["model"].get("grad_checkpointing")` 읽어서 전달.
- `configs/base.yaml`: `model.grad_checkpointing: true`.
- 효과: finetune 역전파 시 중간 activation 저장 대신 재계산 → VRAM ~30% 절감. 연산량 약 20% 증가로 상쇄.

**(5) 품질 임계값 캘리브레이션**

- `calibrate_quality` CLI 실행 (전체 5378장 기준, 5th percentile).
- 결과를 `configs/base.yaml`에 반영:

| 항목 | 이전 | 이후 |
|---|---|---|
| `blur_score_min` | 4.5 | 248.929 |
| `brightness_mean_min` | 33.0 | 126.870 |

- 기존 값(4.5)으로는 전체 데이터셋에서 저품질 이미지가 단 한 장도 필터링되지 않음을 확인.

**(6) 체크포인트 버전 관리**

- `configs/base.yaml`: `project.version` 키 추가 (현재 `v1`).
- `train/runner.py`: 체크포인트 저장 경로를 `{checkpoint_dir}/{version}/`으로 변경. 버전 미설정 시 기존 동작 유지.
- `infer.checkpoint_path`: historical v1 checkpoint path로 업데이트. 현재 migrated artifact set에는 v1 산출물이 존재하지 않는다.

**추론 파이프라인 (v0.5.0 이후)**

```
이미지 입력
  → RGB 변환
  → raw_image 추출 (품질 검사용 원본 보존)
  → FundusPreprocess(output_size=448): Circular Crop → Ben Graham (sigmaX adaptive) → 448×448 리사이즈
  → 기본 품질 검사 (blur_score ≥ 248.9, mean_brightness ≥ 126.9 — 원본 기준)
  → QuickQual 품질 검사 (DenseNet121+SVM: Good / Usable / Reject)
  → eval_transform (Resize 448 → no-op → CenterCrop 448 → Normalize)
  → EfficientNet-B5 (ECA + Spatial Attention) forward
  → Sigmoid → abnormal_probability
  → threshold 0.5 → 예측 레이블
  → Grad-CAM 히트맵 생성 (선택)
  → InferenceResult 반환
```

---

### v3 — 패키지 리네이밍 + BUG-05 수정

**배경**

- v0.5.0에서 도입한 Gradient Checkpointing(`model.set_grad_checkpointing(True)`)이 `_SpatialAttnWrapper`와 비호환임을 v1 학습 실험에서 확인. 학습 자체는 진행되나 체크포인트가 저장되지 않아 summary `best_val_auroc: 0.5092`로 보고됨 (실제 로그에서는 AUROC 0.9947 달성).
- 체크포인트 저장 조건 `val_sensitivity ≥ MIN_SENSITIVITY(0.95) AND val_auroc > best`에서 MIN_SENSITIVITY가 너무 엄격해 finetune 전 epoch 동안 저장이 차단된 것이 근본 원인 (BUG-05).
- 패키지명 `fundus_dr_ai`가 역할을 모호하게 표현 → `drscreen`으로 변경.

**내용**

**(1) 패키지명 변경: `fundus_dr_ai` → `drscreen`**

- 디렉토리 `ai/fundus_dr_ai/` → `ai/drscreen/`
- 모든 Python import, config 문자열, CLI 진입점 일괄 치환.

**(2) Gradient Checkpointing 비활성화 (BUG-05 연계)**

- `configs/base.yaml`: `model.grad_checkpointing: true → false`.
- `_SpatialAttnWrapper`로 각 MBConv 블록을 감싼 상태에서 timm의 `set_grad_checkpointing`을 호출하면 gradient flow 불안정 발생. 향후 wrapper 구조를 유지하면서 checkpointing을 적용하려면 `torch.utils.checkpoint.checkpoint`를 wrapper 단위로 직접 적용해야 함.

**(3) BUG-05 수정 — `MIN_SENSITIVITY` 하드코딩**

- `train/runner.py:MIN_SENSITIVITY = 0.95` → `_DEFAULT_MIN_SENSITIVITY = 0.80`으로 완화, config `train.min_checkpoint_sensitivity`로 외부 주입 가능하도록 변경.
- 수정 전: finetune 전 epoch에서 sensitivity가 0.95를 넘지 못해 AUROC 0.9947임에도 체크포인트 미저장.
- 수정 후: sensitivity ≥ 0.80 조건에서 올바른 epoch에 저장.

**(4) DataLoader 파라미터 조정**

- `batch_size: 32 → 16`: grad_checkpointing 제거로 VRAM 여유가 사라짐. OOM 방지.
- `num_workers: 0 → 4`: 오프라인 전처리 완료 후 per-image CPU 부담이 없어 worker 4개가 유효함을 확인.

**v3 학습 결과 (현재 최고 체크포인트)**

| 항목 | 값 |
|---|---|
| 체크포인트 | `artifacts/runs/00_baselines_and_early/v3/checkpoints/best.pt` |
| best_epoch | 14 / 18 |
| val AUROC | **0.9967** |
| test AUROC | **0.9727** |
| test Sensitivity | 97.5% |
| test Specificity | 85.8% |
| Messidor AUROC | 0.6905 |
| Messidor Sensitivity | 7.3% |

**Messidor 도메인 시프트 분석**

val/test(APTOS 계열) AUROC 0.99 대비 Messidor 0.69로 큰 낙차 발생. 원인:
- Messidor: 프랑스 3개 병원, TIF 포맷, 다른 카메라 장비 → 색감·밝기 분포 이질적.
- Ben Graham 전처리로 조명 보정을 시도했으나 잔류 도메인 갭 존재.
- 모델이 APTOS/IDRiD 병변 텍스처에 과적합. Messidor 이상 이미지를 정상으로 분류(sensitivity 7.3%).

**다음 개발 방향**

처음 보는 도메인(카메라, 병원, 국가)에 대해 재학습 없이 적응하는 AI를 목표로 설정. FEAT-10 참조.

---

### v3 (Ablation) — 분류기 초기화 개선 + Dropout Ablation Study

**배경**

v3 학습 중 `train_loss=14.5605`가 관측됨. 비정상적으로 높은 초기 loss의 원인을 진단하고, 도메인 일반화 개선을 위해 dropout과 분류기 초기화 전략을 ablation으로 검증함.

**원인 분석 — train_loss=14.5605**

`BCEWithLogitsLoss(14.5, 0) ≈ 14.5`이므로 모델이 초기에 ±14 수준의 logit을 출력하고 있음.

EfficientNet-B5는 ImageNet으로 사전학습되었으나 안저(fundus) 이미지는 분포가 매우 다르다. Pretrained backbone이 추출하는 feature 벡터의 norm이 크고, Kaiming 초기화된 분류기 가중치와 결합하면 logit의 표준편차가 다음과 같이 추정된다.

$$\sigma_{\text{logit}} \approx \sqrt{2048 \times 0.031^2 \times \sigma_{\text{feat}}^2} \approx 14$$

기존에 `gradient_clip_norm: 1.0`과 warmup scheduler가 있어 학습 자체는 수렴했으나, 초기 loss가 크면 첫 번째 warmup epoch의 업데이트 품질이 저하됨.

**내용**

**(1) Classifier Zero Initialization**

분류기 Linear 레이어의 weight/bias를 0으로 초기화 → 초기 logit ≈ 0, loss ≈ log(2) ≈ 0.693.

- `train/runner.py`: 모델 빌드 직후 분류기 파라미터를 `nn.init.zeros_`로 초기화.
- `configs/base.yaml`: `model.zero_init_classifier` 플래그 추가 (기본값 `false`).

```python
if bool(config["model"].get("zero_init_classifier", False)):
    classifier = get_classifier_module(architecture, model)
    for module in classifier.modules():
        if isinstance(module, nn.Linear):
            nn.init.zeros_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
```

**(2) Global Best 자동 승격 로직**

학습 완료 후 version best(`artifacts/runs/00_baselines_and_early/v3/checkpoints/best.pt`)가 global best(`best.pt`)보다 val_auroc가 높으면 자동으로 복사.

- `train/runner.py`: `run_training()` 종료 시 비교 및 `shutil.copy2` 수행.
- summary에 `promoted_to_global_best` 필드 추가.

**Ablation Study — Dropout × Zero Init (Messidor external_test 기준)**

| 실험 | classifier_dropout | zero_init_classifier | val AUROC | Messidor AUROC |
|---|---|---|---|---|
| A (기존 global best) | 0.0 | X | **0.9967** | **0.694** |
| B | 0.4 | O | 0.9574 | 0.581 |
| C | 0.4 | X | 0.9564 | 0.535 |
| D | 0.0 | O | 0.9858 | 0.549 |

**결론**

- **Dropout 0.4**: val AUROC 저하 + Messidor AUROC 0.159 감소. 도메인 일반화에 역효과. 원인: dropout이 APTOS/IDRiD 병변 텍스처에 특화된 feature 학습을 방해하지 않고, 오히려 일반화에 필요한 표현을 손상시킨 것으로 추정.
- **Zero init**: 단독으로는 Messidor AUROC 개선 없음 (0.694 → 0.549). 초기 loss 안정화 효과는 있으나 최종 수렴 품질에는 neutral하거나 소폭 부정적.
- **최종 설정**: `classifier_dropout: 0.0`, `zero_init_classifier: false` 유지 (기존 global best 조건 재현이 목표).

**현재 최고 체크포인트 (변경 없음)**

| 항목 | 값 |
|---|---|
| 체크포인트 | `artifacts/checkpoints/best.pt` (legacy/global artifact) |
| best_epoch | 14 |
| val AUROC | 0.9967 |
| Messidor AUROC | 0.694 |
| Messidor Sensitivity (threshold=0.19) | 55.5% |

**다음 개발 방향**

Messidor AUROC 0.694는 임상 적용 기준 미달. FEAT-10 (MixStyle 도메인 일반화) 진행 필요.

---

### v4 — SimCLR SSL 사전학습 + 도메인별 평가 분리

**배경**

v0.5.2 ablation 결과 Messidor external_test AUROC가 0.549로 사실상 랜덤 수준임을 확인. 원인 분석:

- v3 체크포인트는 APTOS + IDRiD만으로 학습됨. Messidor는 프랑스 3개 병원의 TIF 이미지로, 카메라·조명·대비 분포가 APTOS/IDRiD와 이질적.
- Ben Graham 전처리로 조명 편차를 일부 보정했으나, 모델이 APTOS/IDRiD 병변 텍스처에 과적합되어 Messidor 분포를 학습한 적 없음.
- 기존 평가 JSON에 도메인별 분리 수치가 없어 도메인별 성능 진단이 불가했음.

목표: **재학습 없이 새 도메인에도 적응 가능한 도메인 강건 표현 학습.** P1 논문(MAE 기반 연합학습, MESSIDOR 포함 5개 도메인 실험)에서 SSL 사전학습이 non-IID 도메인 간 일반화를 최대 9.6%p 개선했다는 근거를 참고.

**내용**

**(1) 전처리 일관성 오진 수정 및 문서화**

기존 backlog P1 항목은 "학습/추론 전처리 불일치"를 Messidor 저성능의 원인으로 지목했으나, 코드 검증 결과 불일치가 이미 해소된 상태였음:

- 학습: `manifest_preprocessed.csv` (Ben Graham 오프라인 적용) + `use_preprocessing: false` → 동등하게 Ben Graham 적용됨.
- 추론: raw 이미지 + `infer.use_preprocessing: true` → live Ben Graham 적용.

`docs/AI_HANDOFF.md` 및 `docs/architecture.md`에 명시적 경고 추가:

> `data.use_preprocessing: false`는 의도된 값. `true`로 변경하려면 `manifest.csv`(raw)로 함께 전환해야 하며, 그렇지 않으면 Ben Graham이 이중 적용됨.

실제 Messidor 저성능 원인은 전처리 불일치가 아닌 순수 도메인 시프트였음을 확인.

**(2) 도메인별 평가 리포트 분리 (FEAT-01 구현)**

`manifest_preprocessed.csv`에 이미 `domain` 컬럼(APTOS / IDRiD / Messidor)이 존재함을 확인.

- `drscreen/train/runner.py:run_split_evaluation()`: `dataset.frame["domain"]` 컬럼을 읽어 도메인별 logit/target 분리 후 `compute_binary_classification_metrics`를 각 도메인에 적용.
- 평가 JSON 출력에 `domain_breakdown` 필드 추가. `domain` 컬럼이 없는 manifest와 하위 호환.

```json
"domain_breakdown": {
  "APTOS":    { "auroc": 0.9904, "sensitivity": 0.9760, "specificity": 0.9598, "num_examples": 366 },
  "IDRiD":    { "auroc": 0.8947, "sensitivity": 0.8841, "specificity": 0.5588, "num_examples": 103 },
  "Messidor": { "auroc": 0.7608, "sensitivity": 0.8593, "specificity": 0.3205, "num_examples": 1200 }
}
```

**(3) SimCLR SSL 사전학습 인프라 구축**

비지도 대조 학습(SimCLR, Chen et al. 2020)을 단일 머신 오프라인 사전학습 단계로 구현. P1 논문의 MAE 기반 접근을 단일 GPU 환경에 맞게 조정한 것으로, 핵심 아이디어(레이블 없이 다중 도메인 이미지를 동시에 학습하여 도메인 불변 표현 획득)는 동일.

신규 모듈:

| 파일 | 역할 |
|---|---|
| `drscreen/ssl/loss.py` | NT-Xent 대조 손실. FP16 오버플로 방지를 위해 내부에서 `.float()` 강제 캐스트 (`1/temperature ≈ 14.3` → `exp(14.3) ≈ 1.6M` > FP16 최댓값 65504). |
| `drscreen/ssl/augmentations.py` | `SSLAugmentationPair`: Ben Graham 1회 적용 후 독립적인 강화 증강 2회 → 동일 이미지의 두 뷰 생성. `use_preprocessing` 파라미터로 오프라인 전처리 이미지 재사용 가능. |
| `drscreen/ssl/dataset.py` | `SSLManifestDataset`: manifest의 모든 행을 split 구분 없이 로딩. Messidor `external_test` 이미지를 레이블 없이 사전학습에 참여시키는 것이 핵심. |
| `drscreen/ssl/runner.py` | `run_ssl_pretraining()`: SimCLR 학습 루프. EfficientNet-B5(`num_classes=0`) + 2층 MLP Projection head. epoch 내 20% 단위 배치 진행 로그 출력. |
| `drscreen/cli/pretrain.py` | CLI 진입점 (`python -m drscreen.cli.pretrain --config configs/ssl_pretrain.yaml`) |
| `configs/ssl_pretrain.yaml` | SSL 전용 설정. `manifest_preprocessed.csv` + `use_preprocessing: false` 조합으로 live FundusPreprocess CPU 병목 제거. |

**현재 구조 주의**: 위 표는 v4 당시의 SSL 구현 경로다. 이후 RETFound 관련 정리 과정에서 SSL 코드 구조가 바뀌었고, 현재 재현용 SimCLR 경로는 `drscreen/ssl/simclr.py`, `drscreen/ssl/trainer.py`, `drscreen/cli/ssl_pretrain.py`, `configs/ssl_simclr_pretrain.yaml` 기준이다.

SimCLR 설계 근거:

- 두 뷰의 유사도를 높이고 배치 내 다른 이미지와는 멀어지도록 학습 → 색상·밝기 단서가 아닌 구조적 특징(병변 형태, 혈관 패턴) 위주로 표현 학습.
- Messidor 이미지가 비지도 사전학습에 포함되므로 fine-tuning 전에 Messidor 특유의 도메인 분포를 표현 공간에 인코딩.

AMP 관련 수정 (`drscreen/ssl/runner.py`):

```python
amp_dtype = (
    torch.bfloat16
    if device.type == "cuda" and torch.cuda.is_bf16_supported()
    else torch.float16
)
```

RTX 5070 Ti (Blackwell)는 BF16 지원. FP16 기본값으로는 NT-Xent 분모 계산에서 오버플로 → NaN 발생 (1 epoch 진행 불가 현상의 원인). `torch.autocast` 기본 dtype이 CUDA에서 FP16이므로 명시적 BF16 지정 필요.

**SSL 학습 설정**

| 항목 | 값 |
|---|---|
| 데이터 | 전체 5,378장 (APTOS + IDRiD + Messidor, split 무관) |
| 배치 크기 | 32 |
| 이미지 크기 | 224 × 224 |
| Projection head | Linear(2048→2048) → BN → ReLU → Linear(2048→128) |
| Epochs | 100 |
| Optimizer | AdamW (lr=3e-4, weight_decay=1e-4) |
| Scheduler | CosineAnnealingLR |
| Temperature | 0.07 |
| AMP | BF16 (RTX 5070 Ti) |
| 최종 best_loss | **0.4405** (초기 ~4.7 대비 90% 이상 감소) |

**(4) 지도학습 파인튜닝에 SSL 백본 로딩 지원**

- `drscreen/train/runner.py:run_training()`: `config["train"].get("pretrained_backbone_path")` 읽어 `strict=False`로 백본 가중치 로딩.
- classifier 키 2개만 missing (SSL backbone은 `num_classes=0`이므로 classifier 없음) — 정상.
- `zero_init_classifier: true`와 조합하여 classifier는 0으로 초기화 후 head-only phase에서 학습.
- `configs/base.yaml`에 주석으로 사용법 명시:

```yaml
# pretrained_backbone_path: artifacts/ssl/backbone_best.pt
```

**(5) v4 학습 결과**

- `project.version: v4`, historical `infer.checkpoint_path` pointed to v4 best checkpoint. Current migrated artifact set contains v4 evaluation JSONs only; v4 checkpoint is not present.
- SSL 백본(best_loss=0.440) 로딩 후 동일 지도학습 파이프라인(head 3 epochs + finetune 15 epochs) 실행.
- best_epoch: 9, val AUROC: 0.9941.

**v3 → v4 성능 비교**

| 지표 | v3 (체크포인트 복원) | v4 (SSL 파인튜닝) | 변화 |
|---|---|---|---|
| val AUROC | 0.9967 | 0.9941 | -0.003 |
| test AUROC | 0.9727 | **0.9826** | **+0.010** |
| test Accuracy | 0.9168 | **0.9254** | +0.009 |
| test Sensitivity | 0.9746 | 0.9492 | -0.025 |
| test Specificity | 0.8584 | **0.9013** | **+0.043** |
| Messidor AUROC | 0.549 | **0.761** | **+0.212** ✅ |
| Messidor Sensitivity | 0.761 | **0.859** | +0.098 |
| Messidor Specificity | 0.267 | 0.321 | +0.054 |

도메인별 breakdown (v4 test):

| 도메인 | AUROC | Sensitivity | Specificity | N |
|---|---|---|---|---|
| APTOS | 0.9904 | 0.9760 | 0.9598 | 366 |
| IDRiD | 0.8947 | 0.8841 | 0.5588 | 103 |
| Messidor | 0.7608 | 0.8593 | 0.3205 | 1200 |

**Messidor AUROC 0.549 → 0.761: backlog 목표(≥ 0.75) 달성.**

내부 sensitivity 소폭 하락(0.9746 → 0.9492)은 v4가 specificity를 더 균형 있게 학습한 결과. optimal_threshold=0.93으로 조정하면 스크리닝 민감도 요건을 충족할 수 있음.

**신뢰성 주의 — SSL 도메인 노출 문제**

현재 `SSLManifestDataset`은 split 구분 없이 전체 5,378장을 로딩하므로 **Messidor 1,200장(external_test)이 레이블 없이 SSL 사전학습에 포함됐다.** SSL은 레이블을 사용하지 않지만, 인코더가 Messidor의 이미지 분포(색감·대비·카메라 특성)를 표현 공간에 인코딩하게 된다. 평가 시점에 모델이 Messidor 도메인을 완전히 처음 보는 것이 아니므로 AUROC 0.761이 과대평가일 가능성이 있다.

진짜 일반화 성능을 검증하려면 아래 ablation이 필요하다:

| 실험 | SSL 데이터 | Messidor AUROC | vs v3 | 해석 |
|---|---|---|---|---|
| v3 (baseline) | 없음 | 0.549 | — | SSL 없음 |
| v4b (실험 B) | APTOS + IDRiD만 | **0.679** | +0.130 | 순수 SSL 효과 |
| v4 (실험 A) | APTOS + IDRiD + Messidor | **0.761** | +0.212 | SSL + 도메인 노출 |

**결론**: v4b가 v4 대비 0.082 낮다 (기준 ±0.03 초과). 두 요인이 모두 유의미하게 기여.
- SSL 자체 효과: +0.130 (v3 → v4b)
- 도메인 노출 추가 효과: +0.082 (v4b → v4)
- v4의 개선이 단순 도메인 노출만으로 설명되지는 않으나, Messidor가 SSL 사전학습에 포함된 것이 일부 이점을 제공한다.

**다음 개발 방향**

- ~~ablation(실험 B) 실행으로 v4 개선의 신뢰성 검증~~ → 완료 (v4b AUROC 0.679 확인).
- Messidor specificity가 0.32로 여전히 낮음 — 정상 이미지 과검출(false positive) 잔존. threshold 튜닝(FEAT-02) 또는 추가 도메인 augmentation으로 보완 가능.
- IDRiD specificity 0.56: 소규모 데이터셋 특성 반영, 추가 실험 필요.
- 5-class 분류 확장(backlog P3) 진입 가능한 시점.

---

### v4.1 — se_layer 어텐션 통합 재설계 + RandomResizedCrop 조정

**배경**

- v0.3.0에서 도입한 `_SpatialAttnWrapper` / `_inject_spatial_attention` 방식은 어텐션을 MBConv 블록 **외부**에 래퍼로 적용하는 구조였다. 이 경우 `model.blocks[-1]` 출력이 어텐션이 이미 적용된 표면이 되어, Grad-CAM 타깃으로 사용할 때 순수 residual 출력이 아닌 변조된 표면을 후킹하게 되는 문제가 있었다.
- `RandomResizedCrop scale=(0.7, 1.0)` 설정이 이미지 면적의 최대 30%를 제거 가능 → 망막 주변부 병변(미세혈관류, 출혈) 손실 위험이 있음을 확인.

**내용**

**(1) 어텐션 통합 방식 교체: `_SpatialAttnWrapper` → `_EcaSpatialAttn` as `se_layer`**

- `_SpatialAttnWrapper` 클래스 및 `_inject_spatial_attention()` 함수 제거.
- `_EcaSpatialAttn(nn.Module)` 신규 작성: `EcaModule` + `SpatialAttn(kernel_size=7)`을 순차 적용하는 단일 모듈.
- `timm.create_model('efficientnet_b5', se_layer=_EcaSpatialAttn)` 방식으로 주입 → 어텐션이 MBConv 내부 SE 위치에 통합됨.
- 효과: `model.blocks[-1]` 출력이 표준 residual 출력으로 유지 → Grad-CAM 타깃이 의도한 레이어를 정확히 반영.
- `use_attention=False` 시 기존대로 `EcaModule`만 사용.

```python
# 이전 (블록 외부 래퍼)
model = timm.create_model('efficientnet_b5', se_layer=EcaModule, ...)
if use_attention:
    _inject_spatial_attention(model)

# v0.6.1 (블록 내부 통합)
se_layer = _EcaSpatialAttn if use_attention else EcaModule
model = timm.create_model('efficientnet_b5', se_layer=se_layer, ...)
```

**(2) RandomResizedCrop scale 조정**

- `data/transforms.py`: `scale=(0.7, 1.0)` → `scale=(0.8, 1.0)`.
- 하한 0.7에서는 이미지 면적의 최대 30%가 제거될 수 있어 망막 주변부 병변 손실 위험. 0.8로 상향해 주변부 병변 보존.

**(3) profiles.py rationale 갱신**

- `efficientnet_b5` 프로파일의 rationale 문구를 se_layer 통합 방식 기준으로 업데이트.

**v4.1 학습 결과 (현재 최고 체크포인트)**

| 지표 | v4 | v4.1 | 변화 |
|---|---|---|---|
| val AUROC | 0.9941 | **0.9973** | +0.003 |
| Messidor AUROC | 0.761 | **0.802** | **+0.041** ✅ |

- 체크포인트: `artifacts/runs/01_ssl_lineage/v4.1/checkpoints/best.pt` (global `best.pt`로 자동 승격)
- Messidor AUROC 0.80 목표 최초 달성.

**Temperature Scaling 실험 (성능 향상 없음)**

val 분할 NLL 최소화로 T=1.894를 산출해 external_test에 적용했다. threshold=0.5 기준 지표는 raw와 완전히 동일했으며, Youden's J 최적점(sensitivity 0.638, specificity 0.861)도 변하지 않았다. Temperature scaling은 threshold 튜닝의 수학적 재포장으로, AUROC 0.802가 결정하는 sensitivity 상한을 바꾸지 못한다. 근본 원인은 도메인 시프트이므로 FEAT-10(MixStyle)이 필요하다.

**FEAT-10-A MixStyle 구현 시도 (미해결 — loss 폭발)**

MixStyle(Zhou et al., ICLR 2021)을 EfficientNet-B5에 통합 시도. 현재 loss 폭발 현상이 발생하며 원인 조사 중.

*구현 경과*

1. `drscreen/models/mixstyle.py` 신규 작성, `build.py`에 `_inject_mixstyle()` 추가.
2. forward hook 방식으로 `model.blocks[0:3]`에 MixStyle 삽입 — `nn.Sequential` 래핑은 state_dict 키를 변경하므로 hook 방식 채택.
3. MixStyle 모듈은 `model.add_module(f"_mixstyle_{i}", ms)`로 등록해 `train()/eval()` 전파를 보장.

*loss 폭발 현상*

모든 시도에서 head phase 첫 epoch부터 train_loss가 5천만~1억 수준으로 폭발. 정상 범위는 ~0.7.

| 시도 | MixStyle forward 내용 | train_loss |
|---|---|---|
| 1차 | normalize → mix stats → denormalize | ~5.3억 |
| 2차 | mean-only mixing (variance 미사용) | ~7600만 |
| 3차 | detached mean shift | ~7900만 |
| 4차 | **pure identity** (`return x`) | ~9800만 |
| 5차 | `use_mixstyle: false` (hook 미등록) | ~5500만 |

5차 시도에서 MixStyle 코드가 전혀 실행되지 않는데도 loss가 폭발하므로, **MixStyle 자체는 원인이 아님**.

*진단 결과*

- `debug_loss.py`로 모델 단독 테스트: 랜덤/실제 데이터 모두 loss=0.6931 (정상), 1 step 후 0.6845 (정상 감소).
- 배치별 디버그: batch 0 정상, **batch 1에서 logit이 -60억까지 폭발**. classifier 가중치가 ±0.0002 수준인 상태에서 이 값은 backbone feature가 극단적으로 커졌음을 시사.
- 단일 배치 forward는 정상 → 학습 루프 내 반복 과정에서 모델 상태가 변질되는 것으로 추정.

*v4.1과의 config 차이*

- `project.version`: v4.1 → v5 (checkpoint 경로만 변경)
- `model.use_mixstyle`: 신규 추가 (false)
- `model.zero_init_classifier`: 신규 추가 (true) — 단독 테스트에서는 정상 동작 확인
- `train.pretrained_backbone_path`: SSL backbone 주석 처리 (v0.6.0에서 도입했으나 v0.6.1 아키텍처와 키 불일치로 비활성화)

*다음 조사 방향*

1. 학습 루프 내 배치별 backbone feature 크기 추적 — batch 1에서 feature가 폭발하는지 확인.
2. config을 v4.1 상태로 복원(zero_init_classifier 제거, version 복원)해 재현 여부 확인.
3. 재현 시 engine.py 또는 runner.py의 head-only 학습 로직에서 BatchNorm 모드 전환 등 미묘한 상태 변질 가능성 조사.

---

### v5 — MixStyle 코드 정리 + MixStyle 전략 폐기

**배경**

v0.6.1에서 MixStyle 구현 시도 중 head-only 단계 loss 폭발(~32B) 현상이 발생했으나 MixStyle 코드 자체가 원인이 아님을 확인. `zero_init_classifier: false`로 복원 후 MixStyle을 정식으로 재작성하여 v5 학습을 완료하고 Messidor 평가를 수행.

**내용**

**(1) head-only loss 폭발 원인 확인**

`zero_init_classifier: false`(v4.1 기본값) 복원 후 v5 학습 실행 결과 head-only 단계에서 여전히 수백억대 loss가 관측됨. 이로써 `zero_init_classifier`가 원인이 아님이 확인됨.

실제 원인: head-only 단계에서 backbone이 eval 모드(BN running stat 고정)로 frozen된 상태에서 ImageNet 학습 통계가 안저 이미지의 분포와 극단적으로 이질적이어 backbone feature magnitude가 폭발. logit이 -60억 수준까지 발산.

finetune 단계에서 backbone도 학습에 참여하면 즉각 정상화(epoch 4: loss 1.21 → epoch 9: loss 0.20). head-only 단계의 폭발이 최종 수렴을 막지 않으므로 구조적 허용 범위로 판단.

**(2) 코드 정리**

| 파일 | 변경 내용 |
|---|---|
| `drscreen/train/engine.py` | DEBUG 블록 34줄 전체 제거 (`print`, `sys.exit`, `forward_features` 재실행) |
| `drscreen/models/mixstyle.py` | Zhou et al. 논문 기준 전면 재작성 (FP32 통계 계산, dtype 복원, eval/batch≤1 identity 보장) |
| `drscreen/models/build.py` | MixStyle 주입 위치 변경: top-level `model.add_module` → `model.blocks[i].add_module` (block group submodule). top-level `children()` 순회 오염 방지. |

**(3) v5 학습 결과**

| epoch | phase | train_loss | val_auroc | val_sensitivity |
|---|---|---|---|---|
| 1–3 | head | ~26B | ~0.47 | (폭발) |
| 4 | finetune | 1.21 | 0.862 | 0.789 |
| **9** | finetune | **0.20** | **0.9973** | **0.985** |

- best_epoch: 9, val AUROC: 0.9973 (v4.1과 동률)
- `promoted_to_global_best: False` — global best 갱신 없음, v4.1 유지

**(4) v5 Messidor 평가 결과**

| 지표 | v4.1 | v5 (MixStyle) | 변화 |
|---|---|---|---|
| Messidor AUROC | **0.802** | 0.773 | **-0.029** |
| Sensitivity (threshold=0.5) | — | 0.563 | — |
| Specificity (threshold=0.5) | — | 0.890 | — |

**MixStyle이 Messidor AUROC를 오히려 하락시킴.**

**(5) MixStyle 폐기 근거**

MixStyle의 효과는 배치 내에 여러 도메인 샘플이 섞여 있을 때 cross-domain 통계 혼합이 발생하는 것을 전제한다. 현재 학습 데이터는 APTOS + IDRiD만으로 구성되어 있어 배치 내 도메인 다양성이 부족하다. 결과적으로 MixStyle이 동일 도메인 내 샘플끼리 통계를 교환하게 되어 병변 패턴을 교란시키는 방향으로 작용했다. sensitivity 하락(환자 과검출 증가)이 이를 뒷받침한다.

**결론:** `use_mixstyle` 옵션은 코드에 유지하되(미래 multi-domain 학습 시 재시도 가능), 현재 데이터 구성에서는 활성화하지 않는다. v4.1이 global best로 유지된다.

---

### v6 — Focal Loss 도입 + Ablation (alpha vs gamma 기여도 분리)

**배경**

v4.1 체크포인트 기준 Messidor external_test sensitivity가 0.524(threshold=0.5)로, 스크리닝 도구의 임상 요건에 미달한다. temperature calibration(T=1.894) 적용 및 threshold를 0.11로 낮춰도 0.638이 한계였다. AUROC 0.8018은 판별 능력이 존재함을 나타내지만, 모델이 Messidor 양성 샘플에 일관되게 낮은 확률을 할당하는 구조적 문제가 있었다.

Focal Loss는 낮은 확률이 할당된 hard example에 gradient를 집중시키는 특성상, 도메인 시프트로 인해 저확률이 할당되는 Messidor 양성 샘플에 유효할 것으로 판단했다. 추가로 alpha(positive class weighting)와 gamma(hard example focusing)의 기여도를 분리하기 위해 세 가지 ablation을 설계했다.

**내용**

**(1) BinaryFocalLoss 구현**

- `drscreen/train/loss.py` 신규 작성.
  - `FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)`
  - `gamma`: focusing parameter. 0이면 weighted BCE와 동일.
  - `alpha`: positive class 가중치. None이면 class weighting 비활성.

- `drscreen/train/runner.py:_build_criterion()` 수정.
  - `train.loss: bce | focal` 분기 추가.
  - `bce`는 기존 `BCEWithLogitsLoss` 유지, 선택적 `pos_weight` 지원 추가.
  - `focal`은 `BinaryFocalLoss(gamma, alpha)` 반환.
  - `criterion.to(device)` 추가: `pos_weight` 텐서의 device 불일치 버그 선제 방지.

**(2) Ablation 실험 설계**

세 가지 config를 v4.1 체크포인트에서 fine-tuning:

| 실험 | gamma | alpha | 목적 |
|---|---|---|---|
| `v6_gamma_only` | 2.0 | None | gamma 단독 효과 |
| `v6_alpha_only` | 0.0 | 0.75 | alpha 단독 효과 (= weighted BCE) |
| `v6` (full) | 2.0 | 0.75 | 두 효과 합산 |

공통 설정: `head_epochs: 0` (v4.1 head 재사용), `finetune_epochs: 15`, `backbone_lr: 4e-05` (v4.1 수렴 상태 반영하여 절반), `use_mixstyle: false` (v5 회귀 확인으로 제거).

**Ablation 결과 — Messidor external_test**

| 실험 | Messidor AUROC | Sensitivity@0.5 | Specificity@0.5 |
|---|---|---|---|
| v4.1 (baseline) | 0.8018 | 0.524 | 0.958 |
| v6_gamma_only | 0.8231 | 0.586 | 0.943 |
| v6_alpha_only | **0.8697** | 0.705 | 0.910 |
| v6 full | 0.8616 | **0.769** | 0.824 |

**내부 테스트셋 (APTOS + IDRiD)**

| 실험 | AUROC | Sensitivity@0.5 | Specificity@0.5 |
|---|---|---|---|
| v4 | 0.9826 | 0.949 | 0.901 |
| v6_gamma_only | **0.9931** | 0.962 | 0.927 |
| v6_alpha_only | 0.9893 | 0.979 | 0.901 |
| v6 full | 0.9919 | **0.983** | 0.884 |

**기여도 분석**

v4.1 대비 Messidor AUROC 개선을 분해하면:

- gamma 단독: +0.021 — hard example focusing 효과. 소폭 개선.
- alpha 단독: +0.068 — positive class weighting 효과. 개선의 대부분을 설명.
- full 조합: +0.060 — alpha 단독(0.870)보다 AUROC가 오히려 낮음. gamma와 alpha가 단순 가산되지 않음.

alpha=0.75는 훈련 중 positive 샘플의 gradient를 일괄적으로 강화하여 모델이 전반적으로 positive를 더 강하게 예측하도록 유도한다. 이는 threshold를 낮추는 것과 유사한 메커니즘이며, Messidor 특화 일반화가 아닌 도메인 전반의 sensitivity 향상이다. 단, AUROC 개선(+0.068)은 threshold 독립 지표이므로 판별 능력 자체의 향상도 일부 포함된다.

Messidor 학습 관여도: 레이블 0%. SSL 단계(v0.6.0)에서 비레이블 이미지 22%(1,200/5,378장)가 백본에 인코딩된 상태를 v4.1 경유로 상속.

**현재 best 체크포인트 (v6_alpha_only)**

| 항목 | 값 |
|---|---|
| 체크포인트 | `artifacts/runs/01_ssl_lineage/v6_alpha_only/checkpoints/best.pt` |
| val AUROC | 0.9893 |
| Messidor AUROC | **0.8697** |
| Messidor Sensitivity@0.5 | 0.705 |
| Messidor Specificity@0.5 | 0.910 |
| 내부 test AUROC | 0.9893 |
| 내부 test Sensitivity@0.5 | 0.979 |

v6_alpha_only를 global best로 선정한 근거: Messidor AUROC 최고(0.870), 내부 specificity 회귀 없음(0.901), sensitivity-specificity 균형.

**다음 개발 방향**

- Messidor sensitivity 0.705는 스크리닝 목적의 임상 기준(≥0.80) 미달. 근본 해결을 위해 Messidor 레이블 데이터를 지도학습에 편입하는 방안 검토 필요 (단, 외부 평가셋 교체 선행 필요).
- alpha가 주도적 요인임이 확인됨 — threshold 조정으로 동일 효과를 낼 수 있으므로, 추론 시 calibrated threshold 적용을 병행 검토.
- IDRiD specificity 하락(v6 full 기준 0.35) 잔존 — 소규모 음성 샘플(34개) 문제로 통계적 신뢰도 낮음.

---
**[SPRINT 1 종료] — (2026.04.09)**
---

### [SPRINT 2] — v7_messidor_train

#### 배경

v6_alpha_only는 Messidor AUROC 0.8697, sensitivity 0.705로 스크리닝 임상 기준(≥0.80) 미달. 근본 원인은 학습 도메인(APTOS/IDRiD)과 외부 테스트 도메인(Messidor) 간 도메인 시프트. 두 가지 변화를 동시 적용:

1. **Messidor → train 편입**: 외부 평가셋에서 제외하고 지도학습에 포함. `train_exclude_domains: [Messidor]` 제거.
2. **DDR → external_test 교체**: 중국 23개 병원, 12,522장, ~1:1 클래스 균형. 기존 Messidor보다 규모와 도메인 다양성 모두 우세.

#### 데이터 변경

| 항목 | 이전 | 변경 후 |
|---|---|---|
| train | APTOS + IDRiD (4,543장) | APTOS + IDRiD + Messidor (5,743장) |
| external_test | Messidor (1,200장) | DDR (12,522장) |

#### 코드 변경

- `manifest_builder.py`: `_build_ddr_rows()` 추가, `_build_messidor_rows(split=...)` 파라미터화
- `build_manifest.py`: `--messidor-as-train`, `--include-ddr` CLI 플래그 추가
- `configs/v7_messidor_train.yaml`: `train_exclude_domains: []`, focal α=0.75, γ=0.0
- `service.py`: XAI 실패 시 `xai_error_code="XAI_001"` payload 반환

#### 학습 결과

| 항목 | 값 |
|---|---|
| 체크포인트 | `artifacts/runs/02_domain_generalization/v7_messidor_train/checkpoints/best.pt` |
| best_epoch | 12 |
| val AUROC | 0.9983 |

#### DDR external_test 평가 결과

| 지표 | threshold=0.5 | optimal threshold=0.09 |
|---|---|---|
| AUROC | **0.8725** | 0.8725 |
| Sensitivity | 0.5914 | **0.7626** |
| Specificity | 0.9603 | 0.8417 |
| F1 | 0.7251 | 0.7939 |
| Accuracy | 0.7760 | 0.8022 |

#### 분석

- **AUROC 기준 신규 best** (0.8725 > v6_alpha_only Messidor 0.8697), 10배 큰 데이터셋에서 달성
- **Threshold 이슈**: optimal threshold 0.09로 극단적으로 낮음. 모델이 DDR 이상 이미지에 낮은 확률을 부여하는 경향 → 도메인 시프트 잔존. 배포 시 threshold=0.5 사용 불가, 적응형 threshold 또는 캘리브레이션 필요.
- val AUROC는 v6 대비 0.0007 소폭 하락(0.9990→0.9983) — Messidor 편입으로 val 분포와 미세 불일치 발생, 허용 범위 내.

**현재 best 체크포인트 (v7_messidor_train)**

| 항목 | 값 |
|---|---|
| 체크포인트 | `artifacts/runs/02_domain_generalization/v7_messidor_train/checkpoints/best.pt` |
| val AUROC | 0.9983 |
| DDR AUROC | **0.8725** |
| DDR Sensitivity@0.5 | 0.5914 |
| DDR Specificity@0.5 | 0.9603 |
| DDR Sensitivity@0.09 | 0.7626 |
| DDR Specificity@0.09 | 0.8417 |

---

### [SPRINT 2] — v8_mixstyle (MixStyle 3-도메인 재시도) — 폐기

#### 배경

v7_messidor_train 이후 DDR external_test에서 threshold 편이(optimal=0.09)와 sensitivity@0.5=0.591 문제가 지속. 로드맵(ai_domain_adaptation_roadmap.md)에 따라 MixStyle을 3-도메인(APTOS+IDRiD+Messidor) 구성에서 재시도. v5 실패 원인이 "2-도메인 배치 다양성 부족"이었으므로, 3-도메인으로 재시도 시 cross-domain 통계 교환 효과를 기대.

#### 변경 내용

- `configs/v8_mixstyle.yaml` 신규: `use_mixstyle: true`, focal α=0.75 γ=0.0, v7 체크포인트에서 fine-tune

#### DDR external_test 결과

| 지표 | v7_messidor_train | v8_mixstyle | 변화 |
|---|---|---|---|
| DDR AUROC | **0.8725** | 0.8371 | **-0.035 (회귀)** |
| DDR Accuracy@0.5 | 0.765 | 0.765 | 동일 |
| DDR Sensitivity@0.5 | 0.596 | 0.596 | 동일 |
| DDR Specificity@0.5 | 0.960 | 0.934 | -0.026 |
| DDR optimal threshold | 0.09 | **0.31** | 편이 감소 |
| DDR Sensitivity@optimal | 0.763 | 0.664 | **-0.099 (악화)** |
| DDR Specificity@optimal | 0.842 | 0.880 | +0.038 |

#### 분석

optimal threshold는 0.09 → 0.31로 개선됐으나, AUROC가 -0.035 회귀하고 sensitivity@optimal이 0.099 하락. MixStyle이 배치 내 도메인 통계를 혼합할 때 소형 병변(microaneurysm 등)의 특징 패턴도 함께 교란하는 것으로 추정. 3-도메인이어도 병변 신호 자체가 스타일보다 강도가 약아 혼합에 취약한 구조.

**결론: v8_mixstyle 폐기. v7_messidor_train이 global best 유지. MixStyle 계열 기법 재시도 금지.**

다음 방향: 병변 신호를 보존하면서 도메인 스타일만 교환하는 순수 Domain Generalization 기법 탐색 → `.omc/plans/dg_research_plan.md` 참조.

---

### [SPRINT 2] — DG 전략 수립 및 전처리 방향 결정

#### 배경

v8_mixstyle 폐기 이후 도메인 일반화 접근 방향을 재검토. 기존 로드맵에 포함되어 있던 DANN(v9_dann) 전략이 DDR을 학습에 사용한다는 점에서 DDR 격리 원칙을 위반함을 확인. 동시에 전처리 파이프라인 고도화 방향도 함께 논의.

#### 주요 결정

**(1) DANN 및 모든 DA 계열 전략 기각**

DANN은 타깃 도메인(DDR) 이미지를 domain loss 계산에 사용 → DDR이 외부 테스트 격리 데이터가 아니게 됨 → 평가 무결성 파괴. 이후 모든 실험은 소스 도메인(APTOS + IDRiD + Messidor)만 사용하는 **Domain Generalization(DG)** 전략으로 제한.

**(2) DG 연구 계획서 작성**

`deep-research-report.md`(DA 논문 13편 조사) 및 capstone 논문 4편, 웹 검색(DRGen/GDRNet/DECO/SWAD/FDA/CORAL)을 종합하여 DG 연구 계획서 작성. 저장 위치: `.omc/plans/dg_research_plan.md`.

실험 우선순위:

| 순위 | 실험 | 기법 |
|------|------|------|
| 1 | v9_fda | 소스 도메인 간 저주파 Fourier 진폭 교환 |
| 2 | v10_swad | 밀집 가중치 평균 (flat minima) |
| 3 | v11_fda+swad | 복합 |
| 4 | v12_ibn | 얕은 블록 IN 교체 |

**(3) FDA 타당성 검토 — v9_fda 확정**

이진 분류(normal vs. abnormal) 태스크 특성상 병변 색상보다 병변 존재 여부(형태·구조)가 핵심 변별 단서임을 확인. FDA의 저주파 진폭 교환은 글로벌 색조(도메인 스타일)를 교환하되 고주파 병변 신호를 보존하며, Ben Graham 정규화가 이미 글로벌 색상 편차를 줄이고 있어 잔존 위험이 낮다고 판단. alpha=0.05 표준값으로 1차 실험 진행 확정.

**(4) 전처리 고도화 방향 결정**

전처리를 규칙 기반 + 품질 보정 구조로 고도화하는 방향을 검토. SwinIR / Real-HAT / SUPIR 세 모델을 분석한 결과:

- Real-HAT: GAN 기반 → 제외
- SUPIR: Diffusion 기반, 병변 hallucination 위험 → 제외
- SwinIR classical (non-GAN): L1 회귀 기반이나 자연 이미지 prior로 인해 소형 병변(microaneurysm 등) 과평활화(over-smoothing) 위험 존재 → 검증 없이 바로 도입 불가

**결론**: 학습 기반 복원 모델은 현 단계에서 배제. 규칙 기반(CLAHE, 조명 균일화, 아티팩트 마스킹) 위주로 전처리 고도화 진행. DG 실험(FDA)과 독립적으로 진행 가능.

---

### retfound — RETFound ViT-Large backbone 교체 실험

**배경**

Sprint 1 종료 시점의 주요 미결 과제는 Messidor external sensitivity 0.705로 임상 기준(≥0.80) 미달이었다. 근본 원인은 학습 도메인(APTOS, IDRiD)과 Messidor 간의 도메인 시프트이며, ImageNet pretrained EfficientNet-B5로는 한계가 있다는 판단 하에 RETFound ViT-Large 백본 교체를 시도했다.

RETFound(Zhou et al., Nature Medicine 2023)는 안저 이미지 90만 장으로 MAE 사전학습된 retinal foundation model(ViT-Large/16 기반)로, EfficientNet-B5 대비 retinal domain representation이 풍부하다는 근거가 있었다.

**내용**

**(1) 신규 파일**

| 파일 | 역할 |
|---|---|
| `drscreen/xai/gradcam_vit.py` | Grad-CAM++ for ViT. 마지막 transformer block 출력 [B, N+1, D]에서 CLS 토큰 제거 후 patch 시퀀스를 [B, D, 14, 14] 공간 그리드로 reshape하여 Grad-CAM++ 공식 적용. |
| `configs/retfound.yaml` | RETFound 전용 설정. `base.yaml` 위에 오버라이드 방식으로 동작. |

**(2) 기존 파일 수정**

| 파일 | 변경 내용 |
|---|---|
| `models/profiles.py` | `retfound` 프로파일 추가. `get_weights_enum()` 호출 이전에 early return으로 삽입. |
| `models/build.py` | `retfound` 빌드 분기 추가(`timm.create_model('vit_large_patch16_224')`). `load_retfound_backbone()` 함수 추가 — MAE 체크포인트에서 decoder 키 필터링 후 encoder만 로딩. `get_classifier_module()`에 `retfound` → `model.head` 케이스 추가. |
| `train/runner.py` | 백본 로딩 시 `architecture == "retfound"`이면 `load_retfound_backbone()` 경로로 분기. 기존 SSL 로딩 경로(else)는 유지. |
| `infer/service.py` | XAI 호출 시 `architecture == "retfound"`이면 `generate_gradcam_plus_vit()`, 그 외에는 기존 `generate_gradcam()` 유지. |

**RETFound 가중치**: `artifacts/retfound/RETFound_mae_natureCFP.pth` (Color Fundus Photography 버전, GitHub: `rmaphoh/RETFound_MAE`)

**체크포인트 로딩 검증 (smoke test)**

```
model built: VisionTransformer
head: Linear(in_features=1024, out_features=1, bias=True)
missing keys: 2  (head.weight, head.bias — 정상, classifier는 scratch 학습)
unexpected keys: 0
output shape: torch.Size([1, 1])
```

**실험 결과**

두 가지 loss 설정으로 실험:

| 실험 | Loss | val AUROC | Messidor AUROC | Sensitivity@0.5 | Specificity@0.5 |
|---|---|---|---|---|---|
| retfound_v1 | BCE | 0.9979 | 0.6722 | 0.662 | 0.577 |
| retfound_v2 | Focal (α=0.75, γ=0.0) | 0.9973 | 0.6611 | 0.558 | 0.700 |
| **v6_alpha_only (기존 best)** | **Focal (α=0.75)** | **0.9893** | **0.8697** | **0.705** | **0.910** |

**결론: RETFound 단독 fine-tuning은 기존 파이프라인 대비 Messidor AUROC -0.197 열세. global best는 v6_alpha_only 유지.**

**원인 분석**

- **Messidor 도메인 노출 부재**: v6_alpha_only는 SSL 단계(v0.6.0)에서 Messidor 1,200장을 비레이블로 인코더에 인코딩한 v4.1에서 fine-tuning했다. RETFound는 안저 이미지 90만 장으로 사전학습됐으나 이 특정 Messidor 도메인 분포는 학습한 적 없음.
- **소규모 데이터에서 ViT 불리**: ~4K 레이블 데이터에서 ViT-Large(307M)는 CNN(23M) 대비 귀납 편향이 약해 도메인 일반화가 오히려 불리하게 작용.
- **focal loss 역효과**: retfound_v2에서 focal loss가 sensitivity를 오히려 하락시킴(0.662 → 0.558). EfficientNet-B5에서 효과적이었던 alpha 가중치가 ViT-Large에서는 다른 양상을 보임.

**다음 개발 방향**

RETFound 인코더에 Messidor 도메인을 노출시킨 뒤 fine-tuning하는 방식으로 개선 시도. 구체적으로 SSL(SimCLR) 사전학습 단계에서 APTOS + IDRiD + Messidor 이미지를 레이블 없이 RETFound 인코더에 추가 적응시킨 후 지도학습 fine-tuning 진행.

---

### retfound_simclr — RETFound SimCLR SSL + LLRD + Focal 실험 및 폐기

**배경**

v0.8.0의 결론대로 RETFound 인코더에 Messidor 도메인을 사전 노출시키는 SimCLR SSL을 구현하고, 추가로 LLRD(Layer-wise LR Decay)와 Focal loss α=0.75 재시도를 적용했다.

**실험 목록 및 결과**

| 실험 | SSL | LLRD | Loss | val AUROC | Messidor AUROC |
|---|---|---|---|---|---|
| retfound_simclr_ft | SimCLR 32ep | - | BCE | 0.9976 | 0.715 |
| retfound_simclr_ft_llrd_focal | SimCLR 32ep | decay=0.75 | Focal α=0.75 γ=0 | 0.9972 | 0.728 |
| **v6_alpha_only (기존 best)** | **SimCLR (EfficientNet)** | **-** | **Focal α=0.75** | **0.9893** | **0.870** |

**SimCLR SSL 세부 설정**

- 전체 manifest 5,378장 (APTOS + IDRiD + Messidor, split 무관) 비레이블 사용
- `frozen_blocks=16` (ViT-Large 24블록 중 하위 16개 동결 → 학습 파라미터 50M/307M)
- `forward_pair()`: 두 뷰를 concat 후 단일 ViT forward pass → 연산량 절반
- Cosine LR 스케줄러 + early stopping(patience=5, min_delta=0.005)
- NTXentLoss 내부 `.float()` 강제 캐스트 (FP16에서 exp(1/0.07)≈1.6M > FP16 최대 65504 오버플로 방지)
- SSL 32 epoch(early stop) → loss 2.69 → 0.087

**LLRD 구현**

ViT-Large 24개 블록 각각에 `backbone_lr × 0.75^depth` 적용 (top block depth=0, bottom depth=23). `runner.py`의 `_build_optimizer()`에 `llrd_decay` config 키 유무로 분기.

**근본 원인 분석**

| 비교 요소 | EfficientNet v6_alpha_only | RETFound |
|---|---|---|
| 입력 해상도 | 448×448 | 224×224 (4배 낮음) |
| 파라미터 수 | 23M | 307M |
| Attention 모듈 | ECA + Spatial | 없음 |
| 누적 fine-tuning | ImageNet → SSL → v4.1 → v6 (3세대) | SSL → 1회 |
| 사전학습 backbone LR | 4e-5 (수렴 상태 반영) | 1e-5 |
| warmup | 2 epoch | 없음 |

SSL로 Messidor를 노출시켰음에도 val AUROC 0.997 vs Messidor AUROC 0.728로 격차 0.27이 지속됐다. 원인:
1. **해상도 열세**: 224px에서 미세 병변(점상출혈, 경성 삼출물) 감지 불리
2. **누적 계보 부재**: EfficientNet은 5세대 fine-tuning으로 점진적 도메인 적응이 축적됨
3. **소규모 데이터 ViT 불리**: 4K 레이블에서 307M 모델은 귀납 편향이 약해 supervised 과적합이 심함
4. **warmup 없음**: finetune 시작 시 큰 lr로 초기 불안정

도메인 적응 일반 기법(TTA, MixStyle, Dropout, FDA, Label Smoothing, DropPath, Pseudo-labeling, Mean Teacher)도 검토했으나:
- TTA, MixStyle, Dropout은 EfficientNet에서 이미 시도 후 폐기 (배치 도메인 다양성 부족이 근본 원인으로 아키텍처 무관)
- 나머지(FDA, Label Smoothing 등)는 AUROC 0.73 수준에서 근본 개선을 기대하기 어렵다고 판단

**결론: RETFound 방향 폐기. EfficientNet-B5 v6_alpha_only(AUROC 0.870)가 global best 유지.**

**코드 정리 (2026-04-10)**

RETFound 관련 코드 전체를 `archive/retfound/`로 이동하고 공유 파일에서 제거:

- 이동: `configs/retfound*.yaml`, `drscreen/cli/ssl_pretrain.py`, `drscreen/ssl/` 전체, `drscreen/xai/gradcam_vit.py`, `smoke_test_retfound.py`, artifact JSON 파일들
- 삭제: `artifacts/RETFound/RETFound_mae_natureCFP.pth`, 모든 `.pt`/`.pth` RETFound 가중치
- `models/build.py`: `retfound` 빌드 분기, `load_retfound_backbone()`, `get_classifier_module()` retfound 케이스 제거
- `models/profiles.py`: `retfound` 프로파일 제거
- `train/runner.py`: `_build_llrd_parameter_groups_retfound()`, LLRD 분기, retfound MAE 로딩 분기 제거
- `infer/service.py`: `gradcam_vit` import 및 retfound GradCAM++ 분기 제거

---

### [SPRINT 2] — v4/v4.1 계보 정정 및 SSL 오염 검증 (2026.04.11)

#### v4/v4.1 체크포인트 동일성 확인

DEVLOG v0.6.0에서 "v4 = SSL(Messidor 포함) fine-tune, val AUROC 0.9941"로 기록했으나, 실제 체크포인트 파일을 분석한 결과 사실과 다름이 확인됐다.

| 항목 | DEVLOG 기록 | 실제 체크포인트 |
|---|---|---|
| `v4/best.pt` best_val_auroc | 0.9941 | **0.9975** |
| `v4/best.pt` best_epoch | 9 | **16** |
| `artifacts/runs/01_ssl_lineage/v4.1/checkpoints/best.pt` best_val_auroc | — | **0.9975** (v4와 동일) |
| `artifacts/runs/01_ssl_lineage/v4.1/checkpoints/best.pt` best_checkpoint_path | — | `checkpoints/v4/best.pt` |

`v4/best.pt`와 `artifacts/runs/01_ssl_lineage/v4.1/checkpoints/best.pt`는 동일한 학습 실행의 결과물이다. `artifacts/runs/01_ssl_lineage/v4.1/checkpoints/best.pt`는 `v4/best.pt`를 복사해 버전 레이블만 변경한 것이며, 두 파일의 training_summary가 완전히 일치한다.

**결론**: DEVLOG의 "v4 SSL 실험(Messidor 포함, val 0.9941)" 체크포인트는 **소실**됐다. 현존하는 `v4/`, `v4.1/` 체크포인트는 동일 학습 실행의 결과물이며 val AUROC 0.9975, epoch 16이다.

**[재정정 — 2026.04.15]** 위의 "SSL 없음" 결론은 오류였다. 체크포인트 메타데이터를 재검증한 결과:

```
artifacts/runs/01_ssl_lineage/v4.1/checkpoints/best.pt → config.train.pretrained_backbone_path: artifacts/ssl/backbone_best.pt
```

v4.1은 SSL backbone(`ssl/backbone_best.pt`)에서 시작한 supervised fine-tune이다. Sprint 2 당시 `best_val_auroc` / `best_epoch` 불일치만 확인하고 `backbone_path`를 누락 확인한 오류였다. 원래 DEVLOG v0.7.0의 "SSL 단계에서 비레이블 이미지가 백본에 인코딩된 상태를 v4.1 경유로 상속"이 사실이다.

**확정된 v6_alpha_only 계보**

```
ImageNet pretrained EfficientNet-B5
  → SSL (APTOS+IDRiD+Messidor, 5378장) → ssl/backbone_best.pt
      → supervised fine-tune → v4.1  [val 0.9975]
          → focal fine-tune (α=0.75, γ=0.0) → v6_alpha_only  [Messidor 0.8697]
```

---

#### v4b_alpha_only — SSL 오염 가능성 검증 실험 (2026.04.11~)

**배경**

Sprint 1에서 SSL(v0.6.0)이 Messidor AUROC를 0.549 → 0.761로 끌어올렸는데, 이 SSL 학습에 Messidor 1200장이 레이블 없이 포함됐다. SPRINT 1 devlog의 v4b ablation 결과:

| 실험 | SSL 데이터 | Messidor AUROC | vs v3 |
|---|---|---|---|
| v3 (baseline) | 없음 | 0.549 | — |
| v4b | APTOS+IDRiD만 (4178장) | 0.679 | +0.130 |
| v4 (소실) | APTOS+IDRiD+Messidor (5378장) | 0.761 | +0.212 |

v4b fine-tune 체크포인트(`artifacts/runs/01_ssl_lineage/v4b/checkpoints/best.pt`, val AUROC 0.9892)는 현존한다. 이를 v6_alpha_only와 동일한 focal 설정으로 재학습하면 "Messidor-free SSL + focal α=0.75" 성능을 측정할 수 있다.

**timm 버전 차이로 인한 key remapping**

v4b 학습 당시 timm과 현재 timm 사이에 EfficientNet 내부 모듈 구조가 변경됐다. `strict=False`로 로딩 시 missing=804, unexpected=957이 발생해 backbone이 실질적으로 로딩되지 않는 문제 확인.

변경 패턴 3가지:
1. `blocks.X.Y.block.*` → `blocks.X.Y.*` (block wrapper 제거)
2. `blocks.X.Y.se.conv.*` → `blocks.X.Y.se.eca.conv.*` (ECA가 se 내부로 통합)
3. `blocks.X.Y.spatial_attn.*` → `blocks.X.Y.se.spatial.*` (spatial attention이 se 내부로 통합)

remapping 적용 후: missing=0, unexpected=0, matching=971. `artifacts/runs/01_ssl_lineage/v4b/checkpoints/best_remapped.pt`로 저장.

**실험 설정**

- Config: `configs/v4b_alpha_only.yaml`
- Backbone: `artifacts/runs/01_ssl_lineage/v4b/checkpoints/best_remapped.pt`
- Loss: focal α=0.75, γ=0.0 (v6_alpha_only와 동일)
- MixStyle: false
- head_epochs: 0, finetune_epochs: 15, backbone_lr: 4e-5
- Early stopping: patience=5, min_delta=0.001

**실제 결과 (Messidor external_test, n=1,200)**

| 모델 | SSL 데이터 | val AUROC | Messidor AUROC |
|---|---|---|---|
| v6_alpha_only | APTOS+IDRiD+Messidor (5,378장) | 0.9990 | **0.8697** |
| v4b_alpha_only | APTOS+IDRiD만 (4,178장) | 0.9973 | **0.7262** |

**결론**: Messidor를 SSL에서 제외하면 AUROC가 0.8697 → 0.7262로 하락(-0.1435). SSL 오염 효과는 실재하나, v6_alpha_only의 0.8697도 순수 일반화 능력이 아닌 Messidor 도메인 노출의 이점을 포함한다. 외부 일반화 개선이 Sprint 3의 핵심 과제로 확정.

- 평가 결과: `artifacts/runs/01_ssl_lineage/v4b_alpha_only/evaluations/external_test_v4b_alpha_only_best_metrics.json`
- 체크포인트: `artifacts/runs/01_ssl_lineage/v4b_alpha_only/checkpoints/best.pt` (best_epoch=6)

---

### [SPRINT 2] — v9_fda: Fourier Domain Adaptation 도메인 일반화 (2026.04.15)

#### 동기

v7_messidor_train(DDR AUROC 0.8725)의 핵심 문제는 threshold bias였다. Optimal threshold가 0.09로, 실제 운용 시 양성 기준이 극단적으로 낮아 특이도 희생 없이는 민감도를 확보하기 어렵다. 근본 원인은 훈련 도메인(APTOS, IDRiD, Messidor)과 테스트 도메인(DDR) 간 색조·조명 분포 차이에 있다. 모델이 DDR 스타일의 이미지에서 양성 확률을 낮게 예측하는 방향으로 편향된 것이다.

FDA(Fourier Domain Adaptation)는 소스 이미지의 저주파 진폭 성분을 참조 이미지(다른 도메인)의 것으로 교체해 스타일 변이를 데이터 레벨에서 무력화한다. 추가 파라미터 없이 on-the-fly 적용이 가능하며, AUROC가 아닌 threshold 보정 효과도 기대할 수 있다.

#### 구현

**`drscreen/data/transforms.py` — `fda_mix(source, reference, alpha)`**

FFT 기반 저주파 진폭 교환 함수. `alpha`는 교환 반경을 `min(H, W)`에 대한 비율로 정의한다. 채널별 독립 FFT 후 중심 기준 `h_cut × w_cut` 영역의 진폭을 참조 이미지 것으로 교체하고 역변환한다. 참조 이미지 해상도가 다를 경우 `cv2.INTER_LINEAR`로 리사이즈 후 적용한다.

**`drscreen/data/datasets.py` — `FDAManifestDataset`**

`ManifestDataset`을 상속하며, `__getitem__`에서 소스 이미지와 크로스도메인 참조 이미지를 함께 로딩해 `fda_mix()` 적용 후 transform 파이프라인에 전달한다. 도메인별 인덱스(`_domain_indices`)를 미리 구성해 O(1) 크로스도메인 샘플링을 지원한다. `rebuild_domain_indices()`는 runner.py에서 도메인 제외 필터링 이후 호출해 인덱스 일관성을 보장한다.

**`drscreen/train/runner.py` — `_build_datasets()` 수정**

`data.use_fda: true`일 때 `FDAManifestDataset`을 인스턴스화하고, 도메인 필터링 직후 `rebuild_domain_indices()`를 호출한다.

#### 실험 설정 (`configs/v9_fda.yaml`)

| 항목 | 값 |
|---|---|
| Backbone | artifacts/runs/02_domain_generalization/v7_messidor_train/checkpoints/best.pt (DDR AUROC 0.8725 기반) |
| FDA alpha | 0.05 (512px 기준 교환 반경 ~25px) |
| train_exclude_domains | [] (Messidor 포함) |
| Loss | focal α=0.75, γ=0.0 |
| head_epochs / finetune_epochs | 0 / 15 |
| backbone_lr | 4e-5 |
| MixStyle | false |

#### 결과 (DDR external_test, n=12,522)

| 지표 | v7_messidor_train | v9_fda | 변화 |
|---|---|---|---|
| AUROC | 0.8725 | **0.8812** | +0.0087 |
| Optimal threshold | 0.09 | **0.19** | +0.10 |
| Sensitivity (optimal) | 0.7626 | 0.7353 | -0.0273 |
| Specificity (optimal) | 0.8417 | **0.8894** | +0.0477 |
| F1 (optimal) | 0.7939 | **0.7966** | +0.0027 |
| Accuracy (optimal) | 0.8022 | **0.8124** | +0.0102 |

#### 분석

AUROC가 0.8725 → 0.8812로 향상됐으며, threshold bias가 0.09 → 0.19로 대폭 개선됐다. 0.09는 운용 임계값으로 위험한 수준이었으나, 0.19는 실용적인 범위다. Sensitivity가 0.0273p 하락했으나 이는 threshold가 달라진 조건에서의 비교로, 판별 능력 자체를 나타내는 AUROC 기준으로는 v9_fda가 명확히 우위다.

FDA가 저주파 진폭(전역 색조·조명) 편향을 훈련 시점에 제거해, 모델이 도메인 불변 고주파 특징(혈관 구조, 삼출물, 출혈)을 학습하도록 유도한 효과로 해석된다.

**결론: v9_fda를 legacy/global best로 승격. `artifacts/checkpoints/best.pt` ← `artifacts/runs/02_domain_generalization/v9_fda/checkpoints/best.pt`.**

---

### [SPRINT 2] — v10_swad: Stochastic Weight Averaging Dense (2026.04.15)

#### 동기

v9_fda에서 threshold bias(0.09 → 0.19) 개선을 확인했으나, 더 강한 도메인 일반화를 위한 독립 실험이 필요했다. SWAD(Cha et al., NeurIPS 2021)는 학습 후반부 매 스텝의 가중치를 밀집 평균하여 flat loss minima를 탐색하며, DRGen(MICCAI 2022)에서 DR DG에 유효성이 확인됐다. FDA와 독립적으로 적용해 각 기법의 기여를 분리하기 위해 v7_messidor_train을 backbone으로 사용했다.

#### 구현

**`drscreen/train/engine.py` — `SWADBuffer`**

`deque(maxlen=n)` 기반 rolling buffer. `update(model)`은 현재 가중치를 CPU에 복사해 적재하고, `get_averaged_state_dict()`는 float 텐서를 평균하되 `num_batches_tracked` 등 정수 버퍼는 최신 스냅샷 값을 유지한다.

**`drscreen/train/runner.py` — SWAD 통합**

- 학습 전: `swad_last_n_epochs` config로 `SWADBuffer` 초기화 (0이면 비활성)
- 매 finetune epoch 종료 후, 조기종료 직전: `swad_buffer.update(model)` — 조기종료 epoch 포함
- 모든 학습 종료 후: 버퍼 평균 → BN 통계 갱신(train mode forward pass) → val 평가 → 기존 best보다 우수하면 `best.pt` 교체

`start_epoch` 절대값 대신 `swad_last_n_epochs`(마지막 N epoch 평균)를 사용해 조기종료 시에도 항상 작동하도록 설계했다.

#### 실험 설정 (`configs/v10_swad.yaml`)

| 항목 | 값 |
|---|---|
| Backbone | artifacts/runs/02_domain_generalization/v7_messidor_train/checkpoints/best.pt |
| FDA | 비활성 (SWAD 단독 ablation) |
| swad_last_n_epochs | 5 |
| Loss | focal α=0.75, γ=0.0 |
| finetune_epochs | 15 |
| backbone_lr | 4e-5 |

#### 결과 (DDR external_test, n=12,522)

> **[BN 재보정 버그 수정 후 재평가 — 2026.04.21]**
> SWAD BN recalibration 단계에서 `model.train()` 호출이 stochastic depth/dropout까지 활성화하는 버그가 발견되어 수정(`model.eval()` + BN-only train()) 후 재학습·재평가됨. 아래 수치는 수정 후 결과.

| 지표 | v7_messidor_train | v9_fda | v10_swad | v7 대비 |
|---|---|---|---|---|
| AUROC | 0.8725 | 0.8812 | **0.8863** | +0.0138 |
| Optimal threshold | 0.09 | **0.19** | 0.05 | -0.04 |
| Sensitivity (optimal) | 0.7626 | 0.7353 | 0.7212 | -0.0414 |
| Specificity (optimal) | 0.8417 | **0.8894** | 0.9033 | +0.0616 |
| F1 (optimal) | 0.7939 | 0.7966 | 0.7934 | -0.0005 |

#### 분석

BN 버그 수정 후 SWAD의 AUROC 향상폭이 +0.02 → +0.0138로 줄었고, threshold bias는 0.08 → 0.05로 오히려 더 악화됐다. 원래 버그 상태에서는 BN 재보정 중 stochastic depth가 의도치 않은 정규화 효과를 제공했던 것으로 해석된다.

- **FDA**: 저주파 스타일 교환으로 도메인 갭을 augmentation 레벨에서 직접 제거 → threshold bias 해소
- **SWAD**: flat minima 탐색으로 전반적 판별력 향상 → AUROC 향상, threshold bias는 미개선

두 기법의 역할 분리는 여전히 유효하나, SWAD 단독으로는 v9_fda 대비 threshold 열위(0.19 vs 0.05)가 더 커졌다.

**결론: v10_swad는 BN 수정 후에도 AUROC(0.8863)는 v9_fda(0.8812) 대비 우위이나 threshold bias가 0.05로 악화되어 배포 부적합 판정이 강화됐다. v9_fda 배포 지정 유지.**

---

### [SPRINT 2] — v11_fda_swad: FDA + SWAD 복합 실험 (2026.04.15)

#### 동기

v9_fda(threshold 개선, AUROC +0.009)와 v10_swad(AUROC 대폭 향상, threshold 악화)가 상보적으로 보여 복합 적용 시 두 효과가 중첩될 것으로 예측했다.

#### 실험 설정 (`configs/v11_fda_swad.yaml`)

| 항목 | 값 |
|---|---|
| Backbone | artifacts/runs/02_domain_generalization/v7_messidor_train/checkpoints/best.pt |
| FDA alpha | 0.05 |
| swad_last_n_epochs | 5 |
| Loss | focal α=0.75, γ=0.0 |
| finetune_epochs | 15 |

#### 결과 (DDR external_test, n=12,522)

| 지표 | v7 | v9_fda | v10_swad* | v11_fda_swad |
|---|---|---|---|---|
| AUROC | 0.8725 | 0.8812 | 0.8863 | 0.8539 |
| Optimal threshold | 0.09 | 0.19 | 0.05 | **0.31** |
| Sensitivity (optimal) | 0.7626 | 0.7353 | 0.7212 | 0.7088 |
| Specificity (optimal) | 0.8417 | 0.8894 | **0.9033** | 0.8936 |
| F1 (optimal) | 0.7939 | 0.7966 | 0.7934 | 0.7808 |

*v10_swad: BN 재보정 버그 수정 후 재평가 수치 (2026.04.21)

#### 분석

Threshold는 0.31로 역대 최고를 기록했으나 AUROC가 0.8539로 v7 baseline보다 낮아지는 심각한 회귀가 발생했다. FDA와 SWAD는 단독으로는 각각 효과적이었으나 함께 사용하면 서로 간섭한다.

두 가지 메커니즘이 의심된다.

**1. 손실 지형 충돌**: SWAD는 학습 후반부 손실 지형이 평탄하게 수렴한다는 가정 위에서 작동한다. FDA는 매 epoch마다 다른 스타일 이미지를 생성해 손실 지형에 지속적 노이즈를 주입한다. 결과적으로 마지막 5 epoch 체크포인트가 각기 다른 방향을 향하고, 평균 시 어느 방향도 아닌 흐릿한 가중치가 된다.

**2. BN 통계 오염**: SWAD 후 BN 통계 갱신을 FDA 변환된 이미지로 수행했다. 추론 시 원본 DDR 이미지가 들어오므로 running statistics 분포 불일치가 발생했을 가능성이 있다.

Threshold 0.31 개선은 FDA가 여전히 출력 분포를 교정하고 있음을 보여주지만, SWAD가 AUROC를 크게 훼손했다.

**결론: FDA + SWAD 복합은 현재 구성에서 역효과. v9_fda(AUROC 0.8812, threshold 0.19)가 global best 유지.**

---

### [SPRINT 2] — v12_fda_imagenet: FDA + ImageNet 초기화 (backbone 없이) (2026.04.15)

#### 동기

v9_fda(v7 + FDA)에서 threshold bias가 0.09 → 0.19로 개선됐으나, 이것이 FDA 자체의 힘인지 v7 backbone에 누적된 도메인 편향을 교정한 결과인지 불분명했다. ImageNet 초기화에서 FDA만으로 처음부터 수렴시켜 두 효과를 분리하고자 했다.

#### 실험 설정 (`configs/v12_fda_imagenet.yaml`)

| 항목 | 값 |
|---|---|
| Backbone | ImageNet pretrained EfficientNet-B5 (v7 없음) |
| FDA alpha | 0.05 |
| SWAD | 비활성 |
| Loss | focal α=0.75, γ=0.0 |
| head_epochs / finetune_epochs | 5 / 20 |

#### 결과 (DDR external_test, n=12,522)

| 지표 | v7 | v9_fda | v12_fda_imagenet |
|---|---|---|---|
| AUROC | 0.8725 | **0.8812** | 0.8498 |
| Optimal threshold | 0.09 | **0.19** | 0.05 |
| Sensitivity (optimal) | **0.7626** | 0.7353 | 0.6726 |
| F1 (optimal) | 0.7939 | **0.7966** | 0.7540 |

#### 분석

v7 baseline보다 AUROC와 threshold 모두 악화됐다. ImageNet에서 FDA와 함께 처음부터 학습하는 것은 역효과다.

FDA는 잘 수렴된 backbone 위에서 잔존 domain bias를 교정하는 기법이지, ImageNet 초기화에서 domain-invariant 표현을 처음부터 유도하는 기법이 아니다. FDA 증강은 훈련을 어렵게 만들어 fundus 특화 특징(혈관 구조, 병변) 자체를 충분히 학습하지 못한 채 수렴하게 만든다. Optimal threshold 0.05는 v7(0.09)보다 오히려 더 극단적으로 낮아졌다.

**v7 backbone의 5세대 누적 fine-tuning이 DG 성능의 전제조건임을 실험적으로 확인했다.**

**결론: v7 없이 FDA 단독 적용은 실패. v9_fda(AUROC 0.8812, threshold 0.19) global best 유지.**

---

### [SPRINT 2] — v13_fda_swad: FDA + SWAD BN 오염 수정 재실험 (2026.04.16~)

#### 배경

v11_fda_swad 실패 원인을 분석한 결과 두 가지 메커니즘이 의심됐다. 손실 지형 충돌(FDA의 epoch별 스타일 노이즈 vs SWAD의 평탄 수렴 가정)과 BN 통계 오염이다.

BN 오염 문제: v11에서 SWAD 후 BN 통계 갱신을 `train_loader`(FDAManifestDataset)로 수행했다. FDA 혼합 이미지는 훈련 전용 증강이며 추론 시에는 실제 이미지가 입력된다. 이 분포 불일치가 BN running statistics를 왜곡해 추론 성능을 저하시켰을 가능성이 있다.

#### 수정 사항

**`drscreen/train/runner.py`** — SWAD BN 통계 갱신 로직 변경

```
수정 전: train_loader (FDAManifestDataset, FDA 혼합 이미지)
수정 후: _build_eval_dataset(train_split) (ManifestDataset, 원본 이미지 + eval_transform)
```

추론 시 분포(원본 이미지)와 일치하는 BN 통계를 확보한다. 이 수정은 FDA 여부와 무관하게 모든 SWAD 실험에 적용된다.

#### 실험 설정 (`configs/v13_fda_swad.yaml`)

v11_fda_swad와 동일 설정. BN 갱신 방식만 변경.

| 항목 | 값 |
|---|---|
| Backbone | artifacts/runs/02_domain_generalization/v7_messidor_train/checkpoints/best.pt |
| FDA alpha | 0.05 |
| swad_last_n_epochs | 5 |
| Loss | focal α=0.75, γ=0.0 |
| finetune_epochs | 15 |
| v11 대비 변경 | BN 갱신을 FDA-free 원본 이미지로 수행 |

#### 예상 결과

- BN 오염이 v11 실패의 주요 원인이었다면 → AUROC·threshold 모두 개선
- 손실 지형 충돌이 주요 원인이었다면 → BN 수정만으로는 회복 불가

#### 결과 (DDR external_test, n=12,522)

| 지표 | v7 | v9_fda | v11_fda_swad | **v13_fda_swad** |
|---|---|---|---|---|
| AUROC | 0.8725 | **0.8812** | 0.8539 | 0.8436 |
| Optimal threshold | 0.09 | **0.19** | 0.31 | 0.05 |
| Sensitivity (optimal) | **0.7626** | 0.7353 | 0.7088 | 0.6084 |
| F1 (optimal) | 0.7939 | **0.7966** | 0.7808 | 0.7311 |

#### 분석

v11보다 더 나빠졌다. best_epoch=4로 SWAD 평가가 val AUROC 0.9978을 넘지 못해 SWAD 모델이 저장되지 않았고, 결과는 epoch 4의 일반 체크포인트다. BN 오염은 원인이 아니었다.

진짜 원인은 FDA가 매 epoch 다른 스타일 분포를 생성해 SWAD의 전제(학습 후반부 손실 지형의 평탄한 수렴)를 구조적으로 깨뜨리는 것이다. BN 통계 갱신 방식을 바꿔도 이 근본 충돌은 해소되지 않는다.

**결론: FDA + SWAD 조합은 현재 아키텍처에서 구조적으로 양립 불가. 두 기법 복합 실험 종료. v9_fda(AUROC 0.8812, threshold 0.19) global best 유지.**

---

### [SPRINT 2] — v14_ibn: IBN-Net 도메인 일반화 (2026.04.17)

#### 동기

IBN-Net(Pan et al., ECCV 2018)은 EfficientNet의 shallow block(blocks[0-2]) BN을 IBN-a(InstanceNorm 절반 + BatchNorm 절반)로 교체하여 스타일 불변 표현을 유도한다. InstanceNorm은 채널별 통계를 정규화해 도메인 특유의 색조·조명 변이를 제거하고, BatchNorm은 판별력 있는 구조 특징을 보존한다. MixStyle·FDA와 달리 학습 중 추가 연산 없이 아키텍처 수준에서 적용된다.

#### 구현

- `drscreen/models/build.py`: `IBN` 클래스 추가 (IBN-a: `InstanceNorm2d(half) + BatchNorm2d(rest)`), `_inject_ibn()` 함수 추가 (blocks[0-2] BN 교체), `build_model(use_ibn: bool)` 파라미터 추가
- `drscreen/train/runner.py`: `build_model()` 호출에 `use_ibn` 전달
- `drscreen/infer/service.py`: `InferenceSession.from_config_path()`의 `build_model()` 호출에 `use_ibn` 추가 (기존 누락 버그 수정)
- `drscreen/settings.py`: `build_effective_checkpoint_config()`에 `use_ibn` 키 propagation 추가
- `configs/base.yaml`: `model.use_ibn: false` 기본값 추가
- `configs/v14_ibn.yaml`: 신규, `use_ibn: true`, v9_fda backbone(`artifacts/runs/02_domain_generalization/v9_fda/checkpoints/best.pt`) 사용

IBN 주입 시 state_dict 키가 변경되므로 (`bn1` → `ibn`) v9_fda backbone 로딩은 `strict=False`로 처리 (runner.py 기존 로직).

#### 학습 결과

- Backbone: `artifacts/runs/02_domain_generalization/v9_fda/checkpoints/best.pt`
- best_epoch: 12, best_val_auroc: 0.9963, early stop at epoch 17 (patience=5)
- promoted_to_global_best: false

#### DDR external_test 결과 (n=12,522)

| 지표 | v9_fda (global best) | v14_ibn | 변화 |
|---|---|---|---|
| AUROC | **0.8812** | 0.8445 | -0.037 |
| Optimal threshold | **0.19** | 0.08 | -0.11 |
| Sensitivity (optimal) | **0.7353** | 0.6995 | -0.036 |
| F1 (optimal) | **0.7966** | 0.7550 | -0.042 |

#### 분석

IBN-Net이 도메인 일반화를 개선하지 못했다. DDR AUROC가 v9_fda 대비 -0.037 하락하고 threshold bias도 0.08로 악화됐다.

원인: DR 판별 단서(미세출혈, 삼출물, 혈관 이상)는 색조·조명(low-frequency) 변이가 아닌 fine-grained structural pattern에 기반한다. IBN-a의 InstanceNorm이 전역 스타일뿐 아니라 진단에 유효한 국소 texture 정보까지 함께 소거한 것으로 추정된다. Shallow block은 병변 구조 특징이 구성되기 시작하는 단계이므로 IN 적용의 부작용이 크다.

**결론: v14_ibn 폐기. v9_fda(AUROC 0.8812, threshold 0.19) global best 유지.**

- 평가 결과: `artifacts/runs/02_domain_generalization/v14_ibn/evaluations/external_test_v14_ibn_best_metrics.json`
- 체크포인트: `artifacts/runs/02_domain_generalization/v14_ibn/checkpoints/best.pt`

---

### [SPRINT 3] — 전처리 파이프라인 고정 및 v9_fda 재기준 수립 (2026.04.27)

#### 배경

Sprint 2 종료 시점의 배포 모델(v9_fda)은 DDR AUROC 0.8812, optimal threshold 0.19를 기록했다. Sprint 3에서 전처리 파이프라인을 재검토하면서 세 가지 문제를 순차적으로 발견·수정했다.

---

#### (1) `_circular_crop` 중심점 계산 개선: bounding box → centroid

**문제**: `data/transforms.py`의 `_circular_crop` 메서드가 비흑색 픽셀의 bounding box 중심(`x + w//2, y + h//2`)을 원 중심으로 사용했다. 안저 이미지는 흔히 한쪽이 더 어두운 비대칭 조명을 가지므로 bounding box 중심이 실제 안저 원반 중심과 어긋날 수 있다.

**수정**: `cv2.moments(mask)`로 이진 마스크의 centroid를 산출하고 이를 원 중심으로 사용. bounding box는 반경 계산(`max(w, h) // 2`)에만 유지.

```python
# data/transforms.py — _circular_crop
M = cv2.moments(mask)
if M["m00"] > 0:
    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
else:
    x, y, w, h = cv2.boundingRect(coords)
    cx, cy = x + w // 2, y + h // 2
x, y, w, h = cv2.boundingRect(coords)
radius = max(w, h) // 2
```

**효과**: 비대칭 조명 이미지에서 안저 원반이 정확하게 중앙에 위치하도록 개선.

---

#### (2) QuickQual 품질 필터 폐기 및 휴리스틱 필터 완전 제거

**문제 1 — QuickQual sklearn 버전 불일치**: QuickQual(DenseNet121+SVM)의 `.pkl` 가중치가 sklearn 1.2.2로 직렬화되어 있으나, 현재 환경은 sklearn 1.7.2(Python 3.14 미지원으로 다운그레이드 불가). 로드 시 90% 이상 이미지를 reject로 분류하는 오판 발생.

**문제 2 — 휴리스틱 임계값 오캘리브레이션**: `blur_score_min: 248.929`, `brightness_mean_min: 126.870`은 사전 전처리된(Ben Graham 적용) 이미지 기준으로 캘리브레이션됐다. 그러나 `preprocess_images.py`는 원본(raw) 이미지에 필터를 적용했으므로 전체 이미지가 필터에 걸리는 오동작 발생 (raw 이미지 blur=4-101, brightness=22-85 vs 임계값 248/126).

**현재 코드 기준 수정 상태**:
1. `preprocess_images.py`에서 품질 필터 로직 전체 제거. 전처리-저장 기능만 순수하게 유지.
2. `configs/base.yaml`에는 blur/brightness/QuickQual 임계값이 없다. 학습·평가·오프라인 전처리는 품질 필터 없이 전체 manifest를 대상으로 수행한다.
3. 추론 품질 판정은 AI 모듈이 아니라 backend 단의 QuickQual task로 분리됐다. `drscreen/infer/service.py`는 backend 호환을 위해 `quality_warning`, `quality`, `quality_grade`, `quality_grade_confidence`를 `None`으로만 채운다.

**결과**: `manifest_preprocessed.csv`에 17,900장 전체 포함 (이전 품질 필터 오작동으로 89~1,800장만 포함되던 문제 해소).

---

#### (3) Alignment 실험 시리즈 → use_align=false 고정

**배경**: 안저 원반의 중심을 이미지 프레임 중앙으로 이동시키는 정렬 보정(translation)이 도메인 일반화에 도움이 될지 실험.

**실험 결과**:

| 실험 | 정렬 | DDR optimal threshold | 결론 |
|---|---|---|---|
| v9_fda (재학습 기준) | 없음 | 0.06 | 베이스라인 |
| center-only alignment | translation 보정 | < 0.06 | threshold 악화 |
| full alignment (tilt+center) | 기울기 + translation | < 0.06 | threshold 더 악화 |

**원인 분석**: Alignment는 기하학적 변이를 제거하여 이미지를 균일화한다. 이 기하학적 다양성이 APTOS/IDRiD/Messidor 학습 시 DDR에 대한 암묵적 augmentation 역할을 했을 가능성이 높다. 제거 시 오히려 도메인 갭이 증가하여 threshold가 더 낮아진다.

**결정**: `use_align: false` 영구 고정 (`configs/base.yaml`). `_correct_alignment()` 메서드는 코드에 보존하되 미사용.

---

#### (4) v9_fda 재학습 결과 (새 전처리 + 17,900장 manifest)

**설정**: `configs/v9_fda.yaml` 동일, manifest만 17,900장으로 재구성된 `manifest_preprocessed.csv` 사용.

| 항목 | 값 |
|---|---|
| 체크포인트 | `artifacts/runs/02_domain_generalization/v9_fda/checkpoints/best.pt` (재학습으로 덮어씌움) |
| best_epoch | 1 (early stop at epoch 6, patience=5) |
| val AUROC | 0.9963 |
| DDR AUROC | **0.8825** |
| DDR optimal threshold | **0.06** |

**Sprint 2 v9_fda 대비 변화**

| 지표 | Sprint 2 v9_fda | Sprint 3 재학습 | 변화 |
|---|---|---|---|
| DDR AUROC | 0.8812 | **0.8825** | +0.0013 ✅ |
| DDR optimal threshold | **0.19** | 0.06 | -0.13 ⚠️ |
| DDR sensitivity@optimal | 0.7353 | 0.7498 | +0.0145 |
| DDR specificity@optimal | 0.8894 | 0.8559 | -0.0335 |
| DDR F1@optimal | 0.7966 | 0.7917 | -0.0049 |

**분석**: centroid crop이 AUROC를 소폭 개선했으나, 품질 필터 제거(더 많은 저품질 이미지 포함)와 alignment 제거의 복합 효과로 threshold가 0.19 → 0.06으로 크게 악화됐다. AUROC·threshold 균형 기준으로 Sprint 2 v9_fda(threshold 0.19)가 여전히 우위이나 그 체크포인트는 현재 덮어씌워진 상태다.

**현재 상태**: Sprint 3 재학습 결과가 `artifacts/runs/02_domain_generalization/v9_fda/checkpoints/best.pt`에 저장됨. Sprint 2 v9_fda 체크포인트 소실.

---

#### Sprint 3 방향

- **현재 병목**: threshold 0.06으로 배포 부적합. AUROC는 0.8825로 소폭 개선.
- **다음 실험 후보** (`.omc/plans/threshold_auroc_optimization.md` 참조):
  1. Ensemble (v9_fda + v10_swad 로짓 평균) — 학습 없음, 당일 실행
  2. FDA α=0.10 (현재 0.05) — threshold bias 직접 교정 레버
  3. focal_gamma=1.0 (현재 0.0) — hard example focusing으로 DDR calibration 개선 가능성
  4. Temperature Scaling — threshold=0.5 UX 개선 (AUROC 불변)

---

### [SPRINT 3] — v15_fda_a10: FDA α=0.10 실험 (2026.04.27) — 폐기

v9_fda(α=0.05)의 threshold bias 개선을 위해 FDA α를 0.10으로 확장 실험.

| 지표 | v9_fda (재학습) | v15_fda_a10 | 변화 |
|---|---|---|---|
| DDR AUROC | **0.8825** | 0.8579 | -0.025 |
| Optimal threshold | 0.06 | 0.05 | 개선 없음 |
| Sensitivity@opt | **0.7498** | 0.6391 | -0.111 |

AUROC 0.88 미달, threshold 개선 없음. α=0.10은 논문 권장 범위(0.01~0.09) 상한을 초과하여 병변 구조 특징까지 왜곡한 것으로 판단. **FDA α 확장 실험 종료. v9_fda(α=0.05)가 FDA 계열 최선.**

---

### [SPRINT 3] — v16/v17: focal_gamma 탐색 (2026.04.27)

#### 배경 및 설계 원칙

Sprint 3 재학습 이후 threshold가 0.06으로 악화된 상태에서, 모든 v7~v14 실험이 `focal_gamma=0.0`(사실상 weighted BCE)을 사용했다는 점에 주목했다. γ>0의 진정한 Focal Loss는 DDR threshold 문제의 구조적 원인에 직접 작용할 수 있다.

**Focal Loss 수식:** $FL(p_t) = -\alpha_t \cdot (1-p_t)^\gamma \cdot \log(p_t)$

`(1-p_t)^γ`가 핵심이다. γ=0이면 모든 샘플이 동등한 가중치(weighted BCE). γ>0이면 모델이 이미 잘 맞히는 쉬운 샘플의 loss를 지수적으로 감쇠시켜, 어려운 샘플에 gradient를 집중시킨다.

| γ | 쉬운 샘플 (p_t=0.95) modulating | 어려운 샘플 (p_t=0.10) modulating | 비율 |
|---|---|---|---|
| 0.0 (weighted BCE) | 1.0 | 1.0 | 1x |
| 1.0 | 0.05 | 0.90 | 18x |
| 2.0 | 0.0025 | 0.81 | **324x** |

v7 backbone 이후 학습 데이터의 "어려운 케이스"(borderline)는 외관상 DDR과 가장 유사한 샘플이다. γ=2.0으로 이 경계 샘플들을 높은 확률로 분류하도록 강제하면, 추론 시 유사하게 생긴 DDR 이상 이미지도 더 높은 확률을 받아 출력 분포가 0.5 방향으로 이동한다.

#### 실험 설정 공통 사항

v9_fda 대비 `focal_gamma`만 변경. 나머지 동일: FDA α=0.05, v7 backbone, focal α=0.75, finetune_epochs=15, backbone_lr=4e-5.

#### 결과 (DDR external_test, n=12,522)

| 모델 | AUROC | Optimal thr | Sensitivity@thr | Specificity@thr | F1@thr | Sensitivity@0.5 |
|---|---|---|---|---|---|---|
| v9_fda (재학습, γ=0.0) | 0.8825 | 0.06 | 0.7498 | 0.8559 | 0.7917 | 0.578 |
| v16_focal_g1 (γ=1.0) | 0.8738 | 0.18 | 0.7355 | 0.8559 | 0.7825 | — |
| **v17_focal_g2 (γ=2.0)** | **0.8911** | **0.42** | **0.7727** | **0.8564** | **0.8063** | **0.717** |

#### 분석

γ=2.0(v17)이 모든 기준에서 최고를 달성했다.

- **AUROC 0.8911**: 단일 모델 신기록 (앙상블 v9+v10의 0.8920에 근접)
- **Threshold 0.42**: 사실상 0.5 근처에서 배포 운용 가능. threshold=0.5에서 sensitivity 0.717, specificity 0.907로 즉각 배포 가능한 수준
- γ=1.0은 threshold를 0.06→0.18로 개선했으나 AUROC가 소폭 하락. γ=2.0은 AUROC까지 함께 개선 — hard example focusing이 DDR의 어려운 경계 케이스 판별력을 직접 강화했기 때문

#### Sprint 3 수락 기준 충족

- ✅ DDR AUROC ≥ 0.88 (0.8911)
- ✅ Optimal threshold ≥ 0.15 (0.42)

**v17_focal_g2를 Sprint 3 현재 best로 확정. γ=3.0 추가 실험 진행.**

---

### [SPRINT 3] — v18_focal_g3: γ=3.0 추가 실험 (2026.04.27) — 폐기

v17_focal_g2(γ=2.0)에서 threshold 0.42를 달성한 후 γ=3.0으로 추가 개선 가능성 탐색.

| γ | AUROC | Optimal threshold | Sensitivity@0.5 | F1@opt |
|---|---|---|---|---|
| 2.0 (v17) | **0.8911** | **0.42** | **0.717** | **0.8063** |
| 3.0 (v18) | 0.8747 | 0.29 | 0.571 | 0.7800 |

γ=3.0은 AUROC와 threshold 모두 후퇴. γ가 너무 크면 borderline 샘플(p_t=0.4~0.6)의 modulating factor가 (0.5)^3=0.125로 과도하게 억제되어, 극단적으로 어려운 소수 샘플에만 gradient가 몰려 일반화 능력이 저하된다. **γ=2.0이 최적점.**

**결론: focal γ 탐색 종료. v17_focal_g2(γ=2.0, AUROC 0.8911, threshold 0.42)가 Sprint 3 최종 배포 best로 확정.**

| 항목 | 값 |
|---|---|
| 체크포인트 | `artifacts/runs/02_domain_generalization/v17_focal_g2/checkpoints/best.pt` |
| best_epoch | 3 |
| val AUROC | 0.9970 |
| DDR AUROC | **0.8911** |
| DDR optimal threshold | **0.42** |
| DDR Sensitivity@0.42 | 0.7727 |
| DDR Specificity@0.42 | 0.8564 |
| DDR Sensitivity@0.5 | 0.717 |
| DDR Specificity@0.5 | 0.907 |

---

### [SPRINT 3] — v19_swad_focal_g2: SWAD + focal γ=2.0 조합 (2026.04.28) — 폐기

#### 배경 및 설계

v10_swad(SWAD, γ=0.0, threshold 0.05)와 v17_focal_g2(γ=2.0, threshold 0.42)의 조합이 한 번도 시도된 적 없다는 점에서 출발. focal γ=2.0의 threshold 교정 효과를 유지하면서 SWAD의 flat minima 일반화를 추가 적용. FDA 없이 순수 SWAD+focal 조합으로 v11/v13 FDA+SWAD 충돌 원인을 제거. backbone: `artifacts/runs/02_domain_generalization/v7_messidor_train/checkpoints/best.pt`, swad_last_n_epochs=5.

#### 결과

| 지표 | v17_focal_g2 (배포) | v19_swad_focal_g2 |
|---|---|---|
| DDR AUROC | **0.8911** | 0.8833 |
| optimal threshold | **0.42** | 0.06 |
| sensitivity@0.5 | **0.717** | 0.502 |
| sensitivity@optimal | ~0.773 | 0.772 |

- 평가 결과: `artifacts/runs/02_domain_generalization/v19_swad_focal_g2/evaluations/external_test_v19_swad_focal_g2_best_metrics.json`

#### 분석

SWAD가 focal γ=2.0의 threshold 교정 효과를 희석했다. focal γ=2.0의 threshold 보정은 학습 후반 수렴 단계에 집중되는데, SWAD가 마지막 5 에폭을 평균 내면서 수렴 이전의 덜 교정된 체크포인트들을 혼합해 threshold가 0.42 → 0.06으로 역행했다. v10_swad의 threshold(0.05)와 거의 동일한 수준으로 복귀. AUROC도 v17 대비 회귀(-0.0078). v11/v13과 원인은 다르지만 결과적으로 SWAD는 focal γ 기반 calibration과 충돌한다.

**결론: SWAD + focal γ=2.0 충돌 확인. v17_focal_g2가 여전히 Sprint 3 best. 폐기.**

---

### [SPRINT 3] — v20_coral: Deep CORAL + FDA + focal γ=2.0 (2026.04.29) — 폐기

#### 배경 및 설계

v17_focal_g2(γ=2.0, AUROC 0.8911)가 Sprint 3 best로 확정된 이후, 학습 도메인(APTOS, IDRiD, Messidor) 간 feature 공분산 정렬이 외부 도메인(DDR) 일반화를 추가로 개선할 수 있다는 가설 하에 Deep CORAL을 시도했다. FDA(α=0.05)와 focal(γ=2.0)을 유지하면서 CORAL 손실(λ=1.0)을 추가하는 방식이다.

**CORAL 손실 수식:**

$$\mathcal{L}_{\text{CORAL}} = \frac{1}{4d^2} \left\| C_S - C_T \right\|_F^2$$

여기서 $C_S$, $C_T$는 소스·타깃 도메인의 feature 공분산 행렬이고 $d$는 feature 차원(EfficientNet-B5 pooled head 출력).

**구현 세부:**
- `drscreen/train/loss.py`: `CoralLoss` 클래스 추가
- `drscreen/train/engine.py`: `_has_timm_feature_api()`, `_forward_with_features()`, `_compute_coral_loss()` 헬퍼 추가, `train_one_epoch()`에 `coral_criterion`/`lambda_coral` 파라미터 추가
- `drscreen/data/datasets.py`: `ManifestDataset`·`FDAManifestDataset`의 `__getitem__()` 반환에 `domain` 키 추가
- `configs/v20_coral.yaml`: FDA α=0.05, focal γ=2.0, CORAL λ=1.0, backbone: artifacts/runs/02_domain_generalization/v7_messidor_train/checkpoints/best.pt

#### 결과

| 지표 | v17_focal_g2 (배포 best) | v20_coral |
|---|---|---|
| DDR AUROC | **0.8911** | 0.8754 |
| Optimal threshold | **0.42** | 0.20 |
| Sensitivity@optimal | **0.7727** | 0.7625 |
| Specificity@optimal | 0.8564 | 0.8519 |
| F1@optimal | **0.8063** | 0.7981 |
| Precision@optimal | 0.9066 | 0.8371 |

- 평가 결과: `artifacts/runs/02_domain_generalization/v20_coral/evaluations/external_test_v20_coral_best_metrics.json`
- threshold=0.5 기본: sensitivity 0.5700, specificity 0.9726 (극단적 보수 판정)

#### 분석

CORAL은 학습 도메인 쌍(APTOS↔IDRiD↔Messidor) 간 covariance를 정렬하지만, 실제 평가 도메인인 DDR은 학습에 포함되지 않는다. 즉 CORAL이 줄인 도메인 갭은 source–source 갭이며, source→DDR 갭은 직접 표적화되지 않는다. λ=1.0은 분류 손실에 비해 CORAL 정규화 강도가 과도하여, 특히 focal γ=2.0의 hard example calibration 효과를 방해했을 가능성이 크다 — threshold가 0.42→0.20으로 후퇴한 것이 그 증거다. SWAD(v19)가 focal calibration을 희석한 것과 동일한 메커니즘.

**결론: CORAL이 학습 소스 도메인 간 정렬에는 작동하나, unseen target(DDR) 일반화 개선에는 기여하지 못했다. λ 감소 또는 source-DDR 직접 정렬 없이는 효과 없음. v17_focal_g2가 Sprint 3 최종 best. 폐기.**

---

### [SPRINT 3] — v7_512_messidor_train + v21_512_layercam: 512px 해상도 전환 실험 (2026.05.03)

#### 배경 및 설계

v17_focal_g2(448px, AUROC 0.8911)를 기준으로, 입력 해상도를 512px로 올리면 민감도 부족(0.773 → 목표 0.90) 문제를 완화할 수 있다는 가설을 검증했다. 주요 논거:

1. **해상도 이득**: 512px feature map은 16×16 (448px는 14×14), Layer-CAM 정밀도 30% 향상
2. **Layer-CAM 전환**: Grad-CAM의 전역 gradient 평균화 → 소병변 localization 개선
3. **일관된 학습 체인**: v7 재학습(512px)→v21(512px)으로 해상도 불일치 제거

**변경 사항:**
- `configs/base.yaml`: `resize_size`, `image_size`, `preprocess_size` 448→512
- `drscreen/xai/gradcam.py`: `method` 파라미터 추가, Layer-CAM 분기 (`torch.relu(gradient)` element-wise)
- `drscreen/infer/service.py`: `config["infer"]["gradcam_method"]`로 method 선택
- `configs/v7_512_messidor_train.yaml`, `configs/v21_512_layercam.yaml` 신규 생성

**batch_size 이슈**: 학습 중 VRAM 97.8% 포화(RTX 5070 Ti, 16GB)로 에포크당 60분 소요. batch_size 16→12로 수정 후 2분/에포크로 정상화.

#### 결과

**v7_512_messidor_train (val 기준, 내부):**
- val AUROC: 0.9974 (finetune epoch 5), 최종: ~0.9974
- val sensitivity: 0.9845, val specificity: 0.9942

**v21_512_layercam (DDR external_test, n=12,522):**

| 지표 | v17_focal_g2 (배포 best) | v21_512_layercam | 변화 |
|---|---|---|---|
| DDR AUROC | **0.8911** | 0.8775 | ↓ -0.014 |
| Optimal threshold | **0.42** | 0.54 | 후퇴 |
| Sensitivity@optimal | **0.7727** | 0.7356 | ↓ -0.037 |
| Specificity@optimal | 0.8564 | 0.8982 | ↑ +0.042 |
| F1@optimal | **0.8063** | 0.8006 | ↓ -0.006 |

- 평가 결과: `artifacts/runs/03_resolution_layercam/v21_512_layercam/evaluations/external_test_v21_512_layercam_best_metrics.json`

#### 분석

512px 해상도 전환은 외부 도메인 일반화를 개선하지 못했다. 두 가지 가설:

**가설 A — 오버피팅**: val 셋(366샘플)은 소규모·in-distribution이다. v7_512가 val AUROC 0.9974에 도달했다는 것은 val 분포에 과적합되었음을 시사한다. val-DDR 갭이 0.116(v21)으로 v17(~0.07 추정)보다 크다. 512px의 높은 spatial resolution이 도메인 특이적 텍스처를 더 촘촘히 학습해 unseen DDR에서 역효과를 냈을 가능성.

**가설 B — FDA 보정 미흡**: 448px에서 α=0.05는 b=22px 교환. 512px에서는 b=25px로 절대값이 소폭 증가하나 비율(5%)은 동일. 단, 512px 입력에서 도메인 특이적 중주파 패턴의 공간 규모가 더 크며, FDA의 저주파 교환만으로는 이를 커버하지 못할 수 있다. α=0.10은 v15_fda_a10(448px)에서 AUROC -0.025를 보였으므로 단순 상향은 배제.

**결론: 512px 전환 단독으로는 일반화 개선 불충분. 오버피팅 완화 또는 512→448 crop 방식 재고 필요. v17_focal_g2가 Sprint 3 기준 최종 best 유지.**

---

### [SPRINT 3] — Attention Ablation + Messidor 포함 효과 분리 실험 (2026.05.06)

#### 배경 및 설계

Sprint 3에서 v24(attention ✓, Messidor ✗)가 DDR AUROC 0.8452로 v21(0.8775) 대비 오히려 하락했다. 원인이 attention 모듈인지, Messidor 제외인지 특정할 수 없는 confounded 설계였다. 이를 해소하기 위해 2×2 ablation matrix를 구성했다.

| 모델 | attention | Messidor 학습 포함 | ext AUROC (DDR) |
|---|---|---|---|
| v24_multitask | ✓ | ✗ | 0.8452 |
| v28_no_attention | ✗ | ✓ | **0.8924** |
| v29_with_attention | ✓ | ✓ | 0.8628 |

**v28 설계**: `use_attention: false` — 당시 명명은 "no attention"이지만 현재 코드 taxonomy로는 `attention_mode: eca`에 해당한다. 즉 CBAM spatial은 제거됐지만 EfficientNet SE 위치의 ECA channel module은 유지된다. `base.yaml`의 `train_exclude_domains: [Messidor]` → `[]` 로 변경해 Messidor를 학습에 포함.

**v29 설계**: `use_attention: true` + Messidor 포함. v28과 Messidor만 동일하게 고정하여 attention 단독 효과를 분리. attention 모듈 가중치가 v7 사전학습 체크포인트에 없어 random 초기화로 시작 → epoch 1 val AUROC 0.8094로 v28(0.9395) 대비 느린 수렴.

**추가 실험 — v27 MIL attention**: Gated Attention Pooling(Ilse et al., NeurIPS 2018)을 분류 헤드로 적용. 공간 위치를 독립 인스턴스로 처리해 어텐션 가중치를 XAI 히트맵으로 직접 활용 가능하도록 설계. `drscreen/models/mil_attention.py` 신규 구현.

#### 결과

**v28_no_attention** (val best epoch 9, finetune 14 epochs):
- val AUROC: 0.9995
- DDR external_test AUROC: 0.8924, optimal threshold: 0.45
- DDR sensitivity@optimal: 0.748, specificity@optimal: 0.906

**v29_with_attention** (val best epoch 12, finetune 12 epochs):
- val AUROC: 0.9994
- DDR external_test AUROC: 0.8628, optimal threshold: 0.44
- DDR sensitivity@optimal: 0.699, specificity@optimal: 0.899

**Ablation 분리 결과:**

| 비교 | 고정 변수 | 변화 변수 | AUROC 차이 |
|---|---|---|---|
| v28 vs v29 | Messidor 포함 | attention 유무 | **−0.030** (attention이 유해) |
| v24 vs v29 | attention 동일(✓) | Messidor 포함 유무 | **+0.018** (Messidor가 유익) |

- 평가 결과: `artifacts/runs/05_xai_attention_ablation/v28_no_attention/evaluations/external_test_v28_no_attention_best_metrics.json`
- 평가 결과: `artifacts/runs/05_xai_attention_ablation/v29_with_attention/evaluations/external_test_v29_with_attention_best_metrics.json`

#### 분석

**ECA+CBAM attention의 DDR 일반화 저해**: attention은 val 세트(0.9994~0.9995)에서는 거의 동일한 성능을 보이나 DDR에서 −0.030의 격차를 만든다. attention 가중치가 학습 도메인(APTOS + IDRiD + Messidor)의 특이적 패턴에 과도하게 집중해 unseen DDR에서 shortcut-driven mislocalization을 강화하는 것으로 해석된다. Sprint 3의 block sweep에서 v24(attention ✓)가 낮은 레이어에서 shortcut attribution을 보인 관찰과 일치한다.

**Messidor 포함의 긍정적 효과**: v24→v29(+0.018)는 Messidor 단독 기여다. Messidor는 유럽 기반 안저 카메라 도메인으로, DDR(중국 기반)과 완전히 겹치지 않으나 추가 도메인 다양성이 외부 일반화에 기여한다.

**결론: 이 시점 기준으로는 v28_no_attention이 최고 외부 성능 모델(DDR AUROC 0.8924). attention 제거와 Messidor 포함의 두 결정이 모두 일반화에 기여했으며, 특히 attention 제거 효과(−0.030 방지)가 더 크다.**

#### 코드베이스 리팩토링 (동반 작업)

`drscreen/train/runner.py`의 과도한 책임을 분리해 테스트 가능성 및 재사용성을 높였다.

- `train/model_setup.py` 신규: `resolve_device`, `build_criterion`, `build_model_for_eval`, `validate_training_scope`, `TrainingPhase`
- `train/checkpointing.py` 신규: `checkpoint_payload()` 저장 로직
- `train/data_loader_factory.py` 신규: DataLoader / transform 생성 팩토리
- `train/evaluate.py` 신규: `run_split_evaluation()` — 평가 실행 + 출력 경로 자동 결정
- `xai/evaluation.py` 신규: XAI IoU 평가 핵심 로직 (`eval_xai_iou.py`에서 분리)
- `data/mask_providers.py` 신규: `LesionMaskProvider` Protocol + `IDRiDMaskProvider` / `NullMaskProvider`
- `infer/payload.py` 신규: `InferencePayload` dataclass

Artifact 경로도 `artifacts/checkpoints/`, `artifacts/evaluations/` 단일 플랫 구조에서 `artifacts/runs/<primary_group>/<run_id>/` 계층 구조로 전면 이동했다. 모든 config의 `checkpoint_path` / `pretrained_backbone_path`가 새 경로로 업데이트됐다.

---

## 4. 수정 대기 항목

### BUG-03 — 정규화 통계가 ImageNet 기준

**파일**: `drscreen/models/profiles.py`

**현상**

현재 mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)로 ImageNet 통계를 사용한다. 안저 이미지는 녹색 채널이 지배적이고 전체적으로 어두운 분포를 가지므로 도메인 mismatch가 존재한다.

**수정 방향**

APTOS + IDRiD 학습 분할 전체에서 채널별 mean/std를 직접 계산하여 config에 명시. 단, pretrained weights와 함께 쓰는 경우 ImageNet 통계 유지가 일반적으로 권장되므로, 실험적으로 둘을 비교한 후 결정한다.

**우선순위**: 낮음 — 성능 개선이 불확실하므로 실험 후 판단

---

## 5. 개선 계획

### [완료 — Sprint 2] FEAT-11 — v9_fda: FDA 저주파 색조 교환 (Domain Generalization)

**결과**: DDR AUROC 0.8812 (v7 대비 +0.0087), optimal threshold 0.09 → 0.19. 성공 기준 모두 충족. **v9_fda가 현재 global best(배포 지정)**. 상세 내용은 변경 이력 v9_fda 섹션 참조.

---

### [완료 — Sprint 2] v10_swad: SWAD (보조 참고)

**결과 (BN 재보정 버그 수정 후, 2026.04.21 재평가)**: DDR AUROC 0.8863, optimal threshold 0.05. AUROC는 v9_fda(0.8812)보다 높으나 threshold bias가 0.05로 더 악화되어 배포 부적합 판정 강화. v9_fda가 AUROC·threshold 균형 기준으로 배포 지정됨.

---

### [폐기 — Sprint 3] FEAT-12 — CORAL: 2차 통계량 정렬 도메인 일반화

**목표**: 소스 도메인 간 feature 공분산 행렬을 정렬해 도메인 불변 표현 학습. v20_coral로 구현·평가 후 폐기 완료. 현재 코드는 재현을 위해 `CoralLoss`/`use_coral` 경로를 유지한다.

**기법**: CORAL (Sun & Saenko, ECCV 2016) — 소스 도메인 쌍 feature 공분산 차이를 Frobenius norm으로 최소화.

$$\mathcal{L}_{\text{CORAL}} = \frac{1}{4d^2} \left\| C_S - C_T \right\|_F^2$$

**구현 범위**: `drscreen/train/loss.py`에 `CoralLoss` 추가, runner.py에 domain-grouped feature 추출 로직 추가.

**결과**: v20_coral — DDR AUROC 0.8754 (v17 대비 -0.0157), optimal threshold 0.20 (v17 대비 역행). CORAL이 source–source 갭만 정렬하고 unseen target(DDR)에는 직접 작용하지 않음. λ=1.0 과도 정규화로 focal calibration 희석. 상세 내용은 변경 이력 v20_coral 섹션 참조. **폐기 확정(2026.04.29).**

---

### [TODO — Sprint 3] Temperature Scaling (post-hoc 캘리브레이션)

**목표**: 모델 가중치 변경 없이 단일 스칼라 $T$를 학습해 출력 확률을 재보정. optimal threshold를 0.5 근방으로 끌어올리는 것이 목적.

**전제 조건**: DDR 레이블 있는 calibration split 확보 (external_test 12,522장에서 일부 분리 필요).

---

### [폐기 — Sprint 3] v19_swad_focal_g2: SWAD + focal γ=2.0

**결과**: DDR AUROC 0.8833 (v17 대비 -0.0078), optimal threshold 0.06 (v17 대비 대폭 역행). SWAD 에폭 평균이 focal γ=2.0의 후반 수렴 calibration을 희석. 상세 내용은 변경 이력 v19 섹션 참조.

---

### TTA (Test-Time Augmentation) — 폐기

artifacts/runs/01_ssl_lineage/v6_alpha_only/checkpoints/best.pt 기준 test split에서 4-view TTA(원본, H-flip, V-flip, H+V-flip) 적용 결과 AUROC 0.9893 → 0.9910으로 미세 개선됐으나, false positive가 23 → 26으로 증가해 specificity가 0.9013 → 0.8884로 하락. IDRiD specificity는 0.4706 → 0.3824으로 특히 악화됐다. flip 앙상블이 정상 이미지의 예측 확률을 이상 방향으로 편향시키는 부작용이 있어 폐기. 코드에서 제거됨.

---

### [폐기 — Sprint 2] FEAT-05 — Foundation model backbone 교체 (RETFound)

**결과**: retfound_v1(BCE) Messidor AUROC 0.6722, retfound_v2(Focal) 0.6611 — 기존 v6_alpha_only(0.8697) 대비 -0.197~-0.208. **폐기 확정(2026.04.10).**

**실패 원인**: 224px 해상도 열세, 소규모 데이터(4K)에서 307M ViT 귀납 편향 부족, EfficientNet의 5세대 누적 fine-tuning 계보 부재.

관련 코드 전체 `archive/retfound/`로 이동 완료. 상세 내용은 변경 이력 retfound / retfound_simclr 섹션 참조.

---

### FEAT-04 — 다중 레이블 분류 확장 (장기)

현재 normal / abnormal 이진 분류만 지원한다. 추후 DR 중증도 5단계(0–4) 또는 DME 위험도 3단계 분류로 확장할 때를 대비해 `model.num_outputs`와 loss 선택 로직을 일반화한다. 현재 `_validate_training_scope()`에서 `num_outputs == 1`을 강제하는 부분이 진입점이다.

---

### [TODO — Sprint 3] v22_512pre_448model: 512 전처리 + 448 model input

**배경**: v21(512→512)이 v17(448→448) 대비 DDR AUROC -0.014, sensitivity -0.037로 역전됐다. 처음부터 권장된 512→448 접근(512로 전처리해 정보 보존, 모델 입력은 EfficientNet-B5 native 456px에 가까운 448)을 검증한다.

**설계:**
- `preprocess_size: 512` (유지)
- `resize_size: 512` (유지)
- `image_size: 448` (←변경, CenterCrop 448)
- backbone: `artifacts/runs/03_resolution_layercam/v7_512_messidor_train/checkpoints/best.pt` (유지)
- 나머지: v21과 동일 (FDA α=0.05, focal γ=2.0, backbone_lr=4e-5)

**가설**: 전처리 단계에서 512px 해상도로 원형 자르기 및 Ben Graham 정규화를 수행해 주변부 정보를 더 많이 보존하면서, 모델 입력은 448px CenterCrop으로 EfficientNet-B5 설계 해상도(456px)에 가깝게 맞춘다. 512→512에서 발생한 도메인 특이적 고주파 텍스처 과적합을 줄일 수 있다.

**예상 config 변경**: `base.yaml`에서 `image_size: 512 → 448` 한 줄. 또는 v22 전용 config에서 오버라이드.

---

## Sprint 3 — XAI 정량 검증: IDRiD 병변 마스크 IoU

### [완료] XAI IoU 평가 인프라 구축

**목적**: Layer-CAM 히트맵이 DR 분류 근거로 임상적으로 올바른 위치(병변 영역)를 보고 있는지 정량 측정.

**추가 파일**:
- `drscreen/xai/iou.py` — IoU 유틸리티 (마스크 로딩, CAM 이진화, IoU/Pointing-Game 계산)
- `eval_xai_iou.py` — CLI 평가 스크립트 (IDRiD segmentation set, top-N% 이진화, JSON 출력)

**방법론**:
1. IDRiD segmentation training set 54장 사용 (MA · HE · EX · SE 4종 병변 마스크 보유)
2. Layer-CAM 히트맵을 retina 픽셀 상위 N%로 이진화 (N = 10, 20, 30)
3. 4종 마스크를 union하여 "병변 전체 영역" GT 마스크 생성
4. 이진 CAM과 GT 마스크 간 IoU 계산
5. Pointing Game: CAM 최대 활성화 픽셀이 병변 마스크 내부에 있는지 (0/1)
6. 공간 정합성을 위해 FundusPreprocess 비활성화 (v21 학습 시에도 미적용)

**v21_512_layercam 결과** (`artifacts/runs/03_resolution_layercam/v21_512_layercam/evaluations/xai_iou_v21_512_layercam_train.json`):

| 지표 | 값 |
|---|---|
| Pointing Game 정확도 | **0.1111** (6/54장) |
| 평균 Union IoU (top-10%) | 0.0306 |
| 평균 Union IoU (top-20%) | 0.0300 |
| 평균 Union IoU (top-30%) | 0.0304 |

**해석**:

임의(random) 기준선 추정: 병변이 retina 면적의 약 5~10%를 차지할 때, top-20% 이진화 기준 random IoU ≈ 4~7%. **실측 IoU 3%는 임의 기준선보다 낮다.** Pointing Game 11%도 우연 수준(retina 내 병변 커버리지 기준 기대값 ~5~15%).

→ **v21은 분류 결정을 병변 위치가 아닌 전역적(global) 이미지 통계/텍스처 기반으로 수행하고 있음.** AUROC 0.8775를 달성했지만 Layer-CAM이 가리키는 영역은 임상적 병변과 정렬되지 않는다.

**원인 가설**:
- 이미지 레벨 weak supervision: 픽셀 레벨 지도 신호 없으므로 모델이 병변 위치를 학습할 유인이 없음
- 병변(MA, 미세동맥류 등)은 크기가 작아 CAM 해상도(16×16 feature map)에서 검출이 어려움
- 모델이 광역 텍스처(혈관 패턴, 색상 분포)로 DR을 분류하는 단축 학습(shortcut learning) 가능성

**임상적 함의**: 현재 AI의 히트맵은 "어디가 문제인지"를 임상의에게 설명하는 용도로 사용 불가. XAI 오버레이를 "진단 근거"가 아닌 "모델 관심 영역" 참고 정보로만 제시해야 함.

**다음 단계 옵션**:
1. v17(Grad-CAM)과 동일 평가 → Layer-CAM vs Grad-CAM 비교
2. 히트맵 스무딩 / 다중 레이어 앙상블 CAM 실험
3. 병변 레벨 지도 신호 추가 (예: IDRiD segmentation mask를 auxiliary loss로 활용)

---

### [완료] 옵션 A: 얕은 레이어 CAM 비교 실험

**배경**: blocks[-1] (16×16)이 너무 낮은 공간 해상도여서 미세 병변(MA 등)이 소실될 수 있다는 가설 검증.

**방법**: `eval_xai_iou.py --target-block {2,3,4}` 로 각 레이어에서 동일한 IoU 평가 수행.

**결과** (v21_512_layercam, IDRiD training set 54장):

| Block | Feature Map | Pointing Game | IoU top-10% | IoU top-20% | IoU top-30% |
|---|---|---|---|---|---|
| blocks[2] | 64×64 | 0.111 | 0.026 | 0.027 | 0.027 |
| blocks[3] | 32×32 | 0.056 | 0.024 | 0.025 | 0.026 |
| blocks[4] | 32×32 | 0.019 | 0.029 | 0.028 | 0.029 |
| **blocks[6] (기본)** | **16×16** | **0.111** | **0.031** | **0.030** | **0.030** |

**결론**: 레이어 변경으로는 개선 없음. 더 얕은 레이어는 오히려 Pointing Game이 낮아진다. 모든 레이어에서 IoU ~2.5~3.1%로 임의 기준선(~4~7%)을 밑돈다.

**근본 원인 확정**: feature map 해상도 문제가 아니라 **학습 목표의 부재**. 이미지 레벨 이진 레이블만으로는 어떤 레이어도 병변 위치를 학습할 유인이 없다. → **옵션 B(multi-task auxiliary loss)만이 근본 해결책.**

---

### [완료 — Sprint 3] 옵션 B: Multi-task Auxiliary Loss

**목표**: 분류 정확도를 유지하면서 XAI 공간 정렬(IoU)을 개선.

**설계**:
- 기존 분류 head(EfficientNet-B5 + ECA attention) 유지
- 추가: 얕은 segmentation auxiliary head (blocks[2] 또는 blocks[3] 출력 → upsampling → 1×H×W sigmoid)
- Loss = `focal_cls_loss + λ * bce_seg_loss` (λ = 0.3~0.5 탐색)
- IDRiD 54장 (MA+HE+EX+SE union mask)를 auxiliary supervision으로 사용
- IDRiD 외 이미지: seg head loss = 0 (마스크 없음)

**기대 효과**: 분류 backbone이 병변 위치를 인코딩하도록 유도 → Layer-CAM이 임상적으로 의미 있는 위치를 가리킴.

**위험**: IDRiD 54장은 소규모 → seg head 과적합 가능. 해결책: seg head를 frozen backbone 위에 thin head로 제한, 강한 dropout(0.5).

**구현** (`drscreen/models/aux_seg.py`):
- `MultiTaskModel`: EfficientNet-B5 backbone + `_SegAuxHead` (blocks[2] → 1×1 Conv → upsample → 512×512)
- forward hook으로 blocks[2] feature 캐싱, train 시 `(cls_logits, seg_logits)` 반환, eval 시 `cls_logits`만 반환
- `predict_seg()`: eval 모드에서 backbone 실행 후 seg head sigmoid 출력 [B,1,H,W] 반환
- `drscreen/train/engine.py`: `train_one_epoch(lambda_aux_seg=0.3)` — IDRiD 마스크 있는 배치만 seg loss 적용

**결과 — v24_multitask (λ=0.3, 2026.05.04)**:

분류 성능 artifact는 internal `test` split 469장과 DDR `external_test` 12,522장 기준이 모두 존재한다. 초기 문서에는 `external_test_v24_multitask_best_metrics.json` 부재로 기록됐으나, 2026-05-06 registry 재분류 시점에는 해당 artifact가 존재한다.

| 구분 | 지표 | 값 | 근거 artifact |
|---|---|---|---|
| v24 분류 | test AUROC | **0.991989** | `artifacts/runs/04_lesion_supervision/v24_multitask/evaluations/test_v24_multitask_best_metrics.json` |
| v24 분류 | optimal threshold | 0.86 | `artifacts/runs/04_lesion_supervision/v24_multitask/evaluations/test_v24_multitask_best_metrics.json` |
| v24 분류 | sensitivity@opt / specificity@opt | 0.923729 / 0.987124 | `artifacts/runs/04_lesion_supervision/v24_multitask/evaluations/test_v24_multitask_best_metrics.json` |
| v24 DDR 분류 | external_test AUROC | 0.845189 | `artifacts/runs/04_lesion_supervision/v24_multitask/evaluations/external_test_v24_multitask_best_metrics.json` |
| v24 DDR 분류 | external_test optimal threshold | 0.17 | `artifacts/runs/04_lesion_supervision/v24_multitask/evaluations/external_test_v24_multitask_best_metrics.json` |
| v21 DDR 참고 | external_test AUROC | 0.877540 | `artifacts/runs/03_resolution_layercam/v21_512_layercam/evaluations/external_test_v21_512_layercam_best_metrics.json` |
| v21 XAI 기준 | train 54장 Pointing / top-20 IoU | 0.111111 / 0.029956 | `artifacts/runs/03_resolution_layercam/v21_512_layercam/evaluations/xai_iou_v21_512_layercam_train.json` |
| v24 XAI | test 27장 Pointing / top-20 IoU | 0.037037 / **0.032054** | `artifacts/runs/04_lesion_supervision/v24_multitask/evaluations/xai_iou_v24_multitask_default_test.json` |

분류 internal test 성능은 유지됐고 XAI IoU는 v21 train 기준선 대비 소폭 상승했다. 단, v21 train 54장과 v24 test 27장은 같은 split이 아니므로 직접 비교에는 한계가 있다. 당시 active config/deploy 지정은 v24_multitask로 변경됐으나, DDR external_test artifact 기준으로는 AUROC 0.845189로 v17/v21/v28 대비 외부 일반화 성능이 약하다. 이후 active config는 v28_no_attention으로 변경됐다.

---

### [폐기 — Sprint 3] v25_multitask_l1: λ=1.0 보조 손실 강화 실험

**목적**: λ=0.3이 너무 약해 XAI 개선 폭이 미미한 것으로 판단, λ=1.0으로 seg head 감독 강화.

**결과** (분류: internal test split 469장, XAI: IDRiD test 27장, 2026.05.04):

| 지표 | v24 (λ=0.3) | v25 (λ=1.0) | 변화 |
|---|---|---|---|
| test AUROC | 0.9920 | 0.9271 | **−0.0649** |
| XAI IoU top-20% | 0.032 | 0.025 | −0.007 |
| XAI Pointing Game | 0.037 | 0.000 | −0.037 |

v25는 현재 internal `test`와 XAI artifact만 있고 DDR `external_test` artifact는 없다. v24는 2026-05-06 registry 재분류 시점에 DDR `external_test` artifact가 존재한다. **결론**: internal test 분류 성능이 크게 하락했고 XAI 지표도 개선되지 않았다. seg loss가 강해질수록 backbone 표현이 병변 위치 분할에 편향되어 이진 분류 능력을 잃는다. λ=1.0 이상은 현재 아키텍처에서 실용 불가. **폐기 확정(2026.05.04).**

---

### [폐기 — Sprint 3] v24 seg_head 직접 히트맵 활용 (Option A)

**목적**: Layer-CAM 대신 v24의 aux seg head sigmoid 출력을 XAI 히트맵으로 직접 사용. seg head가 IDRiD 픽셀 마스크로 훈련되었으므로 Layer-CAM보다 병변 위치 정렬이 우수할 것으로 기대.

**구현**: `eval_xai_iou.py --use-seg-head` 플래그 추가. `_process_image()` 내 `predict_seg()` 호출 분기 추가.

**결과** (2026.05.05):

| 방법 | IoU top-20% | Pointing Game |
|---|---|---|
| v21 Layer-CAM (train 54장 기준선) | 0.030 | 0.111 |
| v24 Layer-CAM (test 27장) | 0.032 | 0.037 |
| **v24 seg_head 직접 (test 27장)** | **0.032** | **0.000** |

**결론**: IoU는 Layer-CAM과 동일하나 Pointing Game이 0.000 — 27장 전부에서 최대 활성화 지점이 병변 바깥. seg head가 λ=0.3의 약한 감독 하에서 병변 위치가 아닌 배경 영역에 더 높은 확률을 할당하고 있음. 직접 활용 불가. **폐기 확정(2026.05.05).**

---

### [완료 — Sprint 3] MixStyle · Classifier Dropout 코드 완전 제거

**배경**: 두 기법 모두 DEVLOG 기록상 성능 저하 입증된 상태에서 코드만 잔존했음.
- MixStyle: v5(Messidor −0.029), v8(DDR −0.035) 입증 → "재시도 금지"
- Classifier Dropout: v3 ablation AUROC −0.159 입증

**제거 범위**:
- `drscreen/models/mixstyle.py` 파일 전체 삭제
- `drscreen/models/build.py`: `use_mixstyle`, `classifier_dropout` 파라미터 및 관련 로직 제거
- `drscreen/settings.py`, `eval_ensemble.py`, `drscreen/ssl/trainer.py`: legacy checkpoint/config 전파 및 `build_model()` 호출부 정리
- `configs/*.yaml`: `use_mixstyle` 항목 제거. `classifier_dropout`/`zero_init_classifier`는 현재 config와 모델 팩토리에서 사용하지 않음

---

### [완료 — Sprint 3] XAI 평가 정상화: AUPRC / AUC-IoU / 기준선

**배경**: 단일 top-20% IoU는 threshold 선택 편향이 있어, 방법 간 공정 비교 불가 (Choe et al., CVPR 2020).

**추가 지표**:
- **Pixel AUPRC**: CAM을 연속 ranking score로 보고 병변 마스크 대비 PR-AUC 계산. Score calibration 영향 없음.
- **AUC-IoU**: threshold를 min→max로 sweep하여 평균 IoU. 단일 threshold 편향 제거.
- **기준선**: random heatmap, center Gaussian (σ=min(H,W)/4), retina uniform.

**구현**: `drscreen/xai/iou.py`에 `normalize_cam_fov()`, `compute_auprc()`, `compute_auc_iou()`, 기준선 생성기 3종 추가. `eval_xai_iou.py`에 `--baselines`, `--methods` 플래그 추가.

---

### [완료 — Sprint 3] XAI 방법 비교 실험 (v24, IDRiD test 27장)

**구현**: `drscreen/xai/gradcam.py`에 Grad-CAM++, Score-CAM(top-64 채널), Integrated Gradients(50-step) 추가.

**전체 결과**:

| Method | AUPRC | AUC-IoU | IoU-10% | IoU-20% | IoU-30% | Pointing Game |
|---|---|---|---|---|---|---|
| Grad-CAM | 0.0380 | 0.0104 | 0.0176 | 0.0210 | 0.0287 | **0.111** |
| Layer-CAM | 0.0390 | 0.0098 | 0.0221 | **0.0321** | 0.0333 | 0.037 |
| Grad-CAM++ | 0.0364 | 0.0113 | **0.0302** | 0.0316 | **0.0353** | 0.037 |
| IG | **0.0399** | 0.0039 | 0.0266 | 0.0317 | 0.0342 | 0.037 |
| Score-CAM | 0.0382 | **0.0300** | 0.0224 | 0.0265 | 0.0288 | 0.074 |
| *(Center Gaussian)* | *(0.0526)* | — | — | *(0.0436)* | — | — |
| *(Random)* | *(0.0348)* | — | — | *(0.0282)* | — | — |

**해석**:
- 5가지 방법 모두 center Gaussian 기준선(AUPRC 0.053)에 미달 — image-level supervision의 구조적 한계 재확인
- Score-CAM AUC-IoU(0.030)가 gradient 기반 방법 대비 3~8배 우수 — threshold 전 범위에서 더 균일한 공간 정렬
- IG AUPRC(0.040)가 가장 높음 — ranking 기반 localization 품질 최우수
- Layer-CAM이 fixed threshold(top-20%) IoU에서 Layer-CAM이 우수하고, 계산 비용 최소 → **Layer-CAM 유지 결정**
- 참고 논문(RobustDRNet 2026) 기준: Grad-CAM++ ~0.06, IG ~0.10 → 현재 우리 ~0.032는 아직 미달. 전용 분할 모델 없이 달성하기 어려운 수준

---

### Sprint 3 XAI 정량 개선 결론

**2026-05-05 기준 한계**: attention-enabled v24 계열에서는 image-level weak supervision 하에서 XAI IoU의 구조적 상한선이 ~0.032 수준으로 보였다. λ 강화, seg head 직접 활용, 5가지 CAM 방법 비교 모두 center Gaussian 기준선(0.044)을 넘지 못했다.

**2026-05-06 재분류 후 정정**: v24/v28 matched block sweep 결과, 문제는 "모든 backbone block이 병변 위치를 못 배우는 것"이 아니라 attention-enabled feature path에서 shortcut-driven XAI mislocalization이 발생하는 쪽으로 해석된다. v28_no_attention은 IDRiD test block4에서 Pointing Game 0.4444, IoU top-20 0.0741을 기록해 center Gaussian baseline 0.0436을 초과했다.

**현재 active 모델**: v28_no_attention — internal test AUROC 0.9923, DDR external_test AUROC 0.8924, DDR optimal threshold 0.45, XAI block4 Layer-CAM IoU top-20 0.0741. v24_multitask는 lesion-supervision 실험 기록으로 유지하되 배포 대상에서는 제외한다.

**향후 XAI 개선 가능성 (미실험)**:
- `aux_seg_block=4` (blocks[4], 더 깊은 semantic feature)로 seg head 재훈련
- IDRiD 전체 54장 + test 27장을 seg head 학습에 활용
- 전용 segmentation 모델 앙상블 (분류 모델 CAM 한계 우회)
- LAT-lite: multi lesion-filter head + diversity loss (CVPR 2021 근거)

---
