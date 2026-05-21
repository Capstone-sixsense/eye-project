# SPRINT 5 개발 요약

## 1. 개요
- **프로젝트 명**: eye-project (AI 파트: drscreen)
- **목표**: Sprint 4에서 확인한 shortcut-driven XAI 한계를 바탕으로, Phase 4-G data/representation leverage를 통해 병변 evidence 품질과 분류 성능 개선 가능성을 검증
- **기간**: SPRINT 5 주요 실험 이력 (2026.05.21 ~ 진행 중)
- **대상 범위**: Phase 4-G 전체, TJDR/DDR_SEG 통합, MAPLES ROI 좌표계 보정, v8b standalone lesion evidence baseline, v8b evidence classifier, v31+v8b late fusion 진단 및 AI-side 배포 패키징

> 현재: latency와 backend contract 검증 후 active deployment는 `v31_v8b_fusion_v2`로 변경했다. `configs/base.yaml`은 fusion config를 가리키고, `artifacts/checkpoints/best.pt`에는 v31 classifier + v8b segmenter + numeric meta-classifier composite checkpoint가 들어 있다. 이전 v31-only alias는 `artifacts/checkpoints/best_pre_fusion_v31_only.pt.bak`로 보존한다.

## 2. 주요 개발 및 성과

### 2.1 TJDR 통합 및 aligned retrain
- TJDR을 `data/raw/TJDR`에 확보했고, train 448쌍 / test 113쌍의 image-mask pair 무결성을 확인했다.
- `TJDRMaskProvider`는 TJDR label `1=EX`, `2=HE`, `3=MA`, `4=SE`를 프로젝트 channel order `MA/HE/EX/SE`로 재배열한다.
- `seg_evidence_v3_tjdr`는 TJDR 추가 후 IDRiD mDice **0.2055**, union IoU **0.2209**, TJDR mDice **0.3524**, union IoU **0.3490**을 기록했다.
- 그러나 MAPLES mDice는 **0.0051**, union IoU **0.0071**로 낮아 MAPLES cross-domain 문제가 남았다.

### 2.2 MAPLES 일반화 진단
- `seg_evidence_v4_deeplab_tjdr`, `seg_evidence_v5_maples_fda_tjdr`, `seg_evidence_v6_maples_finetune_tjdr`, `seg_evidence_v7_maples_only`를 확인했다.
- stronger encoder, MAPLES-target FDA, MAPLES-heavy fine-tune, MAPLES-only 학습 모두 MAPLES gate를 통과하지 못했다.
- 이 단계까지는 threshold 문제가 아니라 MAPLES mask/geometry 또는 domain representation 문제를 의심했다.

### 2.3 DDR_SEG 통합 및 MAPLES ROI 보정
- FGADR는 접근 절차가 복잡해 active path에서 제외하고, DDR lesion segmentation subset을 대체 데이터로 사용했다.
- `DDRSegMaskProvider`와 `build_manifest --include-ddr-seg`를 추가해 IDRiD/MAPLES/TJDR/DDR_SEG composite 학습을 구성했다.
- 이후 MAPLES annotation이 ROI-space mask라는 점을 확인하고, `MESSIDOR-ROIs.csv` 기반 ROI-to-MESSIDOR 복원을 적용했다.
- 보정 후 `seg_evidence_v8b_ddrseg_tjdr_maplesfix`가 현재 standalone lesion evidence best가 됐다.

### 2.4 v8b evidence classifier 및 late fusion
- G-4에서 v8b lesion-map scalar feature로 calibrated classifier를 학습했지만, best `v8b_evidence_classifier_grid_v1`도 DDR AUROC **0.8942**로 v31 **0.9160**보다 낮았다.
- G-5 초기 full external 탐색에서 v31 score와 v8b evidence feature를 late fusion하자 DDR AUROC **0.9379**로 v31을 넘었다.
- 당시 external balanced threshold 기준 Accuracy **0.8665**, Sensitivity **0.8314**, Specificity **0.9015**, F1 **0.8615**였다.
- 정식 threshold 정책은 DDR `external_test`를 stratified 20% calibration / 80% holdout으로 나누고, calibration split에서 v31 sensitivity guard 이상을 만족하는 threshold 중 specificity가 가장 높은 값으로 확정했다.
- 확정 threshold는 **0.38**이다. Holdout 기준 AUROC **0.9403**, Accuracy **0.8678**, Sensitivity **0.8118**, Specificity **0.9238**, F1 **0.8599**다.
- 이후 numeric meta-classifier checkpoint를 패키징하고 active alias를 교체해 `v31_v8b_fusion_v2`를 AI-side active deployment로 승격했다. backend/frontend 코드는 변경하지 않았다.

## 3. 주요 성능 지표

### 3.1 classifier 후보

| Run | 핵심 변경 | DDR AUROC | Threshold | Sensitivity | Specificity | 판정 |
|---|---|---:|---:|---:|---:|---|
| `v31_no_se_gated` | base classifier inside fusion | 0.9160 | 0.35 | 0.7983 | 0.8677 | previous active / fusion input |
| `v8b_evidence_classifier_grid_v1` | v8b lesion evidence scalar classifier | 0.8942 | 0.56 | 0.7639 | 0.8674 | v31 미달 |
| `v31_v8b_fusion_v2` | v31 score + v8b evidence late fusion, DDR 20/80 calibration | **0.9403** | **0.38** | **0.8118** | **0.9238** | active deployment |

### 3.2 segmentation evidence

| Run | Eval set | best mDice | best union IoU | 판정 |
|---|---|---:|---:|---|
| `seg_evidence_v3_tjdr` | IDRiD test | 0.2055 | 0.2209 | TJDR 추가 후 개선 |
| `seg_evidence_v3_tjdr` | TJDR test | 0.3524 | 0.3490 | 개선 |
| `seg_evidence_v3_tjdr` | MAPLES test | 0.0051 | 0.0071 | ROI 보정 전 실패 |
| `seg_evidence_v8b_ddrseg_tjdr_maplesfix` | IDRiD test | 0.4151 | 0.3903 | current evidence best |
| `seg_evidence_v8b_ddrseg_tjdr_maplesfix` | MAPLES test | 0.2928 | 0.2121 | ROI fix 후 회복 |
| `seg_evidence_v8b_ddrseg_tjdr_maplesfix` | TJDR test | 0.3788 | 0.3149 | current evidence best |
| `seg_evidence_v8b_ddrseg_tjdr_maplesfix` | DDR_SEG test | 0.3945 | 0.2880 | current evidence best |

## 4. 현재 판단 및 다음 작업

- **현재 active classifier**: `v31_v8b_fusion_v2`.
- **현재 active evidence module**: `seg_evidence_v8b_ddrseg_tjdr_maplesfix`.
- **v31 상태**: 단독 배포가 아니라 fusion 내부 base classifier로 보존.
- **다음 작업**:
  - Docker runtime에서 fusion checkpoint 로딩과 payload shape를 최종 확인.
  - 추가 MA/HE/EX/SE 4채널 병변 마스크 데이터 확장은 Sprint 5 후속 개선으로 유지하되, FGADR는 기본 경로에서 제외한다.

## 5. 근거 파일
- `ai/docs/DEVLOG.md`
- `ai/docs/EXPERIMENT_REGISTRY.md`
- `ai/docs/AI_HANDOFF.md`
- `ai/.omc/plans/xai_improvement_phase4g.md`
- `ai/artifacts/runs/09_evidence_segmentation/seg_evidence_v8b_ddrseg_tjdr_maplesfix/evaluations/`
- `ai/artifacts/runs/10_grounded_classifier/v8b_evidence_classifier_grid_v1/evaluations/v8b_evidence_classifier_grid_v1_metrics.json`
- `ai/artifacts/runs/10_grounded_classifier/v31_v8b_late_fusion_sweep_v1/evaluations/v31_v8b_late_fusion_sweep_v1_metrics.json`
- `ai/artifacts/evaluations/external_test_v31_v8b_fusion_v2_best_metrics.json`
- `ai/artifacts/runs/10_grounded_classifier/v31_v8b_fusion_v2/evaluations/v31_v8b_fusion_v2_latency_probe.json`

---
**[SPRINT 5 진행 중 정리]**
