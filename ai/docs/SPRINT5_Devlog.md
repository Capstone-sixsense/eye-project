# SPRINT 5 개발 요약

## 1. 개요
- **프로젝트 명**: eye-project (AI 파트: drscreen)
- **목표**: Sprint 4에서 확인한 shortcut-driven XAI 한계를 바탕으로, Phase 4-G data/representation leverage를 통해 병변 evidence 품질과 분류 성능 개선 가능성을 검증
- **기간**: SPRINT 5 주요 실험 이력 (2026.05.21 ~ 진행 중)
- **대상 범위**: Phase 4-G 전체, TJDR/DDR_SEG 통합, MAPLES ROI 좌표계 보정, v8b standalone lesion evidence baseline, v8b evidence classifier, v31+v8b late fusion 진단 및 AI-side 배포 패키징, safezoom/contentcrop preprocessing 진단, 2026-05-25 domain-overfit mitigation 재검증

> 현재: 2026-05-29 train-serve preprocessing consistency를 위해 active deployment는 `v31_v8b_fusion_quickqual_v1`로 변경했다. `configs/base.yaml`은 QuickQual-line fusion config를 가리키고, `artifacts/checkpoints/best.pt`에는 QuickQual-line v31 classifier + QuickQual-line v8b segmenter + numeric StandardScaler/LogReg meta-classifier composite checkpoint가 들어 있다. 이전 hflip active alias는 `artifacts/checkpoints/best_pre_quickqual_v1_20260529.pt.bak`, 이전 score-level fusion alias는 `artifacts/checkpoints/best_pre_features_hflip_v2_20260527.pt.bak`로 보존한다.
> 2026-05-25 domain-overfit mitigation 실행 후에도 active deployment와 best staging candidate는 변경하지 않았다. classifier DG 후보는 active fusion을 넘지 못했고, segmentation DG 후보는 v8b evidence baseline 대비 모두 회귀했다.
> 2026-05-24 safezoom/contentcrop preprocessing 실험도 active deployment를 넘지 못해 승격하지 않았다.
> 2026-05-26 legacy `processed/images` mask geometry regression을 수정했다. 이후 2026-05-27에는 circular+hflip fusion이 active로 승격됐고, 2026-05-29에는 train-serve preprocessing consistency를 위해 QuickQual-line `v31_v8b_fusion_quickqual_v1`이 active로 교체됐다. 현재 active AI config는 backend QuickQual 결과를 입력으로 받기 위해 `infer.preprocess_mode: none`, threshold `0.06`을 사용한다.

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
- 당시 확정 threshold는 **0.38**이다. Holdout 기준 AUROC **0.9403**, Accuracy **0.8678**, Sensitivity **0.8118**, Specificity **0.9238**, F1 **0.8599**다.
- 이후 numeric meta-classifier checkpoint를 패키징하고 active alias를 교체해 `v31_v8b_fusion_v2`를 AI-side active deployment로 승격했다. 이 버전은 2026-05-27 `v31_v8b_fusion_features_hflip_v2`로 supersede됐다. backend/frontend 코드는 변경하지 않았다.

### 2.5 safezoom/contentcrop preprocessing 진단
- `preprocessing_safezoom_plan.md` 기반으로 contentcrop/safezoom 전처리 manifest와 classifier/segmenter/late-fusion 조합을 분리 진단했다.
- safezoom aspect audit은 17,900장을 대상으로 수행했다. 1.2-1.5 aspect bin에서 foreground height median은 contentcrop 406px -> safezoom 443px로 개선됐고, safe foreground loss p90은 6.41%, 10% 초과 loss는 0건이었다.
- 샘플 `1ae8c165fd53.png`는 content height 396 -> safe height 442, foreground loss 3.27%였다.
- 샘플 `4dd71fc7f22b.png`는 content height 384 -> safe height 422, foreground loss 7.85%였다.
- 최종 late fusion에서는 `v31_v8b_late_fusion_safezoom_v1` AUROC **0.9295**가 `v31_v8b_late_fusion_contentcrop_v1` AUROC **0.9280**보다 소폭 높았지만, active `v31_v8b_fusion_v2` AUROC **0.9403**보다 낮았다.
- safezoom backbone-retrained late fusion(`v31_v8b_late_fusion_safezoom_bbretrain_v1`)은 AUROC **0.9204**로 더 낮아졌다.
- segmenter는 safezoom/contentcrop 모두 MAPLES mDice를 높였지만 IDRiD 및 DDR_SEG 쪽이 회귀했다. 따라서 preprocessing variant는 diagnostic-only로 기록하고 active 경로에는 반영하지 않는다.

### 2.6 domain-overfit mitigation 재검증
- Phase 0에서 `manifest_with_val_mixed.csv`를 만들고 DDR `external_test`를 `external_calibration` 2,504장 / `external_holdout` 10,018장으로 분리했다.
- 배포 기준선 `v31_v8b_fusion_v2` holdout 재현은 AUROC **0.9402549575**, threshold **0.38**, Sens **0.8118**, Spec **0.9238**로 PASS였다.
- classifier 선택은 내부 validation AUROC가 아니라 `external_calibration_auroc`로 수행했다. Selection sanity에서 `external_calibration` vs `external_holdout` Spearman은 **1.0**, `val_mixed` vs `external_holdout` Spearman은 **0.6**이었다.
- `v42_coral_baseline`과 `v42_rsc_coral`은 v31 base classifier보다 AUROC가 소폭 높았지만 active fusion보다 낮고 D5 domain probe가 각각 **0.9855**, **0.9957**로 높아 shortcut 해소로 보지 않았다.
- `v41_ampmix`는 external_holdout AUROC **0.9027**로 v31 base보다도 낮았다.
- segmentation DG 후보 `seg_evidence_v8b_swa`, `seg_evidence_v9_gin`, `seg_evidence_v10_adverin`은 모두 IDRiD/MAPLES/TJDR/DDR_SEG mDice가 v8b baseline보다 낮았다.
- A1 pseudo-lesion augmentation은 v8b mask quality audit fail ratio **0.7701**로 차단했고, B4 CBMT는 MESSIDOR-2/source-free adaptation 입력 부재로 실행하지 않았다. 따라서 `fusion_v3`는 생성하지 않았다.
- 2026-05-26 재검증에서 B-G0 v8b 재현 실패의 주원인은 환경이 아니라 legacy `processed/images`를 contentcrop geometry로 align하던 코드 회귀로 확인됐다. `processed/images`는 circular geometry로 되돌리고, `processed_contentcrop/images`/`processed_safezoom/images`는 각 전처리 geometry를 유지하도록 분리했다.
- `seg_evidence_v8b_repro_seed43_geometryfix`는 Python 3.14 / torch 2.9.1+cu130 / RTX 5070 Ti deterministic 환경에서 best val mDice **0.3330**을 기록해 old v8b **0.3388**에 근접했다. Test best mDice는 IDRiD **0.3867**, MAPLES **0.2772**, TJDR **0.3923**, DDR_SEG **0.3889**였다.

### 2.7 fusion complementarity 진단
- `fusion_complementarity_plan.md` Phase 0/Phase C를 QuickQual active line 기준으로 실행했다.
- Phase 0에서 v31/v8b 조합의 complementarity ceiling은 높지 않았다. Q-statistic은 **0.888**, v8b가 v31 오류를 고친 비율은 **37%**, holdout both-wrong은 **1,326건**이었다.
- Phase C에서 unweighted calibration-fit(`calfit_none`)은 holdout AUROC **0.940840**으로 active train-fit policy **0.934129**보다 높았다. Paired bootstrap CI는 **+0.00417 ~ +0.00923**이었다.
- 다만 `calfit_none`의 sensitivity는 **0.8010**으로 active **0.8234**보다 낮고, residual-weighting 후보들은 모두 `calfit_none`보다 낮았다.
- 결론: residual complementarity 가설은 실패로 기록한다. `calfit_none`은 별도 threshold/meta-fit policy 후보일 뿐이고, sensitivity-guard와 staging 검증 없이는 active 배포로 승격하지 않는다.

## 3. 주요 성능 지표

### 3.1 classifier 후보

| Run | 핵심 변경 | DDR AUROC | Threshold | Sensitivity | Specificity | 판정 |
|---|---|---:|---:|---:|---:|---|
| `v31_no_se_gated` | base classifier inside fusion | 0.9160 | 0.35 | 0.7983 | 0.8677 | previous active / fusion input |
| `v41_ampmix` | AmpMix domain-generalization diagnostic | 0.9027 | 0.40 | 0.7882 | 0.8496 | v31/fusion 미달 |
| `v42_coral_baseline` | CORAL domain-generalization diagnostic | 0.9203 | 0.25 | 0.7940 | 0.8949 | v31 소폭 상회, fusion/D5 gate 미달 |
| `v42_rsc_coral` | RSC+CORAL domain-generalization diagnostic | 0.9201 | 0.44 | 0.7920 | 0.9021 | v31 소폭 상회, fusion/D5 gate 미달 |
| `v8b_evidence_classifier_grid_v1` | v8b lesion evidence scalar classifier | 0.8942 | 0.56 | 0.7639 | 0.8674 | v31 미달 |
| `v31_v8b_late_fusion_contentcrop_v1` | contentcrop preprocessing late fusion | 0.9280 | 0.23 | 0.8152 | 0.9106 | active 미달 |
| `v31_v8b_late_fusion_safezoom_v1` | safezoom preprocessing late fusion | 0.9295 | 0.05 | 0.8166 | 0.8999 | contentcrop 소폭 상회, active 미달 |
| `v31_v8b_late_fusion_safezoom_bbretrain_v1` | safezoom + backbone retrain late fusion | 0.9204 | 0.18 | 0.8288 | 0.8921 | active/contentcrop 미달 |
| `v31_v8b_fusion_v2` | v31 score + v8b evidence late fusion, DDR 20/80 calibration | 0.9403 | 0.38 | 0.8118 | 0.9238 | previous active / rollback |
| `v31_v8b_fusion_features_hflip_v2` | extended lesion features + hflip Option A | **0.9431** | **0.3931** | **0.8124** | **0.9230** | previous active / rollback |
| `v31_v8b_fusion_quickqual_v1` | QuickQual-line v31 + v8b fusion, train-serve geometry consistency | 0.9341 | 0.06 | 0.8234 | 0.9086 | active deployment |

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
| `seg_evidence_v8b_contentcrop_v1` | IDRiD/MAPLES/TJDR/DDR_SEG | IDRiD 0.2410 / MAPLES 0.3694 / TJDR 0.3775 / DDR_SEG 0.3362 | - | MAPLES 개선, IDRiD/DDR_SEG 회귀 |
| `seg_evidence_v8b_safezoom_v1` | IDRiD/MAPLES | IDRiD 0.3085 / MAPLES 0.3655 | - | MAPLES 개선, IDRiD 회귀 |
| `seg_evidence_v8b_swa` | all test sets | IDRiD 0.3536 / MAPLES 0.1753 / TJDR 0.3541 / DDR_SEG 0.3169 | - | v8b 대비 회귀 |
| `seg_evidence_v9_gin` | all test sets | IDRiD 0.3197 / MAPLES 0.1705 / TJDR 0.3471 / DDR_SEG 0.3073 | - | v8b 대비 회귀 |
| `seg_evidence_v10_adverin` | all test sets | IDRiD 0.3501 / MAPLES 0.1875 / TJDR 0.3370 / DDR_SEG 0.3122 | - | v8b 대비 회귀 |
| `seg_evidence_v8b_repro_seed43/44/45` | current-env deterministic repeat | mean IDRiD 0.3392 / MAPLES 0.1993 / TJDR 0.3389 / DDR_SEG 0.3069 | - | old v8b 재현 실패; mean val mDice 0.2522 vs old 0.3388 |
| `seg_evidence_v8b_repro_seed43_compat` | deterministic off control | IDRiD 0.2959 / MAPLES 0.1035 / TJDR 0.3041 / DDR_SEG 0.2688 | - | deterministic flag 자체가 회귀 주범은 아님 |
| `seg_evidence_v8b_repro_seed43_geometryfix` | current-env after legacy geometry fix | IDRiD 0.3867 / MAPLES 0.2772 / TJDR 0.3923 / DDR_SEG 0.3889 | IDRiD 0.3698 / MAPLES 0.2090 / TJDR 0.3250 / DDR_SEG 0.2880 | old v8b 근접 복원; 재현 실패 주원인은 geometry regression |

## 4. 현재 판단 및 다음 작업

- **현재 active classifier**: `v31_v8b_fusion_quickqual_v1` composite checkpoint의 `v31_no_se_gated_quickqual_v1` classifier source.
- **현재 active evidence module**: `v31_v8b_fusion_quickqual_v1` composite checkpoint의 `seg_evidence_v8b_quickqual_v1` segmenter source. Circular-era standalone best는 `seg_evidence_v8b_ddrseg_tjdr_maplesfix`로 보존한다.
- **v31 상태**: 단독 배포가 아니라 fusion 내부 base classifier로 보존.
- **2026-05-24 safezoom/contentcrop preprocessing**: active/staging 변경 없음. safezoom은 contentcrop보다 낫지만 active fusion 미달.
- **2026-05-25 domain-overfit mitigation**: active/staging 변경 없음. classifier DG는 active fusion 미달, segmentation DG는 v8b evidence baseline 미달.
- **2026-05-26 current-env v8b reproducibility gate 정정**: 최초 seed 43/44/45 재현 실패(val mDice **0.2522 ± 0.0448**)는 환경 단독 문제가 아니라 legacy `processed/images` mask geometry가 contentcrop으로 바뀐 코드 회귀가 주원인이었다. `processed/images`를 circular geometry로 복원하고 `seg_evidence_v8b_repro_seed43_geometryfix`를 재학습한 결과 val mDice **0.3330**으로 old v8b **0.3388**에 근접했다. 따라서 old v8b baseline은 다시 유효한 비교 기준으로 사용할 수 있으나, 새 전처리 실험은 각 manifest prefix와 geometry mode를 명시적으로 고정해야 한다.
- **2026-05-26 previous active raw-live holdout 검증**: `v31_v8b_fusion_v2` runtime으로 DDR raw image 10,018장을 직접 추론했다. Missing raw 0건, XAI hard error 0건, AUROC **0.9401**, threshold **0.38**, Sens **0.8164**, Spec **0.9162**였다. 이는 formal offline holdout AUROC **0.9403**과 동등해 live preprocessing 경로가 정상 복구됐다고 판단한다.
- **2026-05-27 hflip staging 승격**: `v31_v8b_fusion_features_hflip_v2`를 동일한 raw-live holdout 10,018장으로 재검증했다. Missing raw 0건, XAI hard error 0건, AUROC **0.9427**, threshold **0.3931**, Sens **0.8176**, Spec **0.9176**였고, smoke latency는 mean **139 ms**, p95 **152 ms**였다. `configs/base.yaml`과 `artifacts/checkpoints/best.pt`를 hflip 후보로 승격했고, 이전 fusion active checkpoint는 `artifacts/checkpoints/best_pre_features_hflip_v2_20260527.pt.bak`로 백업했다.
- **2026-05-29 QuickQual-line 승격**: backend QuickQual geometry와 AI 학습/평가 manifest를 맞추기 위해 `v31_v8b_fusion_quickqual_v1`을 active로 교체했다. Formal DDR 20% calibration / 80% holdout 기준 AUROC **0.9341**, threshold **0.06**, Sens **0.8234**, Spec **0.9086**이다. 이 버전은 peak AUROC보다 train-serve consistency를 우선한 배포선이며, hflip/circular peak는 `artifacts/checkpoints/best_pre_quickqual_v1_20260529.pt.bak`로 보존한다.
- **2026-06-02 metric packaging 정정**: active version 이름으로 `artifacts/evaluations/external_test_v31_v8b_fusion_quickqual_v1_best_metrics.json` compact metric alias를 생성했다. 원본은 `artifacts/runs/99_misc/v31_v8b_late_fusion_quickqual_v1/evaluations/v31_v8b_late_fusion_quickqual_v1_metrics.json`의 `classification_domains:late_fusion` 결과다. 이로써 `service.py`의 `eval_metrics` runtime lookup도 active version 기준으로 동작한다.
- **2026-05-31 fusion complementarity 진단**: residual weighting으로 v31/v8b complementarity를 더 끌어올리는 방향은 실패했다. `calfit_none`은 AUROC만 보면 active보다 높지만 sensitivity가 낮아 별도 정책 후보로만 보관한다.
- **완료된 검증**:
  - Docker runtime에서 fusion checkpoint 로딩과 `/analyze` payload shape 최종 확인을 완료했다.
- **다음 작업**:
  - 추가 MA/HE/EX/SE 4채널 병변 마스크 데이터 확장은 Sprint 5 후속 개선으로 유지하되, FGADR는 기본 경로에서 제외한다.

## 5. 근거 파일
- `ai/docs/DEVLOG.md`
- `ai/docs/EXPERIMENT_REGISTRY.md`
- `ai/docs/AI_HANDOFF.md`
- `ai/.omc/research/phase4g_data_access_gate.json`
- `ai/.omc/research/phase4g_maples_roi_fix_result.json`
- `ai/.omc/research/phase4g_v8_ddrseg_result.json`
- `ai/artifacts/runs/09_evidence_segmentation/seg_evidence_v8b_ddrseg_tjdr_maplesfix/evaluations/`
- `ai/artifacts/runs/10_grounded_classifier/v8b_evidence_classifier_grid_v1/evaluations/v8b_evidence_classifier_grid_v1_metrics.json`
- `ai/artifacts/runs/10_grounded_classifier/v31_v8b_late_fusion_sweep_v1/evaluations/v31_v8b_late_fusion_sweep_v1_metrics.json`
- `ai/artifacts/evaluations/external_test_v31_v8b_fusion_v2_best_metrics.json`
- `ai/artifacts/evaluations/external_test_v31_v8b_fusion_quickqual_v1_best_metrics.json`
- `ai/artifacts/runs/10_grounded_classifier/v31_v8b_fusion_v2/evaluations/v31_v8b_fusion_v2_latency_probe.json`
- `ai/.omc/research/preprocessing_safezoom/final_comparison_safezoom_v1.json`
- `ai/.omc/research/preprocessing_safezoom/safezoom_aspect_stats.json`
- `ai/.omc/research/domain_overfit_mitigation_execution_summary.json`
- `ai/.omc/research/a1_v8b_mask_quality_audit.json`
- `ai/.omc/plans/fusion_complementarity_plan.md`
- `ai/.omc/research/fusion_complementarity/phase0_power_ceiling.json`
- `ai/.omc/research/fusion_complementarity/phase_c_calfit_ablation.json`
- `ai/.omc/research/v42_coral_baseline_shortcut_audit.json`
- `ai/.omc/research/v42_rsc_coral_shortcut_audit.json`
- `ai/.omc/research/v41_ampmix_shortcut_audit.json`
- `ai/.omc/research/v10_adverin_seg_gate_summary.json`
- `ai/artifacts/runs/99_misc/seg_evidence_v8b_repro_seed43/checkpoints/training_summary.json`
- `ai/artifacts/runs/99_misc/seg_evidence_v8b_repro_seed44/checkpoints/training_summary.json`
- `ai/artifacts/runs/99_misc/seg_evidence_v8b_repro_seed45/checkpoints/training_summary.json`
- `ai/artifacts/runs/99_misc/seg_evidence_v8b_repro_seed43_compat/checkpoints/training_summary.json`
- `ai/artifacts/runs/99_misc/seg_evidence_v8b_repro_seed43_geometryfix/checkpoints/training_summary.json`
- `ai/artifacts/runs/99_misc/seg_evidence_v8b_repro_seed43_geometryfix/evaluations/`
- `ai/.omc/research/v8b_repro_seed43_geometryfix_result_2026-05-26.json`
- `ai/.omc/research/deployment_live_preprocess_smoke_2026-05-26.json`
- `ai/.omc/research/deployment_live_preprocess_full_holdout_2026-05-26.json`
- `ai/.omc/research/staging_hflip_raw_live_full_holdout_2026-05-27.json`
- `ai/.omc/research/active_hflip_promotion_smoke_2026-05-27.json`

---

## 6. Sprint 5 마감 (2026-06-03)

Sprint 5의 마지막 실험 배치(2026-06-01~06-03)를 기록하고 Sprint 5를 종료한다.

### 6.1 Ben Graham serve-path 정합성 검증 (2026-06-02)
- 진단 중 "serve가 Ben Graham을 이중 적용한다"는 가설은 **테스트 오류로 철회**됐다. 저장 `processed_quickqual` 이미지에 이미 BG가 포함(`preprocess_images.py`가 항상 BG 적용)돼 있어, 이미 처리된 이미지를 serve preprocessor에 다시 넣어 double-BG가 됐던 것이다.
- 실제 serve(백엔드 geometry-only → AI BG×1)는 정합적이다. geometry-only 입력 재현에서 v31 AUROC **0.9106** / meta **0.9346**로 문서값을 재현했다. `backend/models/quickqual_wrapper.py:preprocess_fundus_image`는 bbox crop + square pad + 1024 resize만 하고 Ben Graham을 적용하지 않음을 코드로 확인했다. 계약은 `.omc/research/backend_preprocess_contract.json`.
- 단 **double-BG footgun**(이미 전처리된 이미지에 serve config 적용 시 BG 재적용)은 실재 → Problem 2 가드로 대응했다.

### 6.2 Problem 1 — v31 collinearity 제거 + v2 승격 (active 변경)
- meta-classifier가 `v31_probability`+`v31_logit`(sigmoid 종속, near-collinear)을 둘 다 사용해 계수가 분할/불안정했다(표준화 |coef|의 49.8%가 2개 feature). 3-way ablation(prob/logit/both)에서 **`v31_logit` 단일 표현 채택**: 89→88 feature, holdout AUROC **0.9360**(both 0.9341 상회).
- 배포 threshold는 calibration split에서 active 민감도(0.8234)를 타깃해 **0.08563** 선택(holdout leakage 방지). Holdout: AUROC **0.9360** / sens **0.8316** / spec **0.9070** / acc **0.8693** / F1 **0.8641** — v1 대비 sens +0.008 포함 약하게 지배(spec −0.0016).
- **`v31_v8b_fusion_quickqual_v2`로 active 승격**. `best.pt` 교체, `base.yaml` version/threshold, `external_test_v31_v8b_fusion_quickqual_v2_best_metrics.json` 생성. 롤백 `artifacts/checkpoints/best_pre_collinearity_refit_20260603.pt.bak`. backend/frontend 코드 무변경(지표는 `/deploy-metric`으로 자동 전파, 백엔드 재시작 시 반영).

### 6.3 Problem 2 — 전처리 footgun 가드
- `FundusPreprocess`에 `apply_ben_graham` 플래그(기본 True, 동작 불변) + `is_preprocessed_image_path` 헬퍼 추가, `service.predict_image_path`가 이미 전처리된(`processed*`) 입력에 BG를 재적용할 상황에서 경고한다. 회귀 테스트 추가(전체 15개 통과).

### 6.4 Problem 3 — anatomy-aware evidence (research, 미승격, evidence-based stop)
- Phase A: OD/fovea 탐지기(`drscreen/data/anatomy.py`) 구현. raw IDRiD OD median **0.14 OD경**(신뢰), fovea는 실패 꼬리 있어 confidence fallback 필요.
- Phase B: OD-anchored late-fusion 특징 코어 구현·테스트(`late_fusion_features.py`, 미배포).
- Phase C 게이트: (1) BG 이미지에서 OD median **0.21**, 85% 사용 가능(15% fallback). (2) **meta-level counterfactual 프로브** 신규 구축 — fusion meta-probability의 병변/비병변 스왑 민감도 측정.
- **핵심 음성 결과**: 활성 quickqual base/fusion은 matched 프로브에서 **이미 강하게 lesion-grounded**(matched_nonlesion/lesion: base **0.041**, meta **0.046**, shortcut_signal false). 인용돼 온 D7 1.48x는 **circular `v31_no_se_gated` proxy**였고 활성 quickqual에는 해당하지 않음이 확인됐다. 고칠 grounding 결함이 없어 **anatomy refit/serve 배선은 정당성이 없어 중단**. meta-probe는 향후 grounding 모니터링 도구로 보존한다.

### 6.5 프론트엔드 지표 계약
- `/deploy-metric`(`backend/main.py`)은 startup에 `_session.eval_metrics`(활성 버전 compact JSON) + decision_threshold만 캐싱해 내려보낸다. 프론트 `ReportMetrics`에는 **모델 버전 라벨이 없고** metric만 표시한다. v2 지표는 백엔드 재시작 시 자동 반영된다.

### Sprint 5 종료 상태
- **Active deployment: `v31_v8b_fusion_quickqual_v2`** (logit-only refit, threshold 0.08563, holdout AUROC 0.9360 / sens 0.8316 / spec 0.9070).
- Evidence module: `seg_evidence_v8b_quickqual_v1` (불변).
- 이월 항목: 없음. anatomy 트랙은 evidence-based로 종결.

### 6.6 추가 근거 파일
- `ai/.omc/research/backend_preprocess_contract.json`
- `ai/.omc/research/quickqual_v2_logit_matched_sensitivity.json`
- `ai/.omc/research/quickqual_v2_logit_deploy_threshold.json`
- `ai/artifacts/runs/99_misc/v31_v8b_late_fusion_quickqual_v1_v31rep/evaluations/v31_v8b_late_fusion_quickqual_v1_v31rep_metrics.json`
- `ai/artifacts/evaluations/external_test_v31_v8b_fusion_quickqual_v2_best_metrics.json`
- `ai/.omc/research/anatomy_od_fovea_validation.json`
- `ai/.omc/research/anatomy_od_on_bg_validation.json`
- `ai/.omc/research/meta_counterfactual_probe_v2_baseline.json`

---
**[SPRINT 5 CLOSED — 2026-06-03. Active deployment: v31_v8b_fusion_quickqual_v2.]**
