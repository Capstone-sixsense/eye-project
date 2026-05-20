# SPRINT 4 개발 요약

## 1. 개요
- **프로젝트 명**: eye-project (AI 파트: drscreen)
- **목표**: Sprint 3에서 남은 XAI 신뢰성 문제를 제품 기능 기준으로 재검토하고, 병변 위치 설명을 분류기 CAM이 아닌 병변 evidence 구조로 전환할 수 있는지 검증
- **기간**: SPRINT 4 주요 실험 이력 (2026.05.07 ~ 2026.05.20, v31 active 유지 기준 마감)
- **대상 범위**: v30~v39 classifier-routing/decoder-alignment 실험, Phase 4-E/F/G shortcut-free classifier 및 lesion segmentation evidence 실험, TJDR 통합, mask-geometry 보정

> 주의: Sprint 4 종료 기준 active deployment는 `v31_no_se_gated`다. 이후 실험 중 `v37b`, `v37b_aux03`, `cbm_v1` 등 일부 run은 DDR AUROC가 v31보다 높았지만, 병변 위치 정렬 또는 MAPLES 일반화 기준을 만족하지 못해 배포하지 않는다. `artifacts/checkpoints/best.pt`는 계속 v31 checkpoint alias로 유지한다.

## 2. 주요 개발 및 성과 (Milestones)

### 2.1 v30 gated pooling 및 v31 true no-attention 배포 기준 수립
- `v30_gated_pooling`에서 block4 lesion gate를 classifier pooling 경로에 연결했다.
- v30은 DDR AUROC **0.9137**, IDRiD IoU top-20 **0.0788**을 기록해 classifier-routing 기준선이 됐다.
- `v31_no_se_gated`는 `attention_mode: none`으로 SE/ECA/Spatial attention을 제거하고 gated pooling을 유지했다.
- v31은 DDR external_test AUROC **0.9160**, optimal threshold **0.35**, IDRiD block4 Layer-CAM IoU top-20 **0.0785**, AUPRC **0.1409**를 기록해 Sprint 4 active deployment로 고정됐다.

### 2.2 per-lesion routing 실험
- `v32_lesion_seg_evidence`에서 MA/HE/EX/SE 4채널 seg head를 도입했다.
- `v33_per_lesion_routing`, `v34_calibrated_routing`, `v35_warmstart_routing`으로 per-lesion gate와 routing을 검증했다.
- v33은 AUC-IoU **0.0557**, v35는 AUPRC **0.1537**로 일부 XAI 지표가 개선됐지만 DDR AUROC가 v31보다 낮아 배포하지 않았다.
- 결론적으로 4채널 routing은 지표 일부를 올릴 수 있으나, 분류 성능과 위치 정렬을 동시에 안정화하지 못했다.

### 2.3 decoder alignment 및 CAM alignment 실험
- `v36_xai_multi`, `v37_xai_multi_maples`에서 U-Net decoder와 CAM alignment loss를 적용했다.
- v37은 DDR AUROC **0.9103**으로 최소 기준은 넘었지만 threshold가 0.15로 이동하고, IDRiD/MAPLES XAI가 v31보다 낮았다.
- `v37b_xai_unet_only`는 CAM alignment를 제거해 DDR AUROC **0.9200**까지 회복했으나 MAPLES AUPRC는 **0.0161**로 여전히 약했다.
- `v37b_aux03/04/05`와 `v39_unet_2stage`까지 확인한 결과, decoder-only 또는 aux loss 강화는 Layer-CAM을 개선하지 못했다.
- 결론: decoder와 classifier 경로가 분리된 상태에서 decoder 학습만으로 분류기 CAM을 제품용 XAI로 만들 수 없다.

### 2.4 shortcut audit 및 product XAI 정의 변경
- Occlusion/RISE 계열 perturbation attribution을 진단용으로 추가했다.
- Occlusion은 classifier probability faithfulness는 개선했지만 병변 위치 정렬은 Layer-CAM보다 낮아, 제품 XAI 후보로 보지 않았다.
- shortcut audit에서 v31 block4 feature의 domain probe macro AUROC가 **0.9681**로 높았다.
- counterfactual style swap에서도 non-lesion 영역 변화가 lesion 영역 변화보다 abnormal probability에 더 크게 작용했다.
- 결론: 현재 분류기는 병변만으로 판단한다고 보기 어렵다. UI/문서 표현은 "진단 근거"가 아니라 "별도 병변 후보 영역"으로 제한해야 한다.

### 2.5 standalone lesion segmentation evidence 실험
- `seg_evidence_v1`부터 classifier와 독립된 병변 segmentation evidence 경로를 만들었다.
- train-time image/mask spatial transform sync 문제와 offline-preprocessed image/raw-mask geometry mismatch를 확인하고 수정했다.
- `seg_evidence_v2_geomfix_retrain`은 geometry fix 후에도 IDRiD/MAPLES 저데이터 조건에서 실패했다.
- 이 결과로 단순 loss 수정이나 geometry 보정만으로는 충분하지 않고, 더 많은 병변 마스크 데이터 또는 stronger encoder가 필요하다는 결론을 얻었다.

### 2.6 grounded classifier 재설계 진단
- `v31_dfr_v1`은 last-layer reweighting으로 shortcut ratio 일부를 줄였지만 DDR AUROC가 **0.8641**로 하락했다.
- Sparse BagNet 계열은 DDR AUROC **0.6293 / 0.6552**로 크게 실패했다.
- `cbm_v1`은 DDR AUROC **0.9268**을 기록했지만 concept map localization이 낮았다. IDRiD mDice **0.0217**, MAPLES mDice **0.0046**으로 제품 evidence 기준을 만족하지 못했다.
- 결론: v31 계보 위의 작은 구조 변경만으로 shortcut-free classifier를 만들기는 어렵다.

### 2.7 TJDR 통합 및 Phase 4-G segmentation evidence
- TJDR을 `data/raw/TJDR`에 확보했고, train 448쌍 / test 113쌍의 image-mask pair 무결성을 확인했다.
- `TJDRMaskProvider`는 TJDR label `1=EX`, `2=HE`, `3=MA`, `4=SE`를 프로젝트 channel order `MA/HE/EX/SE`로 재배열한다.
- `seg_evidence_v3_tjdr`는 TJDR 추가 후 IDRiD mDice **0.2055**, union IoU **0.2209**, TJDR mDice **0.3524**, union IoU **0.3490**을 기록했다.
- 그러나 MAPLES mDice는 **0.0051**, union IoU **0.0071**로 여전히 실패했다.
- `seg_evidence_v4_deeplab_tjdr`, `seg_evidence_v5_maples_fda_tjdr`, `seg_evidence_v6_maples_finetune_tjdr`, `seg_evidence_v7_maples_only`도 MAPLES gate를 넘지 못했다.
- 결론: TJDR은 IDRiD/TJDR evidence 품질을 높였지만, MAPLES cross-domain lesion evidence 문제는 해결하지 못했다.

## 3. 주요 성능 지표

### 3.1 active 및 주요 classifier run

| 버전 | 핵심 변경 | DDR AUROC | Optimal threshold | Sensitivity@optimal | 판정 |
|---|---|---:|---:|---:|---|
| `v30_gated_pooling` | block4 lesion gate x classifier pooling | 0.9137 | 0.31 | 0.7840 | classifier-routing baseline |
| `v31_no_se_gated` | true no-attention + gated pooling | **0.9160** | **0.35** | **0.7983** | active deployment |
| `v37b_xai_unet_only` | U-Net decoder, no CAM alignment | 0.9200 | 0.27 | 0.8223 | XAI/MAPLES 미달 |
| `v37b_aux03` | aux seg loss 0.3 sweep | 0.9203 | 0.41 | 0.7813 | XAI 회귀 |
| `v31_dfr_v1` | DFR last-layer reweighting | 0.8641 | 0.05 | 0.6554 | DDR 실패 |
| `bagnet_v1_p65_r512` | Sparse BagNet local evidence | 0.6552 | 0.47 | 0.3950 | DDR/XAI 실패 |
| `cbm_v1` | concept bottleneck classifier | 0.9268 | 0.21 | 0.8354 | localization 실패 |

### 3.2 segmentation evidence run

| Run | Eval set | mDice | union IoU | 판정 |
|---|---|---:|---:|---|
| `seg_evidence_v2_geomfix_retrain` | IDRiD test | 0.0335 | 0.0603 | 저데이터 한계 |
| `seg_evidence_v2_geomfix_retrain` | MAPLES test | 0.0047 | 0.0055 | 실패 |
| `seg_evidence_v3_tjdr` | IDRiD test | 0.2055 | 0.2209 | 개선 |
| `seg_evidence_v3_tjdr` | TJDR test | 0.3524 | 0.3490 | 개선 |
| `seg_evidence_v3_tjdr` | MAPLES test | 0.0051 | 0.0071 | 실패 |
| `seg_evidence_v5_maples_fda_tjdr` | MAPLES test | 0.0114 | 0.0133 | 소폭 개선, gate 미달 |
| `seg_evidence_v6_maples_finetune_tjdr` | MAPLES test | 0.0134 | 0.0175 | 소폭 개선, gate 미달 |
| `seg_evidence_v7_maples_only` | MAPLES test | 0.0039 | 0.0056 | target-only 실패 |

## 4. 결론 및 Sprint 5 이월

- **성과**: Sprint 4는 active model을 `v31_no_se_gated`로 확정하고, XAI를 제품 기능으로 내세우기 어려운 원인을 shortcut audit과 segmentation evidence 실험으로 분리했다.
- **핵심 학습**:
  - Layer-CAM 방식 자체를 바꾸는 것만으로는 병변 위치 문제를 해결하지 못한다.
  - v31 계열은 DDR 분류 성능은 유지하지만, 병변만으로 판단하는 shortcut-free classifier라고 보기 어렵다.
  - decoder alignment, aux loss 강화, two-stage decoder, DFR, Sparse BagNet, CBM 모두 제품 기준 XAI를 만족하지 못했다.
  - TJDR 추가는 IDRiD/TJDR segmentation evidence에는 효과가 있었지만 MAPLES 일반화에는 부족했다.
  - mask-geometry 보정은 필요한 수정이었지만, 성능 실패의 유일한 원인은 아니었다.
- **Sprint 5 이월 항목**:
  - MA/HE/EX/SE 4채널 병변 마스크 데이터 확장: FGADR, DDR segmentation subset, Retinal-Lesions 접근성 및 라이선스 검토.
  - classifier logit이 병변 concept/evidence를 반드시 거치는 구조 재설계.
  - 저품질 입력 차단 정책을 backend/frontend와 합의.
  - MAPLES generalization 개선을 위한 stronger fundus/segmentation encoder 검토.

## 5. 근거 파일
- `ai/docs/DEVLOG.md`: Sprint 4 상세 변경 이력. v31 syncfix, Phase 4-E/F/G, TJDR 통합, MAPLES-only 진단 기록.
- `ai/docs/EXPERIMENT_REGISTRY.md`: run group, artifact 위치, DDR/XAI/segmentation 평가 요약.
- `ai/docs/AI_HANDOFF.md`: active deployment, open issues, Phase 4-G 이후 상태.
- `ai/.omc/research/phase4e_shortcut_audit.json`
- `ai/.omc/research/phase4f_v3_selection.json`
- `ai/.omc/research/phase4g_tjdr_partial_audit.json`
- `ai/.omc/research/phase4g_deeplab_tjdr_result.json`
- `ai/.omc/research/phase4g_maples_fda_tjdr_result.json`
- `ai/.omc/research/phase4g_maples_finetune_tjdr_result.json`
- `ai/artifacts/runs/07_lesion_evidence/v31_no_se_gated/evaluations/external_test_v31_no_se_gated_best_metrics.json`
- `ai/artifacts/runs/09_evidence_segmentation/seg_evidence_v3_tjdr/evaluations/`
- `ai/artifacts/runs/09_evidence_segmentation/seg_evidence_v7_maples_only/evaluations/`

---
**[SPRINT 4 종료 기준 정리]**
