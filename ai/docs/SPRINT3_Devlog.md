# SPRINT 3 개발 요약

## 1. 개요
- **프로젝트 명**: eye-project (AI 파트: drscreen)
- **목표**: Sprint 2에서 남은 DDR threshold bias를 완화하고, XAI 설명 이미지가 실제 병변 위치와 정렬되는지 정량 검증한 뒤, 실험 산출물의 버전관리 체계를 정리
- **기간**: SPRINT 3 주요 실험 이력 (2026.04.27 ~ 2026.05.06, v29 기준 마감)
- **대상 범위**: 전처리 파이프라인 재기준, v15_fda_a10 ~ v20_coral 도메인 일반화 후속 실험, v21_512_layercam 해상도 전환, v24~v27 병변 지도/XAI 개선 실험, v28~v29 attention ablation, artifact migration

> 주의: 이 문서는 Sprint 3 AI 파트의 v29 기준 요약이다. Sprint 3 후반 실험에서 `v28_no_attention`이 DDR 외부 성능과 XAI 위치 정렬 모두에서 더 강한 후보로 확인됐고, Sprint 3 당시 active config도 `v28_no_attention`으로 변경됐다. 현재 전체 배포 버전은 이후 Sprint 5에서 `v31_v8b_fusion_v2`로 변경되었다. 다만 현재 코드 taxonomy 기준으로 v28은 true no-attention이 아니라 `attention_mode: eca`(ECA 유지, CBAM spatial 제거)다. 배포 checkpoint path는 `artifacts/checkpoints/best.pt`로 고정하며, threshold는 각 배포 artifact의 DDR external_test optimal threshold를 기준으로 한다.

## 2. 주요 개발 및 성과 (Milestones)

### 2.1 전처리 파이프라인 재기준 및 v9 재학습
- Sprint 2 종료 기준 best였던 `v9_fda`를 현재 전처리 파이프라인 기준으로 재평가했다.
- `resize_size`, `image_size`, `preprocess_size`의 불일치와 학습/추론 전처리 경로 차이를 재검토하고, 이후 실험의 기준 입력 해상도와 전처리 정책을 명확히 정리했다.
- 재학습 결과 DDR AUROC는 유지됐으나 optimal threshold가 낮아지는 문제가 확인되어, Sprint 3의 핵심 과제를 threshold bias 완화로 설정했다.

### 2.2 focal gamma 탐색 및 threshold 보정
- `v15_fda_a10`에서는 FDA alpha를 0.10으로 높였으나 DDR AUROC가 하락하여 폐기했다.
- `v16_focal_g1`, `v17_focal_g2`, `v18_focal_g3`로 focal gamma를 순차 탐색했다.
- `v17_focal_g2`는 DDR AUROC **0.8911**, optimal threshold **0.42**를 기록하여 Sprint 3 중반 기준 best로 지정됐다.
- `v18_focal_g3`는 추가 개선을 보이지 못해 focal gamma 탐색은 γ=2.0에서 종료했다.

### 2.3 SWAD, CORAL 후속 실험 및 폐기
- `v19_swad_focal_g2`는 SWAD와 focal γ=2.0을 결합했으나 threshold가 0.06으로 후퇴하여 폐기했다.
- `v20_coral`은 source domain 간 feature 공분산 정렬을 시도했으나 DDR AUROC **0.8754**, optimal threshold **0.20**으로 `v17_focal_g2`보다 낮아 폐기했다.
- CORAL은 학습 소스 간 정렬에는 작동했으나, unseen external_test 성능 개선에는 직접 기여하지 못한 것으로 판단했다.

### 2.4 512px 해상도 전환 및 Layer-CAM 적용
- 입력 해상도 증가가 소병변 표현과 XAI 해상도에 도움이 되는지 확인하기 위해 512px 전환 실험을 수행했다.
- `v21_512_layercam`은 Layer-CAM을 기본 XAI 방식으로 적용했으나 DDR AUROC **0.8775**로 `v17_focal_g2`보다 낮았다.
- 512px 전환은 내부 검증 성능은 높였지만 DDR 외부 일반화 개선으로 이어지지 않았고, 해상도 증가만으로 XAI 위치 정렬을 해결하기 어렵다는 결론을 얻었다.

### 2.5 XAI IoU 평가 인프라 구축
- IDRiD 병변 마스크를 활용해 XAI heatmap이 실제 병변 위치와 얼마나 겹치는지 정량 평가하는 체계를 구축했다.
- `drscreen/xai/iou.py`와 `eval_xai_iou.py`를 통해 Pointing Game, top-k IoU, Pixel AUPRC, AUC-IoU, center Gaussian baseline을 계산하도록 확장했다.
- v21 Layer-CAM의 train 54장 기준 IoU top-20은 **0.0300**, Pointing Game은 **0.1111**로 낮아, 설명 이미지가 병변 위치를 안정적으로 가리킨다고 보기 어려웠다.

### 2.6 병변 지도 기반 XAI 개선 실험
- `v24_multitask`에서 auxiliary segmentation head를 추가해 IDRiD 병변 마스크를 보조 supervision으로 사용했다.
- `v24_multitask`는 internal test AUROC **0.9920**을 유지했으나 DDR external_test AUROC는 **0.8452**에 그쳤고, XAI IoU top-20도 **0.0321** 수준에 머물렀다.
- `v25_multitask_l1`은 보조 segmentation loss를 강화했으나 internal test AUROC와 XAI 지표가 모두 하락하여 폐기했다.
- v24의 seg_head 출력을 직접 heatmap으로 사용하는 방식도 Pointing Game **0.0000**으로 실패하여 폐기했다.
- `v27_mil_attention`은 분류 성능은 유지했지만 XAI 위치 정렬을 개선하지 못해 주요 관심 영역 강조 기능의 직접 해결책으로 보기 어렵다고 판단했다.

### 2.7 XAI 방법 비교 및 attention ablation
- Grad-CAM, Layer-CAM, Grad-CAM++, Integrated Gradients, Score-CAM을 동일 조건에서 비교했다.
- v24 기준 5가지 XAI 방법 모두 center Gaussian baseline을 넘지 못해, image-level supervision만으로는 병변 위치 정렬에 구조적 한계가 있음을 확인했다.
- 이후 block sweep과 attention ablation을 수행한 결과, `v28_no_attention`의 block4 Layer-CAM이 Pointing Game **0.4444**, IoU top-20 **0.0741**을 기록했다.
- `v29_with_attention`은 attention을 다시 켠 대조군으로 학습했고, DDR AUROC **0.8628**로 `v28_no_attention`보다 낮았다.
- 결론적으로 Sprint 3 후반 기준으로는 ECA+Spatial 대비 CBAM spatial 제거/legacy ECA 유지 ablation이 DDR 일반화와 XAI 위치 정렬 모두에 긍정적인 영향을 준 것으로 판단했다.

### 2.8 Sprint 3 이후 후속 참고 — v30_gated_pooling (2026.05.07)

- v30은 Sprint 3의 v29 마감 범위에는 포함하지 않고, 이후 classifier-routing 기준선으로 참고한다.
- block4 feature에서 lesion gate를 생성해 classifier pooling 경로에 곱하는 방식을 도입했다 (`aux_seg.py:91`, `_forward_gated_classifier`).
- gate는 seg_head logit을 sigmoid한 뒤 공간 평균으로 정규화하여 pooling feature map에 element-wise 곱 적용.
- `v28_no_attention` 대비 DDR AUROC **0.9137**(+0.021), Sensitivity **0.7840**(+0.036), IoU top-20 **0.0788**(+0.005), AUPRC **0.1311**(+0.006), AUC-IoU **0.0443**(+0.007) 개선. Pointing Game은 **0.3704**으로 v28(0.4444) 대비 하락.
- optimal threshold 0.31 (v28 0.45 대비 낮아짐). 분류 성능과 XAI 위치 정렬 지표의 동시 개선 확인.
- seg_head 직접 출력(IoU top-20 0.0669)은 block4 Layer-CAM(0.0788)보다 낮아 직접 heatmap 활용 미채택.
- `06_xai_classifier_routing` primary group 신규 등록 (`settings.py`).

### 2.9 버전관리 및 artifact migration
- 기존 flat 구조의 checkpoint/evaluation/XAI/log artifact를 연구 질문 기준으로 재분류했다.
- canonical 저장 위치를 `artifacts/runs/<primary_group>/<run_id>/`로 통일하고, `EXPERIMENT_REGISTRY.md`를 primary group 기준 색인 문서로 작성했다.
- `current_version.txt` 방식은 제거하고, active deployment는 `configs/base.yaml`의 `project.version`과 고정 checkpoint alias로 명시하도록 정리했다.
- `artifacts/checkpoints/best.pt`는 active deployment checkpoint alias로 고정했다. 새 배포 버전은 이 경로를 바꾸지 않고, 해당 버전의 checkpoint를 이 alias에 배치하는 방식으로 관리한다.

## 3. 주요 성능 지표

### 3.1 DDR external_test 기준 주요 모델

| 버전 | 핵심 변경 | DDR AUROC | Optimal threshold | Sensitivity@optimal | Specificity@optimal | 판정 |
|---|---|---:|---:|---:|---:|---|
| `v17_focal_g2` | FDA + focal γ=2.0 | **0.8911** | **0.42** | **0.7727** | 0.8564 | threshold 보정 기준 best |
| `v21_512_layercam` | 512px + Layer-CAM | 0.8775 | 0.54 | 0.7356 | 0.8982 | 해상도 전환 효과 부족 |
| `v24_multitask` | auxiliary segmentation loss | 0.8452 | 0.17 | 0.6742 | **0.9111** | 이전 active config, 외부 성능 약화 |
| `v28_no_attention` | legacy attention ablation (ECA 유지) | **0.8924** | **0.45** | 0.7481 | 0.9055 | Sprint 3 최종 개선 후보 |
| `v29_with_attention` | attention 대조군 | 0.8628 | 0.44 | 0.6985 | 0.8993 | attention 효과 분리용 대조군 |

### 3.2 XAI 위치 정렬 지표

| 모델 / 방법 | Split | N | Pointing Game | AUPRC | AUC-IoU | IoU top-20% | 비고 |
|---|---|---:|---:|---:|---:|---:|---|
| `v21_512_layercam` Layer-CAM | train | 54 | 0.1111 | — | — | 0.0300 | Sprint 3 XAI 기준선 |
| `v24_multitask` Layer-CAM | test | 27 | 0.0370 | 0.0390 | 0.0098 | 0.0321 | 보조 seg loss 후에도 개선 제한 |
| `v24_multitask` seg_head 직접 | test | 27 | 0.0000 | — | — | 0.0318 | 직접 heatmap 활용 폐기 |
| `v28_no_attention` block4 Layer-CAM | test | 27 | **0.4444** | **0.1253** | **0.0374** | **0.0741** | legacy attention ablation 후 위치 정렬 회복 |
| Center Gaussian baseline | test | 27 | — | 0.0526 | — | 0.0436 | v24 방법 비교 기준선 |

## 4. 결론 및 향후 과제
- **성과**: Sprint 3는 threshold bias 완화, XAI 정량 평가 체계 구축, 병변 지도 기반 개선 실험, attention ablation, artifact migration을 완료했다.
- **핵심 학습**:
  - focal γ=2.0은 DDR threshold를 0.42 수준까지 끌어올리는 데 효과적이었다.
  - 512px 전환, auxiliary segmentation head, MIL attention만으로는 XAI 위치 정렬을 안정적으로 해결하지 못했다.
  - v24 계열의 XAI는 center Gaussian baseline을 넘지 못해, 주요 관심 영역 강조 기능을 완료 처리하기 어렵다.
  - legacy attention ablation 모델인 `v28_no_attention`은 DDR AUROC와 XAI block4 위치 정렬에서 가장 강한 개선 후보로 확인됐다.
  - 버전 번호만으로 실험을 관리하기에는 한계가 있어, primary group 기반 artifact 관리가 필요하다.
- **Sprint 3 종료 당시 남은 과제**:
  - 당시에는 `v28_no_attention` active config 기준으로 추론 파이프라인과 backend 연동 검증이 필요했다. 이후 Sprint 4/5에서 v31 계열과 `v31_v8b_fusion_v2`로 배포 경로가 대체됐다.
  - 주요 관심 영역 강조 기능은 XAI 신뢰성 문제가 해소된 뒤 제품 기능으로 확정해야 한다.
  - block4 CAM, lesion-aware classifier, segmentation supervision 재설계 등 병변 위치 정렬을 직접 강화하는 구조를 추가 검토해야 한다.
  - 새 실험 시작 전 run_id, primary group, config, checkpoint, evaluation artifact 위치를 먼저 등록하는 절차를 고정해야 한다.

## 5. 근거 파일
- `ai/docs/DEVLOG.md`: Sprint 3 상세 변경 이력. v15~v29, XAI 정량 검증, attention ablation, artifact migration 기록.
- `ai/docs/EXPERIMENT_REGISTRY.md`: primary group 기준 artifact 분류, v24/v28/v29 상태, DDR external_test 및 XAI 지표 요약.
- `ai/docs/AI_HANDOFF.md`: 현재 전체 배포 상태의 source of truth. Sprint 3 당시의 active config와 checkpoint 기준은 이 문서의 역사 기록으로만 참조한다.
- `ai/configs/base.yaml`: Sprint 3 당시에는 `project.version: v28_no_attention` 기준이었으나, 현재 파일은 최신 active deployment 기준으로 변경되어 있다.
- `ai/artifacts/runs/02_domain_generalization/v17_focal_g2/evaluations/external_test_v17_focal_g2_best_metrics.json`
- `ai/artifacts/runs/03_resolution_layercam/v21_512_layercam/evaluations/external_test_v21_512_layercam_best_metrics.json`
- `ai/artifacts/runs/04_lesion_supervision/v24_multitask/evaluations/external_test_v24_multitask_best_metrics.json`
- `ai/artifacts/runs/05_xai_attention_ablation/v28_no_attention/evaluations/external_test_v28_no_attention_best_metrics.json`
- `ai/artifacts/runs/05_xai_attention_ablation/v29_with_attention/evaluations/external_test_v29_with_attention_best_metrics.json`
- `ai/artifacts/runs/04_lesion_supervision/v24_multitask/evaluations/xai_iou_v24_multitask_default_test.json`
- `ai/artifacts/runs/05_xai_attention_ablation/v28_no_attention/evaluations/xai_iou_v28_no_attention_block4_test.json`

---
**[SPRINT 3 v29 기준 정리]**
