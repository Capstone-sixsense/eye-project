# AI Handoff

This file is the single source of truth for the current state of the `eye-project/ai` module.
It serves as the primary orientation for both humans and AI agents.

- **Read this file first.**
- Use `docs/DEVLOG.md` for historical context only; it is not the current source of truth.
- If this file conflicts with code or active configs (`configs/base.yaml`), trust the code and configs.

## 1. Program Overview

`drscreen` is the AI module for single-image diabetic retinopathy (DR) screening.

- **Task**: `binary_dr_screening` (Normal vs. Abnormal).
- **Model**: EfficientNet-B5. The active deployment alias currently uses the v31_no_se_gated checkpoint (DDR AUROC 0.9160, classification-best among evaluated runs); v30_gated_pooling is the classifier-routing baseline.
- **Input**: Single fundus image (RGB).
- **Output Head**: Single logit (`num_outputs: 1`).

### Primary References
- **Active Config**: `configs/base.yaml` (all inference settings, including checkpoint path).
- **Experiment Registry**: `docs/EXPERIMENT_REGISTRY.md` (canonical grouping of existing results and artifacts).
- **Core Logic**: `drscreen/infer/service.py` (Inference), `drscreen/train/runner.py` (Training/Eval).
- **Settings**: `drscreen/settings.py` (Config merging and path resolution).

---

## 2. Current Runtime Flow

### Training / Evaluation Path
The system is currently configured for **offline-preprocessed** data to ensure consistency and speed.
- **Manifest**: `data/processed/manifest_preprocessed.csv`.
- **Data Source**: Preprocessed PNGs in `data/raw/processed/images/`.
- **Pre-processing**: Circular crop + Ben Graham normalization (applied offline via `preprocess_images.py`).
- **Live Config**: `data.use_preprocessing: false` is intentional. Setting it to `true` with the preprocessed manifest would apply Ben Graham twice.

### Inference Path
The inference session (`drscreen/infer/service.py`) follows this sequence:
1. **Input Image**: RGB conversion.
2. **Optional Live Preprocessing**: `FundusPreprocess` applies circular crop + Ben Graham normalization (enabled via `infer.use_preprocessing: true`).
3. **Eval Transform**: Resize(512) → CenterCrop(512) → ToTensor → Normalize(ImageNet stats).
4. **Model Forward**: EfficientNet-B5 produces `abnormal_probability`.
5. **Evidence/XAI**: Default mode generates Layer-CAM/Grad-CAM (`evidence_type: cam_research`). Lesion segmentation evidence mode is available for v32-style per-lesion models (`evidence_type: lesion_segmentation`), but active deployment still uses CAM research evidence. CAM failure returns `xai_error_code: "XAI_001"`; lesion evidence failure returns `xai_error_code: "XAI_002"`.
6. **Payload Assembly**: Final structured JSON for the backend.

AI training/evaluation/preprocessing does **not** run a quality filter. Inference-time QuickQual is now a separate backend task; AI payload quality fields are kept for compatibility and filled with `None` in `drscreen/infer/service.py`.

---

## 3. Current Configuration (v31_no_se_gated — Active Deployment Alias)

The system's active inference config points to **v31_no_se_gated**. `artifacts/checkpoints/best.pt` currently contains the `v31_no_se_gated` checkpoint, so the fixed deployment alias and `configs/base.yaml` are aligned.

> **Metric boundary**: v31 DDR external_test reports AUROC 0.9160 / optimal threshold 0.35 (`artifacts/runs/07_lesion_evidence/v31_no_se_gated/evaluations/external_test_v31_no_se_gated_best_metrics.json`). v31 test-split XAI block4 Layer-CAM reports PG 0.3704, AUPRC 0.1409, AUC-IoU 0.0496, IoU top-20 0.0785 (`artifacts/runs/07_lesion_evidence/v31_no_se_gated/evaluations/xai_iou_v31_no_se_gated_block4_test.json`).

### Core Settings (`configs/base.yaml`)
- `model.architecture: efficientnet_b5`
- `model.use_attention: false`
- `model.attention_mode: none`
- `model.use_aux_seg: true`
- `model.aux_seg_block: 4`
- `model.aux_seg_channels: 1`
- `model.use_gated_pooling: true`
- `model.use_ibn: false`
- `data.image_size: 512`
- `data.resize_size: 512`
- `data.train_exclude_domains: []`
- `infer.checkpoint_path: artifacts/checkpoints/best.pt`
- `infer.use_preprocessing: true`
- `infer.threshold: 0.35`
- `infer.gradcam_method: layercam`
- `infer.gradcam_target_block: 4`
- `infer.evidence_type: cam_research`

Runtime threshold policy: inference first uses the selected run artifact's `optimal_threshold` when available (`external_test_<version>_best_metrics.json`). `infer.threshold` is the deployment fallback and is kept aligned with the active artifact's DDR optimal threshold.

Deployment checkpoint policy: `infer.checkpoint_path` is intentionally fixed to `artifacts/checkpoints/best.pt`. When the active version changes, copy/promote that version's checkpoint into this fixed alias instead of changing the deployment path.

### v30_gated_pooling 기준선 설정 (`configs/v30_gated_pooling.yaml`)
- **Auxiliary segmentation**: `model.use_aux_seg: true`, `aux_seg_block: 4`
- **Gated pooling**: `model.use_gated_pooling: true` — block4 feature에서 lesion gate 생성 후 classifier pooling 경로에 곱함 (`aux_seg.py:91`)
- **Auxiliary loss**: `train.lambda_aux_seg: 0.3`
- **Loss**: Focal loss (α=0.75, γ=2.0)
- **Attention**: `model.use_attention: false`, `model.attention_mode: eca`
- **Backbone origin**: `artifacts/runs/03_resolution_layercam/v7_512_messidor_train/checkpoints/best.pt`

Attention taxonomy:
- `attention_mode: eca_spatial`: `_EcaSpatialAttn` (ECA channel + CBAM spatial).
- `attention_mode: eca`: legacy `use_attention=false` behavior; EfficientNet SE 위치에 `EcaModule`이 남아 있다.
- `attention_mode: none`: EfficientNet 내부 SE/ECA/Spatial 위치를 `IdentitySE`로 대체한다. gated pooling(`use_gated_pooling`)은 별개 메커니즘이므로 함께 켤 수 있다. "true no-attention"은 EfficientNet 내부 channel/spatial module 제거를 의미하며, aux seg 기반 lesion gate는 포함하지 않는다.

### Checkpoint Lineage

```
ImageNet pretrained EfficientNet-B5
  → SSL (APTOS+IDRiD+Messidor, 5,378장, SimCLR) → ssl/backbone_best.pt
    → supervised fine-tune → artifacts/runs/01_ssl_lineage/v4.1/checkpoints/best.pt  [val AUROC 0.9975]
      → focal fine-tune (α=0.75, γ=0.0) → artifacts/runs/01_ssl_lineage/v6_alpha_only/checkpoints/best.pt  [Messidor AUROC 0.8697]
        → Messidor 학습 편입 + DDR 외부 테스트 교체 → artifacts/runs/02_domain_generalization/v7_messidor_train/checkpoints/best.pt  [DDR AUROC 0.8725]
          → FDA (α=0.05, γ=0.0) fine-tune → artifacts/runs/02_domain_generalization/v9_fda/checkpoints/best.pt  [DDR AUROC 0.8812, thr 0.19]
          → SWAD (last_n=5) fine-tune    → artifacts/runs/02_domain_generalization/v10_swad/checkpoints/best.pt  [DDR AUROC 0.8863, thr 0.05]  (배포 보류)
          → FDA (α=0.05) + focal γ=2.0   → artifacts/runs/02_domain_generalization/v17_focal_g2/checkpoints/best.pt  [DDR AUROC 0.8911, thr 0.42]
          → 512px Messidor train backbone → artifacts/runs/03_resolution_layercam/v7_512_messidor_train/checkpoints/best.pt [DDR AUROC 0.9046]
            → 512px + Layer-CAM            → artifacts/runs/03_resolution_layercam/v21_512_layercam/checkpoints/best.pt [DDR AUROC 0.8775, XAI train top-20 IoU 0.030]
            → auxiliary segmentation loss  → artifacts/runs/04_lesion_supervision/v24_multitask/checkpoints/best.pt [test AUROC 0.9920, DDR AUROC 0.8452, XAI test top-20 IoU 0.032]
            → attention ablation            → artifacts/runs/05_xai_attention_ablation/v28_no_attention/checkpoints/best.pt [test AUROC 0.9923, DDR AUROC 0.8924, XAI block4 top-20 IoU 0.074]
            → gated pooling (block4 lesion gate × classifier path) → artifacts/runs/06_xai_classifier_routing/v30_gated_pooling/checkpoints/best.pt [DDR AUROC 0.9137, XAI block4 top-20 IoU 0.0788] (classifier-routing baseline)
            → true no-attention + Dice+BCE seg loss → artifacts/runs/07_lesion_evidence/v31_no_se_gated/checkpoints/best.pt [val AUROC 0.9993, DDR AUROC 0.9160, test XAI AUPRC 0.1409] ← ACTIVE DEPLOYMENT ALIAS
            → per-lesion 4ch seg + Dice+BCE → artifacts/runs/07_lesion_evidence/v32_lesion_seg_evidence/checkpoints/best.pt [val AUROC 0.9992, train XAI AUPRC 0.0538 (seg_head)]
            → per-lesion weighted routing → artifacts/runs/07_lesion_evidence/v33_per_lesion_routing/checkpoints/best.pt [DDR AUROC 0.9131, test XAI AUC-IoU 0.0557]
            → calibrated per-lesion routing → artifacts/runs/07_lesion_evidence/v34_calibrated_routing/checkpoints/best.pt [DDR AUROC 0.9129, test XAI PG 0.5185]
            → v31 warmstart per-lesion routing → artifacts/runs/07_lesion_evidence/v35_warmstart_routing/checkpoints/best.pt [DDR AUROC 0.9081, test XAI AUPRC 0.1537]
```

`v31_no_se_gated` and `v32_lesion_seg_evidence` are not checkpoint continuations from v30. Their configs load `artifacts/runs/03_resolution_layercam/v7_512_messidor_train/checkpoints/best.pt` as the pretrained backbone, then train their own no-attention/gated-pooling heads.

**v4.1 SSL 계보 주의**: v4.1은 SSL backbone(`ssl/backbone_best.pt`, APTOS+IDRiD+Messidor 5,378장 비레이블 SimCLR 사전학습)에서 시작한 supervised fine-tune이다. DEVLOG에 한 차례 "SSL 없음"으로 오기재됐으나 2026.04.15 재정정에서 SSL 관여가 확인됐다.

---

## 4. Checkpoint Policy

### Storage
Completed run artifacts are stored under `artifacts/runs/<primary_group>/<version>/`.
- `checkpoints/`: model checkpoints and training summary.
- `evaluations/`: classification metrics and XAI JSON outputs.
- `logs/`: historical shell/log captures when available.

Checkpoint files:
- `best.pt`: Highest validation AUROC among epochs meeting the sensitivity threshold.
- `last.pt`: The final epoch of the training run.

### Deployment Checkpoint
현재 active checkpoint는 `configs/base.yaml`의 `infer.checkpoint_path`가 지정하며, 이 경로는 `artifacts/checkpoints/best.pt`로 고정한다.
`artifacts/checkpoints/best.pt`는 active deployment alias다. `configs/base.yaml`의 `project.version`/model flags와 이 파일의 실제 checkpoint 내용이 반드시 일치해야 한다.
`train.checkpoint_dir: artifacts/runs`는 version 없는 fallback용이다. Versioned training은 `drscreen/settings.py`의 primary group resolver가 `artifacts/runs/<primary_group>/<version>/checkpoints/`로 저장한다.
`train.global_best_checkpoint_path: artifacts/checkpoints/best.pt`도 같은 고정 alias를 사용하므로, 배포 전에는 의도한 버전의 checkpoint가 이 파일에 들어 있는지 확인한다.

새 버전을 배포할 때는 기존 version shift 형식을 유지한다: `configs/base.yaml`의 `project.version`, model flags, 필요한 `infer.threshold`/`infer.gradcam_method`를 새 버전 기준으로 바꾸고, 해당 버전 checkpoint를 `artifacts/checkpoints/best.pt`에 배치한다. `infer.checkpoint_path` 자체는 바꾸지 않는다. Threshold는 각 배포 artifact의 DDR `external_test` optimal threshold를 기준으로 한다.

### Selection Logic
- A checkpoint must meet `train.min_checkpoint_sensitivity` (default: 0.80).
- Among candidates, the one with the maximum validation AUROC is selected.
- `promotion_candidate` is set in `training_summary.json` when a new run's `best_val_auroc` exceeds the previous global best; promotion remains manual.
- Config merging is handled by `build_effective_checkpoint_config()` in `settings.py`, which ensures the runtime environment respects the model architecture and labels saved within the checkpoint.

---

## 5. Payload Contract (Backend Integration)

The `InferenceSession` produces a `prediction.payload` consumed by the `eye-project` backend.

### Core Fields
- `predicted_index`: `0` (Normal) or `1` (Abnormal).
- `predicted_label`: String representation of the prediction.
- `abnormal_probability`: Float (0.0 to 1.0).
- `xai_error_code`: `"XAI_001"` if Grad-CAM generation fails, otherwise `null`.
- `evidence_type`: `"cam_research"` or `"lesion_segmentation"`.
- `lesion_summary`: Per-lesion area ratio / presence score summary when lesion evidence is active, otherwise `null`.
- `evidence_warning`: Evidence quality/status warning such as `"LESION_EVIDENCE_LOW_CONFIDENCE"`, otherwise `null`.

### Artifact Paths
- `checkpoint_path`: Path to the model used.
- `prediction_path`: Path to the saved result JSON.
- `heatmap_path`: Path to the saved evidence overlay PNG. For current CAM runs, this is a Grad-CAM/Layer-CAM overlay.
- `lesion_map_path`: Same overlay path when `evidence_type: lesion_segmentation`, otherwise `null`.

---

## 6. Known Gaps & Next Steps

### Open Issues

- **Active deployment**: Active config points to v31_no_se_gated. `artifacts/checkpoints/best.pt` contains the v31_no_se_gated checkpoint. DDR AUROC 0.9160 is the classification-best result among evaluated runs (v30 0.9137, v28 0.8924), with optimal threshold 0.35, Sens 0.798, Spec 0.868.
- **Threshold**: Active `infer.threshold` is 0.35, aligned with the v31 DDR external_test optimal threshold.
- **v29 status**: XAI artifacts not generated. DDR AUROC 0.8629 (< v28 0.8924). ECA+CBAM spatial attention hurts both classification and XAI — direction discarded.
- **v30 XAI (test, baseline)**: block4 Layer-CAM — AUPRC 0.1311, AUC-IoU 0.0443, IoU top-20 0.0788, PG 0.3704. seg_head 직접 출력(0.0669)은 Layer-CAM 대비 열위. v30은 classifier-routing baseline으로 유지한다.
- **v31 active**: val AUROC 0.9993 (epoch 5, early stop 8). `attention_mode: none` + Dice+BCE seg loss. DDR external_test AUROC **0.9160** (classification-best; optimal thr 0.35, Sens 0.798, Spec 0.868). train XAI: AUPRC 0.1174, AUC-IoU 0.0491, IoU top-20 0.0601, PG 0.3333. test XAI block4 Layer-CAM: AUPRC 0.1409, AUC-IoU 0.0496, IoU top-20 0.0785, PG 0.3704. Active deployment.
- **v32 completed (train XAI only)**: val AUROC 0.9992 (epoch 3). 4-channel per-lesion seg head + Dice+BCE. train XAI (seg_head): AUPRC 0.0538, AUC-IoU 0.0208. 현재 코드의 gated classifier는 4채널이면 per-lesion sigmoid + softmax weighted sum을 사용하지만, v32 artifact는 seg_head train 평가만 있어 제품 XAI 후보로 보지 않는다. v33+가 per-lesion routing 계열을 대체 평가한다.
- **v33 completed**: val AUROC 0.9980 (epoch 3, early stop 7). per-lesion 독립 sigmoid + softmax weighted sum gate. DDR AUROC 0.9131 (v31 0.9160 대비 -0.003). test XAI: AUPRC 0.1478, AUC-IoU **0.0557**, PG 0.4074 — AUC-IoU 기준 최고. 분류 소폭 하락으로 배포 미승격; v31 유지.
- **v35 completed**: val AUROC 0.9992 (epoch 9). v31 warmstart + 4ch per-lesion routing. DDR external_test AUROC 0.9081, optimal thr 0.18, Sens 0.793, Spec 0.874. test XAI: AUPRC **0.1537** (AUPRC 기준 최고), AUC-IoU 0.0525, IoU top-20 0.0796, PG 0.4074. Warmstart도 DDR 회귀를 해소하지 못해 배포 미승격.
- **v27 MIL attention (discarded)**: train XAI AUC-IoU 0.0119 — random baseline(0.0260) 이하. MIL attention 방향 폐기.
- **Phase-0 gate**: test split 기준 전 모델 FAIL. center_gaussian+2σ=0.1089 임계값이 현실적으로 달성 불가 수준. gate 기준 재조정 필요.
- **IDRiD contamination**: XAI eval(`A. Segmentation`)과 분류 학습(`B. Disease Grading`)은 파일은 다르나 동일 환자 포함(patient-level overlap). file-level contamination 없음. 상세 내용은 아래 섹션 참조.
- **Phase-0 gate 기준 재정의**: 기존 2σ threshold는 모든 모델이 달성 불가 수준임을 확인. `gate_sigma` 파라미터화 완료(`eval_xai_iou.py --gate-sigma`). 현재 test split 기준: 2σ=0.1089 (v33 best=0.0557, FAIL), 1σ=0.0728 (v33 FAIL), ~0.5σ=0.0547 (v33 barely PASS). 2σ는 aspirational 기준으로 유지, 현실적 진척도 측정은 1σ 또는 절대값 0.05 사용 권장.
- **DiceBCELoss**: `drscreen/train/loss.py`에 구현 완료. `engine.py:196-209`에 wiring. `seg_loss_type: dice_bce` config로 활성화.
- **v32 구조 상태**: v32 artifact는 4채널 seg_head train 평가만 존재하고 XAI 수치가 낮다. 현재 코드(`drscreen/models/aux_seg.py`)의 4채널 gated classifier는 per-lesion sigmoid + softmax weighted sum을 사용한다. 단, 단일 evidence map 생성용 `predict_seg_union()`은 아직 `amax(dim=1)` union을 사용한다.
- **MAPLES-DR**: 확보 완료. `data/raw/MAPLES-DR/AdditionalData/` (dataset_record.yaml, annotations/ 포함). train 138장 / test 60장, 12종 biomarker 마스크. `MAPLESMaskProvider` 구현 완료 (`drscreen/data/mask_providers.py`)는 MA/HE/EX/CWS 4채널 pathology mask 로딩 범위다. 현재 `eval_xai_iou.py`에는 `--mask-provider maples` 옵션과 anatomy/lesion attribution ratio 지표가 없으므로, Phase 1 anatomy audit는 데이터/로더 기반만 준비된 상태이며 평가 CLI와 지표 구현이 추가로 필요하다.
- **Unit Tests**: `eye-project/ai` 내 `tests/` 디렉토리 없음.
- **Quality / QuickQual**: AI no longer performs quality filtering. QuickQual belongs to a separate backend task; AI payload quality fields are compatibility placeholders filled with `None`.

---

## 7. IDRiD XAI Eval Contamination

### 오염 구조

IDRiD 데이터셋은 두 개의 독립 서브셋으로 구성된다:

- **A. Segmentation** (54 train / 27 test): 병변 픽셀 마스크(MA/HE/EX/SE) 제공. XAI eval에 사용.
- **B. Disease Grading** (413 train / 103 test): 분류 레이블 제공. 모델 학습에 사용.

두 서브셋은 **파일명이 다르므로 file-level 오염은 없다**. 그러나 동일 환자의 이미지가 두 서브셋에 모두 포함될 수 있어 **patient-level overlap이 존재**한다. IDRiD 논문(Porwal et al., 2020)에서 patient-level split 여부를 명시하지 않음.

### 실질적 영향

- 분류 학습 시 `A. Segmentation` 이미지의 환자 일부가 `B. Disease Grading`의 학습 데이터에 포함될 수 있다.
- 이 경우 XAI eval에서 모델이 해당 이미지를 "기억"하여 localization 수치가 과대평가될 가능성이 있다.
- 단, 현재 모델들의 XAI 수치가 모두 낮은 수준(v35 AUPRC 0.1537, v33 AUC-IoU 0.0557)이므로 오염에 의한 인플레이션이 아닌 실질적인 한계로 판단.

### 대응

- 현재: patient-level split 검증 툴링 없음. XAI eval 결과를 "참고 수치"로만 활용.
- 이상적: MAPLES-DR 데이터셋 사용 시 완전히 분리된 코호트에서 XAI eval 가능.

---

## 8. Phase-0 XAI Gate

### 정의

Phase-0 gate는 모델의 병변 localization 능력이 **center_gaussian baseline을 유의미하게 초과**하는지를 판별하는 threshold다.

```
gate = model_AUC_IoU > center_gaussian_mean + N × center_gaussian_std
```

- `N=2` (기존): aspirational 기준. test split 기준 threshold=0.1089. **전 모델 FAIL.**
- `N=1` (완화): progress gate 후보. test split 기준 threshold=0.0728. **전 모델 FAIL (v33 best=0.0557).**
- `N=0.5` (완화): test split 기준 threshold=0.0547. v33(0.0557) barely PASS.

### 현재 결론

- 2σ 기준은 aspirational 목표로만 유지 (단기 달성 불가).
- 단기 진척도 측정은 `--gate-sigma 1.0` 또는 절대값 0.05 기준 사용.
- `eval_xai_iou.py --gate-sigma N`으로 실행 시 동적으로 변경 가능.

### 근거

2σ 기준은 "통계적으로 유의미하게 random baseline을 상회"하는 표준 통계적 해석을 따른 것이다. 그러나 현재 모델 용량(EfficientNet-B5, IDRiD 54장 XAI 감독) 대비 너무 엄격함이 확인됐다. 완화는 목표를 낮추는 것이 아니라 **중간 마일스톤**을 추가하는 것으로 해석한다.

---

## 9. Historical Sprint Notes

### Sprint 2 실험 이력 (2026.04.09~2026.04.17)

#### 완료: RETFound backbone 교체 실험 → 폐기 (2026.04.10)

| 실험 | 외부 테스트셋 | AUROC | 비고 |
|---|---|---|---|
| v6_alpha_only (당시 베스트) | Messidor | 0.8697 | EfficientNet-B5, Focal |
| retfound_v1 (BCE) | Messidor | 0.6722 | RETFound 단독 fine-tuning |
| retfound_v2 (Focal α=0.75) | Messidor | 0.6611 | focal loss 효과 없음 |

RETFound 관련 코드 전체 `archive/retfound/`로 이동 완료.

#### 완료: v4/v4.1 체크포인트 계보 정정 (2026.04.11, 재정정 2026.04.15)

`v4/best.pt`와 `artifacts/runs/01_ssl_lineage/v4.1/checkpoints/best.pt`가 동일한 학습 실행임을 확인 (best_val_auroc / best_epoch 완전 일치). 초기에는 "SSL 없음"으로 오기재됐으나, 2026.04.15 재정정에서 `artifacts/runs/01_ssl_lineage/v4.1/checkpoints/best.pt`의 `config.train.pretrained_backbone_path: artifacts/ssl/backbone_best.pt` 확인 — SSL backbone(APTOS+IDRiD+Messidor 5,378장 SimCLR) 기반 supervised fine-tune임이 확정됐다.

#### 완료: SSL 오염 가능성 검증 — v4b_alpha_only (2026.04.11~2026.04.13)

| 모델 | SSL | Loss | val AUROC | Messidor AUROC |
|---|---|---|---|---|
| v6_alpha_only | 없음 | Focal α=0.75 | 0.9990 | **0.8697** |
| v4b_alpha_only | SSL (Messidor 제외, 4178장) | Focal α=0.75 | 0.9973 | **0.7262** |

결론: Messidor를 SSL에서 제외하면 AUROC 0.8697 → 0.7262로 하락. SSL 오염 효과 실재. 외부 일반화 개선이 핵심 과제.

#### 완료: Domain Generalization 실험 시리즈 (v7~v14, DDR 외부 테스트)

Sprint 3부터 외부 테스트셋이 Messidor(1,200장) → DDR(12,522장)로 교체됨.

| 버전 | 핵심 기법 | DDR AUROC | optimal thr | 비고 |
|---|---|---|---|---|
| v7_messidor_train | Messidor 학습 편입 | 0.8725 | 0.09 | v9/v10 backbone 기반 |
| v8_mixstyle | MixStyle | 0.8371 | 0.31 | AUROC 회귀, 폐기 |
| v9_fda | FDA (α=0.05, γ=0.0) | 0.8825* | 0.06* | Sprint 3 재학습 수치. threshold 악화 |
| v10_swad | SWAD (last_n=5) | 0.8863† | 0.05† | AUROC 우위, threshold 악화로 배포 보류 |
| v11_fda_swad | FDA + SWAD | 0.8539 | 0.31 | 두 기법 손실 지형 충돌 |
| v12_fda_imagenet | FDA + ImageNet 초기화 | 0.8498 | 0.05 | v7 backbone 없이 FDA 단독 실패 |
| v13_fda_swad | FDA + SWAD BN 수정 재시도 | 0.8436 | 0.05 | 근본 충돌 미해소, 악화 |
| v14_ibn | IBN-Net (blocks 0-2) | 0.8445 | 0.08 | IN이 진단 특징 희석 |
| v15_fda_a10 | FDA α=0.10 | 0.8579 | 0.05 | α 과도, 병변 구조 왜곡, 폐기 |
| v16_focal_g1 | FDA + focal γ=1.0 | 0.8738 | 0.18 | threshold 개선, AUROC 소폭 하락 |
| **v17_focal_g2** | **FDA + focal γ=2.0** | **0.8911** | **0.42** | **DEPLOYMENT BEST** ✓ |
| v18_focal_g3 | FDA + focal γ=3.0 | 0.8747 | 0.29 | γ 과도, 역전, 폐기 |

*v9_fda: Sprint 3에서 새 전처리로 재학습. Sprint 2 원본 체크포인트 소실.
†v10_swad: BN 재보정 버그 수정 후 재평가 수치.

**결론**: focal γ=2.0이 hard example focusing을 통해 DDR threshold bias를 0.06→0.42로 대폭 개선하면서 AUROC도 0.8911로 신기록. **v17_focal_g2가 AUROC·threshold 모든 기준 최우수로 배포 지정.**

†v10_swad: SWAD BN 재보정 버그(`model.train()` → stochastic depth 활성화) 수정 후 2026.04.21 재평가. 수정 전 수치는 AUROC 0.8925, threshold 0.08.

#### Sprint 3 완료 요약 (2026.04.27)

- **달성**: focal γ=2.0으로 threshold 0.06→0.42, AUROC 0.8825→0.8911
- **배포 best**: v17_focal_g2 (`artifacts/runs/02_domain_generalization/v17_focal_g2/checkpoints/best.pt`)
- **Sprint 4 후보**: CORAL (2차 통계량 정렬, 미구현), DDR 학습 편입 (별도 외부 테스트셋 필요)
- **데이터 전략**: DDR을 학습 도메인으로 포함하는 방안 검토 (별도 외부 테스트셋 필요)

---

## 10. Operational Commands

### Training
```bash
python -m drscreen.cli.train --config configs/v9_fda.yaml
```

### Evaluation (DDR external test)
```bash
python -m drscreen.cli.evaluate --config configs/v9_fda.yaml --split external_test
```

### Inference
```bash
python -m drscreen.cli.infer --config configs/v9_fda.yaml --image path/to/image.png
```
