# AI Handoff

This file is the single source of truth for the current state of the `eye-project/ai` module.
It serves as the primary orientation for both humans and AI agents.

- **Read this file first.**
- Use `docs/DEVLOG.md` for historical context only; it is not the current source of truth.
- If this file conflicts with code or active configs (`configs/base.yaml`), trust the code and configs.

## 1. Program Overview

`drscreen` is the AI module for single-image diabetic retinopathy (DR) screening.

- **Task**: `binary_dr_screening` (Normal vs. Abnormal).
- **Model**: Active deployment is `v31_v8b_fusion_quickqual_v2` (promoted 2026-06-03; **v31 collinearity refit** of the QuickQual line). Same components as v1 — base classifier `v31_no_se_gated_quickqual_v1`, segmenter `seg_evidence_v8b_quickqual_v1` — but the numeric StandardScaler+LogReg meta-classifier now uses a **single v31 log-odds feature (`v31_logit`)** plus the v8b lesion features (**88 features total**; the redundant `v31_probability` column was dropped because `v31_probability=sigmoid(v31_logit)` is near-collinear and inflated/destabilized the coefficients). Deployment threshold **0.0856** is selected on the DDR 20% calibration split to match the previous active sensitivity. Formal DDR 80% holdout metrics: **AUROC 0.9360, threshold 0.0856, sensitivity 0.8316, specificity 0.9070** (`classification_domains:late_fusion:v31_logit`). At this matched-sensitivity operating point v2 weakly dominates the previous v1 (AUROC +0.0019, sensitivity +0.008, specificity −0.0016, accuracy +0.0033, F1 +0.0041). The 2026-06-03 meta-level counterfactual probe shows the active quickqual base classifier AND the fusion meta-probability are already lesion-grounded on the matched probe (matched_nonlesion/lesion 0.041 base / 0.046 meta, shortcut_signal false; the older D7 1.48x was the circular `v31_no_se_gated` proxy, not this model). D5 domain separability remains a separate concern, so product wording should still describe v8b output as separately detected lesion candidate areas rather than causal proof.
- **Preprocessing geometry**: training uses `data.preprocess_mode: quickqual` (replicates backend `preprocess_fundus_image`: RGB-mean>15 bbox, +20px buffer, square-pad). Serve uses `infer.preprocess_mode: none` (backend already applied QuickQual; AI inference only resizes 512 + Ben Graham). **Footgun**: standalone `drscreen.cli.infer` on a RAW image skips cropping — use a `preprocess_mode: quickqual` config for raw-input inference. backend/frontend code unchanged (backend already emits 1024; frontend consumes images by URL).
- **IDRiD overlay caveat**: IDRiD seg-test mDice is NOT a reliable evidence metric. It collapsed across all QuickQual variants (0.10–0.22) and is not explained by fill ratio; combined with the §7 IDRiD seg/grading patient-level contamination, the circular-era 0.4151 is suspected inflation. Judge evidence quality by MAPLES/TJDR/DDR_SEG instead.
- **Previous active / rollback**: `v31_v8b_fusion_quickqual_v1` (QuickQual line, two-feature `v31_probability`+`v31_logit`, AUROC 0.9341, threshold 0.06) is the immediate rollback baseline, backed up at `artifacts/checkpoints/best_pre_collinearity_refit_20260603.pt.bak`. To roll back, copy that `.bak` into `artifacts/checkpoints/best.pt` and restore `configs/base.yaml` (`project.version: v31_v8b_fusion_quickqual_v1`, `infer.threshold: 0.06`). Deeper fallbacks: `v31_v8b_fusion_features_hflip_v2` (circular, AUROC 0.9431) at `best_pre_quickqual_v1_20260529.pt.bak`, and `v31_v8b_fusion_v2` (circular) at `best_pre_features_hflip_v2_20260527.pt.bak` (circular rollback also needs `infer.preprocess_mode: circular`, `threshold: 0.3931`, `tta_mode: hflip`, `data.manifest_path: manifest_preprocessed.csv`).
- **Latest domain-generalization mitigation result**: The 2026-05-25 domain-overfit mitigation run did not change the active deployment or staging candidate. Classifier CORAL/RSC+CORAL slightly improved over the v31 base classifier but stayed far below active fusion and still failed domain-shortcut probes; segmentation SWA/GIN/AdverIN all regressed against the v8b evidence baseline. The initial 2026-05-26 B-G0 current-env v8b reproducibility failure was later traced to a legacy `processed/images` geometry regression, not environment alone: after restoring circular geometry, `seg_evidence_v8b_repro_seed43_geometryfix` recovered best val mDice 0.3330 versus old v8b 0.3388.
- **Latest preprocessing result**: The 2026-05-24 safezoom/contentcrop preprocessing experiment did not change deployment. Safezoom improved foreground/aspect preservation and slightly beat contentcrop late fusion, but its best late-fusion DDR holdout AUROC 0.9295 stayed below the then-active circular fusion AUROC 0.9403. Current active is the later QuickQual-line `v31_v8b_fusion_quickqual_v1`.
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
- **Active runtime manifest**: `data/processed/manifest_preprocessed_quickqual.csv` in `configs/base.yaml`.
- **Fusion/evidence evaluation manifest**: `data/processed/manifest_with_maples_tjdr_ddrseg_quickqual_preprocessed.csv`.
- **Data Source**: QuickQual-geometry preprocessed PNGs.
- **Pre-processing**: backend-compatible QuickQual crop/square-pad geometry + Ben Graham normalization is applied offline for training/evaluation.
- **Live Config**: `data.use_preprocessing: false` is intentional for offline-preprocessed manifests. Setting it to `true` would apply the image preprocessing twice.
- **Mask geometry**: Pixel-mask providers align raw lesion masks to the manifest image geometry. Masks receive the same crop/pad/resize geometry only; Ben Graham photometric normalization is not applied to masks.
- **Mask-supervised classifier training**: When `train.lambda_aux_seg > 0`, classifier/multitask training now uses synchronized image+mask transforms via `SegmentationManifestDataset`/`SegmentationFDAManifestDataset`. This keeps random crop/flip/rotation identical for the image and its lesion mask during aux_seg loss computation.

### Inference Path
The inference session (`drscreen/infer/service.py`) follows this sequence:
1. **Input Image**: RGB conversion.
2. **Optional Live Preprocessing**: active deployment sets `infer.use_preprocessing: true` but `infer.preprocess_mode: none`. This means the AI runtime expects the backend QuickQual task to have already produced the cropped/square-padded image. Standalone raw-image CLI inference must use a raw-input config with `preprocess_mode: quickqual`.
3. **Eval Transform**: Resize(512) → CenterCrop(512) → ToTensor → Normalize(ImageNet stats).
4. **Model Forward**: `V31V8bFusion` computes the v31 probability/logit and v8b lesion features, then applies the numeric meta-classifier to produce final `abnormal_probability`.
5. **Evidence/XAI**: Active deployment uses v8b lesion segmentation evidence (`evidence_type: lesion_segmentation`). CAM research evidence remains available for older configs. CAM failure returns `xai_error_code: "XAI_001"`; lesion evidence failure returns `xai_error_code: "XAI_002"`; grounded-classifier evidence failure returns `xai_error_code: "XAI_003"`.
6. **Payload Assembly**: Final structured JSON for the backend.

AI training/evaluation/preprocessing does **not** run a quality filter. Inference-time QuickQual is now a separate backend task; AI payload quality fields are kept for compatibility and filled with `None` in `drscreen/infer/service.py`.

---

## 3. Current Configuration (v31_v8b_fusion_quickqual_v2 — Active Deployment Alias)

The system's active inference config points to **v31_v8b_fusion_quickqual_v2** (v31 collinearity refit: single `v31_logit` feature, 88-feature meta-classifier). `artifacts/checkpoints/best.pt` contains the composite QuickQual-line v31 classifier + QuickQual-line v8b segmenter + numeric StandardScaler/LogReg meta-classifier checkpoint, so the fixed deployment alias and `configs/base.yaml` are aligned.

> **Metric boundary**: v31_v8b_fusion_quickqual_v2 uses the formal DDR external_test 20% calibration / 80% holdout policy. The deployment threshold **0.08563088401268978** is selected on the calibration split to match the previous active sensitivity (0.8234). Holdout metrics at that threshold are **AUROC 0.9360 / sensitivity 0.8316 / specificity 0.9070 / accuracy 0.8693 / F1 0.8641**. The runtime compact metrics alias is `artifacts/evaluations/external_test_v31_v8b_fusion_quickqual_v2_best_metrics.json`; its source is `artifacts/runs/99_misc/v31_v8b_late_fusion_quickqual_v1_v31rep/evaluations/v31_v8b_late_fusion_quickqual_v1_v31rep_metrics.json`, key `classification_domains:late_fusion:v31_logit`. `artifacts/checkpoints/best.pt` metadata confirms `architecture=v31_v8b_fusion`, `optimal_threshold=0.08563088401268978`, `feature_schema[0]=v31_logit` (88 features), classifier source `artifacts/runs/99_misc/v31_no_se_gated_quickqual_v1/checkpoints/best.pt`, and segmenter source `artifacts/runs/99_misc/seg_evidence_v8b_quickqual_v1/checkpoints/best.pt`. Immediate rollback is `v31_v8b_fusion_quickqual_v1` (two-feature, AUROC 0.9341, threshold 0.06), backed up at `artifacts/checkpoints/best_pre_collinearity_refit_20260603.pt.bak`.

### Core Settings (`configs/base.yaml`)
- `model.architecture: v31_v8b_fusion`
- `model.use_attention: false`
- `model.attention_mode: none`
- `model.use_aux_seg: true`
- `model.aux_seg_block: 4`
- `model.aux_seg_channels: 1`
- `model.use_gated_pooling: true`
- `model.use_ibn: false`
- `model.segmenter_encoder: resnet50`
- `model.segmenter_out_channels: 4`
- `model.segmenter_decoder_channels: [256, 128, 64, 32]`
- `data.image_size: 512`
- `data.resize_size: 512`
- `data.manifest_path: data/processed/manifest_preprocessed_quickqual.csv`
- `data.train_exclude_domains: []`
- `infer.checkpoint_path: artifacts/checkpoints/best.pt`
- `infer.use_preprocessing: true`
- `infer.preprocess_mode: none`
- `infer.threshold: 0.08563088401268978`
- `infer.amp: false`
- `infer.tta_mode: none`
- `infer.use_meta_classifier: true`
- `infer.gradcam_method: layercam`
- `infer.gradcam_target_block: 4`
- `infer.evidence_type: lesion_segmentation`
- `infer.lesion_threshold: 0.5`

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
            → true no-attention + Dice+BCE seg loss → artifacts/runs/07_lesion_evidence/v31_no_se_gated/checkpoints/best.pt [val AUROC 0.9993, DDR AUROC 0.9160, test XAI AUPRC 0.1409] (v31 base classifier)
              ↳ synchronized mask-transform rerun → artifacts/runs/07_lesion_evidence/v31_syncfix_rerun/checkpoints/best.pt [DDR AUROC 0.9082, IDRiD IoU20 0.0600, MAPLES AUPRC 0.0125] (not promoted)
              ↳ synchronized mask-transform seed repeats → artifacts/runs/07_lesion_evidence/v31_syncfix_seed43/checkpoints/best.pt and v31_syncfix_seed44/checkpoints/best.pt [seed43 DDR 0.8999, seed44 DDR 0.9176 but IDRiD XAI IoU20 <= 0.0661] (not promoted)
              ↳ DFR last-layer reweighting diagnostic → artifacts/runs/10_grounded_classifier/v31_dfr_v1/checkpoints/best.pt [DDR AUROC 0.8641, Sens@Opt 0.6554, D7 matched ratio 0.8720] (failed DDR gate; research only)
              ↳ Sparse BagNet grounded-classifier diagnostic → artifacts/runs/10_grounded_classifier/bagnet_v1_p33_r256/checkpoints/best.pt and bagnet_v1_p65_r512/checkpoints/best.pt [DDR AUROC 0.6293 / 0.6552; p65 IDRiD IoU20 0.0262, MAPLES IoU20 0.0061] (failed DDR + localization gates; research only)
              ↳ Concept Bottleneck grounded-classifier diagnostic → artifacts/runs/10_grounded_classifier/cbm_v1/checkpoints/best.pt [DDR AUROC 0.9268, best-threshold IDRiD mDice 0.0217, MAPLES mDice 0.0046] (failed localization gate; research only)
              ↳ v31 + v8b score-level fusion deployment → artifacts/checkpoints/best_pre_features_hflip_v2_20260527.pt.bak [DDR holdout AUROC 0.9403, threshold 0.38, v8b lesion evidence] (previous active)
              ↳ extended-feature hflip fusion deployment → artifacts/checkpoints/best_pre_quickqual_v1_20260529.pt.bak [DDR holdout AUROC 0.9431, threshold 0.3931, v8b lesion evidence] (previous active / rollback)
              ↳ QuickQual-line v31 + v8b fusion deployment (two-feature) → artifacts/checkpoints/best_pre_collinearity_refit_20260603.pt.bak [DDR holdout AUROC 0.9341, threshold 0.06, QuickQual-line v8b lesion evidence] (previous active / rollback)
              ↳ v31 collinearity refit (single v31_logit, 88 features) → artifacts/checkpoints/best.pt [DDR holdout AUROC 0.9360, threshold 0.0856, sens 0.8316, spec 0.9070] ← ACTIVE DEPLOYMENT ALIAS (v31_v8b_fusion_quickqual_v2)
            → per-lesion 4ch seg + Dice+BCE → artifacts/runs/07_lesion_evidence/v32_lesion_seg_evidence/checkpoints/best.pt [val AUROC 0.9992, train XAI AUPRC 0.0538 (seg_head)]
            → per-lesion weighted routing → artifacts/runs/07_lesion_evidence/v33_per_lesion_routing/checkpoints/best.pt [DDR AUROC 0.9131, test XAI AUC-IoU 0.0557]
            → calibrated per-lesion routing → artifacts/runs/07_lesion_evidence/v34_calibrated_routing/checkpoints/best.pt [DDR AUROC 0.9129, test XAI PG 0.5185]
            → v31 warmstart per-lesion routing → artifacts/runs/07_lesion_evidence/v35_warmstart_routing/checkpoints/best.pt [DDR AUROC 0.9081, test XAI AUPRC 0.1537]
            → U-Net decoder alignment + MAPLES masks → artifacts/runs/08_xai_decoder_alignment/v37_xai_multi_maples/checkpoints/best.pt [DDR AUROC 0.9103, MAPLES AUPRC 0.0136] (discarded)
            → no CAM alignment ablation → artifacts/runs/08_xai_decoder_alignment/v37b_xai_unet_only/checkpoints/best.pt [DDR AUROC 0.9200, IDRiD AUC-IoU 0.0625, MAPLES AUPRC 0.0161] (research only)
            → aux-loss sweep / two-stage fallback → artifacts/runs/08_xai_decoder_alignment/v37b_aux03/checkpoints/best.pt and v39_unet_2stage/checkpoints/best.pt [no deployment promotion]
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
- Classification training is run through `python -m drscreen.cli.train` and currently enforces Python 3.14. Use `py -3.14 -m drscreen.cli.train --config <config>`.
- Default classifier selection uses `train.selection_metric: val_auroc`: a checkpoint must meet `train.min_checkpoint_sensitivity` (default 0.80), then the highest validation AUROC is saved as run `best.pt`.
- Domain-generalization classifier runs may use `train.selection_metric: external_calibration_auroc`: the loader adds `data.external_calibration_split` (default `external_calibration`) and selects by that split's AUROC/sensitivity. In this mode the default sensitivity floor is 0.0 unless the config sets `train.min_checkpoint_sensitivity`, and global-best promotion is skipped because the metric is not comparable to legacy validation AUROC.
- `promotion_candidate` is set in `training_summary.json` only for `val_auroc` runs when a new run's `best_val_auroc` exceeds the previous global best; promotion remains manual.
- Optional classifier features: `keep_last_n_checkpoints` stores rolling epoch checkpoints, `early_stopping_patience/min_delta` stops non-head phases, and `swad_last_n_epochs` is supported only with `val_auroc` selection.
- Standalone lesion segmenter training is run through `py -3.14 -m drscreen.cli.train_seg --config <config>`. It filters to mask-valid rows, creates a deterministic train/val split from `data.seg_val_fraction`, selects the best checkpoint by validation mDice, and writes `best.pt`, `last.pt`, and `training_summary.json`. It also supports domain-weighted sampling, `initial_checkpoint_path`, early stopping, optional SWA, Dice+BCE/Focal-Tversky+BCE, FDA/AmpMix, and AdverIN diagnostics.
- Config merging is handled by `build_effective_checkpoint_config()` in `settings.py`, which ensures the runtime environment respects the model architecture and labels saved within the checkpoint.

---

## 5. Payload Contract (Backend Integration)

The `InferenceSession` produces a `prediction.payload` consumed by the `eye-project` backend.

### Core Fields
- `predicted_index`: `0` (Normal) or `1` (Abnormal).
- `predicted_label`: String representation of the prediction.
- `abnormal_probability`: Float (0.0 to 1.0).
- `decision_threshold`: Threshold used for the final Normal/Abnormal decision.
- `xai_error_code`: `"XAI_001"` for CAM failure, `"XAI_002"` for lesion-segmentation evidence failure, `"XAI_003"` for grounded-classifier patch evidence failure, otherwise `null`.
- `xai_no_region`: `true` when evidence generation produced no usable region, otherwise `false`.
- `evidence_type`: `"cam_research"`, `"lesion_segmentation"`, or research-only `"grounded_classifier"`.
- `lesion_summary`: Per-lesion area ratio / presence score summary when lesion evidence is active, otherwise `null`.
- `evidence_warning`: Evidence quality/status warning such as `"LESION_EVIDENCE_LOW_CONFIDENCE"`, otherwise `null`.
- `should_block`: Backend-gating flag reserved for hard AI-side blocking. Current active model keeps it `false`.
- `eval_metrics`: Compact external-test metrics loaded from `external_test_<version>_best_metrics.json`, merged with XAI/evidence metrics when available.
- `quality`, `quality_warning`, `quality_grade`, `quality_grade_confidence`: compatibility placeholders for backend QuickQual output; AI fills these with `null`.

### Artifact Paths
- `checkpoint_path`: Path to the model used.
- `prediction_path`: Path to the saved result JSON.
- `heatmap_path`: Path to the saved evidence overlay PNG. For current CAM runs, this is a Grad-CAM/Layer-CAM overlay; for Sparse BagNet research runs this is a patch-logit overlay, and for CBM research runs this is a concept-union overlay.
- `lesion_map_path`: Same overlay path when `evidence_type: lesion_segmentation` or `grounded_classifier`, otherwise `null`.

---

## 6. Current State, Known Gaps & Next Steps

### Current State and Completed Diagnostics

- **Active deployment**: Active config points to `v31_v8b_fusion_quickqual_v2` (v31 collinearity refit; single `v31_logit` feature, 88-feature meta-classifier). `artifacts/checkpoints/best.pt` contains the composite QuickQual-line v31 classifier + QuickQual-line v8b segmenter + numeric StandardScaler/LogReg meta-classifier checkpoint. Deployment threshold 0.0856 is calibration-selected to match the previous active sensitivity; holdout AUROC 0.9360, Sens 0.8316, Spec 0.9070, Acc 0.8693, F1 0.8641 (weakly dominates v1: AUROC +0.0019, sens +0.008, spec −0.0016). The active runtime sets `infer.preprocess_mode: none` because backend QuickQual preprocessing is expected to happen before AI inference. Immediate rollback (`v31_v8b_fusion_quickqual_v1`, two-feature) is backed up at `artifacts/checkpoints/best_pre_collinearity_refit_20260603.pt.bak`; deeper fallbacks at `best_pre_quickqual_v1_20260529.pt.bak` (hflip) and `best_pre_features_hflip_v2_20260527.pt.bak`.
- **2026-05-26 current-env v8b reproducibility corrected**: `seg_evidence_v8b_repro_seed43/44/45` initially failed to reproduce old v8b, but investigation found the cause was a code regression: legacy `processed/images` masks were being aligned with contentcrop geometry instead of circular geometry. After restoring `processed/images -> circular` and keeping `processed_contentcrop/images` / `processed_safezoom/images` on their own geometry, `seg_evidence_v8b_repro_seed43_geometryfix` recovered best val mDice 0.3330 versus old v8b 0.3388. Test best mDice: IDRiD 0.3867, MAPLES 0.2772, TJDR 0.3923, DDR_SEG 0.3889. Use the old v8b baseline again, but require explicit geometry mode in every preprocessing manifest/config.
- **2026-05-25 domain-overfit mitigation completed, no promotion**: Phase 0 created `manifest_with_val_mixed.csv` and confirmed the active fusion holdout baseline (AUROC 0.9402549575, threshold 0.38, Sens 0.8118, Spec 0.9238). Classifier candidates `v42_coral_baseline` and `v42_rsc_coral` reached external-holdout AUROC 0.9203/0.9201, a small gain over v31 base AUROC 0.9160 but far below active fusion AUROC 0.9403; `v41_ampmix` regressed to 0.9027. All classifier DG candidates still failed D5 domain separability (macro OVR AUROC 0.9855/0.9957/0.9886). Segmentation candidates `seg_evidence_v8b_swa`, `seg_evidence_v9_gin`, and `seg_evidence_v10_adverin` all regressed on IDRiD/MAPLES/TJDR/DDR_SEG mDice versus the v8b baseline. A1 pseudo-lesion augmentation was blocked by v8b mask audit fail ratio 0.7701, B4 CBMT was blocked by missing MESSIDOR-2/source-free adaptation inputs, and `fusion_v3` was not created.
- **2026-05-24 safezoom/contentcrop preprocessing completed, no promotion**: Safezoom aspect audit covered 17,900 images and improved foreground height in the 1.2-1.5 bin from median 406 to 443 px with p90 foreground loss 6.41% and no >10% loss cases. However, final classification did not beat the then-active circular fusion: `v31_v8b_late_fusion_contentcrop_v1` AUROC 0.9280, `v31_v8b_late_fusion_safezoom_v1` AUROC 0.9295, and `v31_v8b_late_fusion_safezoom_bbretrain_v1` AUROC 0.9204 versus 0.9403. Safezoom/contentcrop segmenters improved MAPLES mDice but regressed IDRiD and/or DDR_SEG, so the preprocessing variants remain diagnostic-only.
- **MAPLES ROI coordinate fix**: MAPLES annotation PNGs are 1500x1500 ROI-space masks and must be restored to original MESSIDOR coordinates using `data/raw/MAPLES-DR/AdditionalData/MESSIDOR-ROIs.csv` before project preprocessing geometry is applied. The old code resized ROI masks directly, so previous MAPLES train/eval metrics for mask-supervised segmentation runs are confounded. `drscreen/xai/iou.py::load_maples_masks()` now performs ROI-to-MESSIDOR canvas restoration, and MAPLES train/eval providers route through it. Validation against the official `maples_dr` loader matched 30 sample union masks at IoU 1.0.
- **Current lesion evidence path**: `seg_evidence_v8b_quickqual_v1` is now the active lesion evidence module inside `v31_v8b_fusion_quickqual_v1`. Threshold-0.5 aggregate evals: MAPLES mDice 0.2492 / union IoU 0.1939, TJDR mDice 0.3493 / union IoU 0.3054, DDR_SEG mDice 0.3536 / union IoU 0.2619. IDRiD mDice 0.1318 / union IoU 0.1299 is recorded but should not drive product evidence decisions because the QuickQual-line IDRiD overlay metric is flagged as unreliable above.
- **Phase 4-G G-4 grounded classifier re-entry completed**: v8b lesion maps were reduced to scalar evidence features and fitted with calibrated logistic classifiers. The best variant was `v8b_evidence_classifier_grid_v1` (all train domains, C-grid), with DDR external_test AUROC 0.8942, threshold 0.56, Sens 0.7639, Spec 0.8674. This is below active v31 AUROC 0.9160, so v8b evidence features alone do not replace the deployment classifier.
- **Phase 4-G G-5 late fusion completed, then superseded by hflip and QuickQual fusion lines**: `v31_v8b_late_fusion_sweep_v1` produced the circular score-level fusion baseline, and `v31_v8b_fusion_features_hflip_v2` superseded it for peak AUROC. The current active line is `v31_v8b_fusion_quickqual_v1`, selected for train-serve preprocessing consistency. Its formal threshold policy uses the same deterministic DDR 20% calibration / 80% holdout split. Calibration split: 2,504 images (normal 1,253 / abnormal 1,251). Holdout split: 10,018 images (normal 5,013 / abnormal 5,005). Selected threshold is **0.06**, with AUROC 0.9341, Acc 0.8660, Sens 0.8234, Spec 0.9086, F1 0.8600.
- **Fusion complementarity diagnostic completed, no deployment change**: `.omc/plans/fusion_complementarity_plan.md` Phase 0/Phase C were executed against the QuickQual active line. Phase 0 found a low-complementarity ceiling (Q-statistic 0.888; v8b corrected 37% of v31 errors; both-wrong 1,326 on holdout), so GPU-heavy B/A tracks remain on hold. Phase C showed `calfit_none` improves holdout AUROC over the active train-fit policy (0.940840 vs 0.934129; paired CI +0.00417 to +0.00923), but sensitivity drops from 0.8234 to 0.8010 and residual-weighting variants all underperform `calfit_none`. Result: residual-complementarity hypothesis failed; `calfit_none` is only a separate threshold/meta-fit policy candidate requiring explicit sensitivity-guard approval and staging validation. Active deployment remains `v31_v8b_fusion_quickqual_v1`.
- **Threshold**: Active `infer.threshold` is 0.06, aligned with the QuickQual fusion holdout threshold and checkpoint metadata.
- **v29 status**: XAI artifacts not generated. DDR AUROC 0.8629 (< v28 0.8924). ECA+CBAM spatial attention hurts both classification and XAI — direction discarded.
- **v30 XAI (test, baseline)**: block4 Layer-CAM — AUPRC 0.1311, AUC-IoU 0.0443, IoU top-20 0.0788, PG 0.3704. seg_head 직접 출력(0.0669)은 Layer-CAM 대비 열위. v30은 classifier-routing baseline으로 유지한다.
- **v31 base classifier**: val AUROC 0.9993 (epoch 5, early stop 8). `attention_mode: none` + Dice+BCE seg loss. DDR external_test AUROC **0.9160** (optimal thr 0.35, Sens 0.798, Spec 0.868). train XAI: AUPRC 0.1174, AUC-IoU 0.0491, IoU top-20 0.0601, PG 0.3333. test XAI block4 Layer-CAM: AUPRC 0.1409, AUC-IoU 0.0496, IoU top-20 0.0785, PG 0.3704. It is the circular-era base classifier preserved for comparison; the current active QuickQual fusion uses `v31_no_se_gated_quickqual_v1` as its classifier source.
- **v31 synchronized-transform rerun**: `v31_syncfix_rerun` reran the then-active v31 architecture after fixing classifier aux_seg image/mask transform sync. Dry-run confirmed `train_dataset_type=SegmentationManifestDataset` and `synchronized_mask_transform=True`; training completed with best epoch 9 / val AUROC 0.9993. DDR external_test regressed to AUROC 0.9082, optimal thr 0.24, Sens 0.7639, Spec 0.8905. IDRiD block4 Layer-CAM also regressed on AUPRC/AUC-IoU/IoU20 (0.1215 / 0.0394 / 0.0600 vs v31 0.1409 / 0.0496 / 0.0785), while MAPLES remained weaker (AUPRC 0.0125, IoU20 0.0067). Follow-up method sweep found v31 Grad-CAM is slightly higher than Layer-CAM on IDRiD AUC-IoU/IoU20 (0.0555 / 0.0827 vs 0.0496 / 0.0785), but syncfix still underperforms. Seed repeats did not rescue it: seed43 DDR AUROC 0.8999, seed44 DDR AUROC 0.9176 but IDRiD XAI IoU20 stayed <= 0.0661. Result: `NOT_PROMOTED`; v31 remains the fusion base classifier but not the whole active deployment. Canonical summary: `.omc/research/v31_syncfix_rerun_result.json` plus `artifacts/runs/07_lesion_evidence/v31_syncfix_seed43|44/`.
- **v32 completed (train XAI only)**: val AUROC 0.9992 (epoch 3). 4-channel per-lesion seg head + Dice+BCE. train XAI (seg_head): AUPRC 0.0538, AUC-IoU 0.0208. 현재 코드의 gated classifier는 4채널이면 per-lesion sigmoid + softmax weighted sum을 사용하지만, v32 artifact는 seg_head train 평가만 있어 제품 XAI 후보로 보지 않는다. v33+가 per-lesion routing 계열을 대체 평가한다.
- **v33 completed**: val AUROC 0.9980 (epoch 3, early stop 7). per-lesion 독립 sigmoid + softmax weighted sum gate. DDR AUROC 0.9131 (v31 0.9160 대비 -0.003). test XAI: AUPRC 0.1478, AUC-IoU 0.0557, PG 0.4074 — per-lesion routing 계열에서 AUC-IoU 기준 최고였다. 분류 소폭 하락으로 배포 미승격.
- **v35 completed**: val AUROC 0.9992 (epoch 9). v31 warmstart + 4ch per-lesion routing. DDR external_test AUROC 0.9081, optimal thr 0.18, Sens 0.793, Spec 0.874. test XAI: AUPRC 0.1537 (per-lesion routing 계열 최고), AUC-IoU 0.0525, IoU top-20 0.0796, PG 0.4074. Warmstart도 DDR 회귀를 해소하지 못해 배포 미승격.
- **v36/v37 decoder-alignment status**: v36(`configs/v36_xai_multi.yaml`)는 DDR external_test AUROC 0.9076으로 즉시 폐기. v37(`configs/v37_xai_multi_maples.yaml`)은 DDR AUROC 0.9103으로 최소 게이트는 통과했지만 v31(0.9160) 미달이고 optimal threshold가 0.15로 이동했다(Sens@0.5=0.5201). v37 test XAI block4 Layer-CAM: PG 0.3333, AUPRC 0.1230, AUC-IoU 0.0442, IoU top-20 0.0663. MAPLES test: PG 0.0167, AUPRC 0.0136, AUC-IoU 0.0037, IoU top-20 0.0086. MAPLES train mask를 포함했는데 MAPLES XAI가 v31 clean-cohort보다 회귀했으므로 배포/승격 후보가 아니다.
- **Phase 4-C diagnosis completed**: D1 MAPLES mask audit showed the old MAPLES-inclusive manifest had 134 `domain=MAPLES` rows with valid_rate 1.0, including 12 R0 rows whose union pixel mean was 0.0000985. Those R0 rows were valid but effectively empty mask supervision. D2 v37 seg_head direct evidence was weak: IDRiD IoU top-20 0.0366/AUPRC 0.0458 and MAPLES IoU top-20 0.0052/AUPRC 0.0069. v37b(`lambda_cam_align=0`) recovered DDR calibration and IDRiD XAI (DDR AUROC 0.9200, opt thr 0.27, IDRiD IoU top-20 0.0816), but MAPLES AUPRC remained 0.0161 (< v31 0.0172). v37c(R1+ MAPLES mask supervision only) fixed the R0 wiring but still regressed (DDR AUROC 0.9188, opt thr 0.31, IDRiD IoU top-20 0.0643, MAPLES AUPRC 0.0127). Conclusion: MAPLES R0 filtering is necessary but insufficient; CAM alignment should not be escalated.
- **Phase 4-D completed**: `v37b_aux03/04/05` λ_aux_seg sweep and `v39_unet_2stage` fallback were executed. aux03 passed DDR (AUROC 0.9203, opt thr 0.41, Sens 0.781, Spec 0.905) but regressed in XAI (IDRiD IoU top-20 0.0487, MAPLES AUPRC 0.0094). aux04/aux05 failed the Sens@Opt guard (0.766/0.770). v39 preserved the frozen v37b classifier path (DDR AUROC 0.9200, opt thr 0.27) and reproduced v37b XAI (IDRiD IoU top-20 0.0816, MAPLES AUPRC 0.0161), but did not improve. v39 seg_head direct output also failed as evidence (IDRiD IoU top-20 0.0387, MAPLES AUPRC 0.0069). At that point deployment stayed on v31; decoder-alignment, aux-loss escalation, and decoder-as-evidence are stopped in their current form.
- **Phase 4-E Track 1 completed**: Occlusion/RISE perturbation attribution was added as diagnostic-only XAI. Occlusion grid16 on v31 IDRiD test produced PG 0.1481, AUPRC 0.0832, AUC-IoU 0.0498, IoU top-20 0.0588, while Layer-CAM block4 remains better for lesion localization (PG 0.3704, AUPRC 0.1409, IoU top-20 0.0785). Occlusion is more faithful to classifier probability by deletion/insertion (deletion AUC 0.5971 vs Layer-CAM 0.7107; insertion-minus-deletion 0.3247 vs 0.2266), but MAPLES localization remains weak (AUPRC 0.0172, IoU top-20 0.0103). Decision: **FAITHFULNESS_ONLY**, not product XAI. This led to the independent lesion segmentation evidence path; the active deployment now uses QuickQual-line v8b lesion evidence inside `v31_v8b_fusion_quickqual_v1`.
- **Phase 4-E shortcut audit completed**: `drscreen/cli/diagnose_shortcut_audit.py` ran against active v31 (`configs/base.yaml`, `artifacts/checkpoints/best.pt`). D5 domain probe over block4 features: macro AUROC 0.9681 / macro F1 0.9119 for DDR vs IDRiD vs MAPLES. D6 MAPLES lesion-presence probe: AUROC 0.4048 / macro F1 0.4634 (limited by only 5 MAPLES no-lesion rows). D7 counterfactual style swap: matched non-lesion pixel swap changed abnormal probability 1.48x more than lesion-pixel swap; full non-lesion swap was 2.06x. Decision: shortcut hypothesis supported. Product wording must avoid causal XAI: use "classified abnormal + separately detected lesion candidate areas", not "classified because of these lesions".
- **Phase 4-E Track 2 segmentation evidence status**: `seg_evidence_v1` added a standalone ResNet50+U-Net 4-channel lesion segmenter with mask-only training and failed badly (best val mDice 0.00335, IDRiD test mDice 0.00129, MAPLES test mDice 0.00142). Follow-up inspection found two implementation issues: train-time image/mask augmentation sync and offline-preprocessed image vs raw-mask geometry alignment. `seg_evidence_v2_focal_tversky` fixed synchronized image/mask transforms and added Focal Tversky+BCE, but it was trained before the mask-geometry fix. Aligned re-eval only reports IDRiD mDice 0.0335 / union IoU 0.0886 and MAPLES mDice 0.0088 / union IoU 0.0148. Treat earlier v1/v2 mask-supervised conclusions as confounded by geometry mismatch unless retrained.
- **Mask-geometry v2 rerun**: `seg_evidence_v2_geomfix_retrain` reran the original v2 low-data setting after the mask-geometry fix (`manifest_with_maples_r1plus_preprocessed.csv`, IDRiD 54 + MAPLES 122 mask-valid rows, ResNet50+U-Net, Focal Tversky+BCE). It still failed: best val mDice 0.0071, IDRiD best union IoU 0.0603, MAPLES best union IoU 0.0055. Conclusion: geometry mismatch made prior v2 conclusions confounded, but fixing it alone does not rescue the low-data segmenter. Later TJDR/data-leverage runs remain the meaningful post-fix baseline.
- **Phase 4-F Step 0 completed**: The Phase 4-E "large lesion-mask data or stronger encoder" fork was resolved into a data-access / representation-leverage diagnostic path, then superseded by the grounded-classifier G1/G2/G3 diagnostics and Sprint 5 Phase 4-G evidence work. Initial access audit is recorded in `.omc/research/phase4f_data_access.json`: FGADR/TJDR/RETFound/SAM were not present locally at that point, FGADR required research-use approval, RETFound was research-only under CC-BY-NC-4.0, and TJDR was selected as the best immediate lesion-mask fallback once downloaded. Phase 4-F kept the target mapping to 4 channels (MA/HE/EX/SE); FGADR IRMA/NV were ignored and logged only.
- **Phase 4-F v3 S0 grounded-classifier prep completed**: `drscreen/cli/diagnose_v31_lesion_probe.py` was added and `.omc/research/phase4f_v3_d12_v31_probe.json` was generated. D12-A IDRiD AUROC 0.9977, D12-B MAPLES+fallback AUROC 0.8965, D12-U pooled AUROC 0.9495. Caveat: D12-B uses 5 native MAPLES no-lesion rows plus 115 Messidor grade-0 color-matched fallback rows; native MAPLES-only small-N check was AUROC 0.6000 with CI 0.20-1.00, so pure MAPLES decodability remains uncertain. `data/processed/lesion_concept_labels.csv` was generated with 18,191 rows, including 279 pixel-mask-valid rows and 8,797 weak-normal rows. v31 DDR external_test regression guard matched the recorded active metrics.
- **Phase 4-F v3 G1 DFR completed**: `drscreen/cli/dfr_relearn.py` generated `v31_dfr_v1` by freezing v31 backbone/gated pooling and retraining only the final classifier over a 4-group balanced set. Training fit the reweighting set well (train AUROC 0.9984), and D7 matched non-lesion/lesion ratio improved from 1.4752x to 0.8720x. However DDR external_test collapsed to AUROC 0.8641, optimal threshold 0.05, Sens@Opt 0.6554, while D5 domain AUROC 0.9681 and D6 MAPLES lesion AUROC 0.4048 were unchanged because the backbone features are unchanged. Result: `FAILED_DDR_GATE_DIAGNOSTIC_COMPLETE`; G1 is not a product candidate.
- **Phase 4-F v3 G3 Sparse BagNet completed**: `drscreen/models/sparse_bagnet.py` was added and wired through `build_model`, training, inference, and `eval_xai_iou.py --method bagnet`. Two research configs were trained: `bagnet_v1_p33_r256` (DDR AUROC 0.6293, Sens@Opt 0.4731) and `bagnet_v1_p65_r512` (DDR AUROC 0.6552, Sens@Opt 0.3950). The better p65 variant also failed lesion localization: IDRiD patch-logit evidence PG 0.1111, AUPRC 0.0372, AUC-IoU 0.0309, IoU top-20 0.0262; MAPLES PG 0.0167, AUPRC 0.0082, AUC-IoU 0.0053, IoU top-20 0.0061. Both are below the active v31 classifier and do not beat center Gaussian on IDRiD/MAPLES IoU. Result: `FAILED_DDR_AND_LOCALIZATION_GATES`; G3 is not a product/deployment candidate.
- **Phase 4-F v3 G2 CBM completed**: `drscreen/models/concept_bottleneck.py` was added and wired through build/train/infer/eval. Stage1 used mask-valid rows only and passed the redundant-solution entropy gate (`normalized_entropy_mean=0.9983`). Stage2 `cbm_v1` preserved classification better than v31 on DDR AUROC (0.9268, opt thr 0.21, Sens@Opt 0.8354, Spec@Opt 0.8770), but failed the lesion localization gate even after a 0.1-0.5 concept threshold sweep: best IDRiD per-class mDice 0.0217 / union IoU 0.0229, best MAPLES per-class mDice 0.0046 / union IoU 0.0043. Seg-head XAI stayed at baseline level (IDRiD IoU top-20 0.0432 vs center Gaussian 0.0436; MAPLES IoU top-20 0.0102 with PG 0.0). D7 matched ratio improved to 1.1913x, but D5 domain AUROC remained high at 0.9870. Result: `FAILED_LOCALIZATION_GATE_DIAGNOSTIC_COMPLETE`; G2 is research-only and not deployable.
- **Phase 4-F selection completed**: `.omc/research/phase4f_v3_selection.json` records that G1/G2/G3 all failed product gates. At that point deployment remained v31. Phase 4-G then shifted to data/representation leverage: TJDR/DDR-segmentation/Retinal-Lesions-style lesion-mask data and research-only fundus-pretrained/segmentation encoder probes.
- **Phase 4-G G-1/G-2 TJDR segmentation evidence completed**: TJDR is acquired under `data/raw/TJDR` with complete pairs: `train/image` 448, `train/annotation` 448, `test/image` 113, `test/annotation` 113. `TJDRMaskProvider` is implemented with TJDR labels `1=EX`, `2=HE`, `3=MA`, `4=SE` remapped to project channel order `MA/HE/EX/SE` as `3/2/1/4`. `build_manifest --include-tjdr` adds TJDR train rows by default; optional `--include-tjdr-test` writes `split='tjdr_test'`. `preprocess_images.py` produced `data/processed/manifest_with_maples_tjdr_preprocessed.csv`, and `configs/seg_evidence_v3_tjdr.yaml` uses it. After the mask-geometry fix, `seg_evidence_v3_tjdr` was retrained: best val mDice 0.2482 at epoch 32. Aligned eval: IDRiD test mDice 0.2055 / union IoU 0.2209, TJDR test mDice 0.3524 / union IoU 0.3490, MAPLES test mDice 0.0051 / union IoU 0.0071. Threshold sweep: IDRiD best threshold 0.05 (mDice 0.2419 / union IoU 0.2674), TJDR best union-IoU threshold 0.50 (mDice 0.3524 / union IoU 0.3490), MAPLES best threshold 0.05 still only mDice 0.0070 / union IoU 0.0091. Conclusion: TJDR improves in-domain/IDRiD lesion segmentation evidence, but MAPLES cross-domain generalization remained unresolved at this step.
- **Phase 4-G G-2 stronger encoder baseline completed**: `seg_evidence_v4_deeplab_tjdr` adds a DeepLabV3-ResNet50 evidence segmenter and `configs/seg_evidence_v4_deeplab_tjdr.yaml`. It was manually stopped at epoch 36 after no best update since epoch 11; `train_seg` now supports `early_stopping_patience/min_delta` for future runs. Best checkpoint eval at threshold 0.5: IDRiD mDice 0.2445 / union IoU 0.2727, MAPLES mDice 0.0096 / union IoU 0.0126, TJDR mDice 0.2543 / union IoU 0.2358. Threshold sweep: IDRiD best mDice 0.2460 at threshold 0.25, MAPLES best mDice 0.0121 at threshold 0.05, TJDR best union IoU 0.2364 at threshold 0.65. Decision: IDRiD improved, but MAPLES still failed the 0.05 mDice gate and TJDR regressed versus v3, so v4 was not promoted.
- **Phase 4-G MAPLES-target FDA diagnostic completed**: `seg_evidence_v5_maples_fda_tjdr` adds `SegmentationFDAManifestDataset`, using MAPLES images as FDA style references for non-MAPLES segmentation training samples. Best epoch 21, val mDice 0.2269, early stop epoch 29. Threshold 0.5 eval: IDRiD mDice 0.2458 / union IoU 0.3068, MAPLES mDice 0.0114 / union IoU 0.0133, TJDR mDice 0.3108 / union IoU 0.2852. Threshold sweep: MAPLES best threshold remains 0.05 with mDice 0.0141 / union IoU 0.0183. Decision: FDA improved IDRiD/TJDR relative to v4 but did not solve MAPLES cross-domain lesion evidence.
- **Phase 4-G MAPLES-heavy fine-tune diagnostic completed**: `seg_evidence_v6_maples_finetune_tjdr` warm-starts from v5 best with MAPLES-heavy domain sampling (MAPLES 4.0 / IDRiD 1.0 / TJDR 0.5) and low LR 2e-5. Best epoch 2, val mDice 0.2145, early stop epoch 6. Threshold 0.5 eval: IDRiD mDice 0.2144 / union IoU 0.2574, MAPLES mDice 0.0134 / union IoU 0.0175, TJDR mDice 0.2816 / union IoU 0.2373. Threshold sweep: MAPLES best mDice 0.0165 / union IoU 0.0201 at threshold 0.05, still far below gate 0.05. IDRiD/TJDR best union IoU slightly regress versus v5. Decision: fine-tune/upweighting was insufficient.
- **MAPLES-only specialist segmenter diagnostic completed**: `seg_evidence_v7_maples_only` trains ResNet50+U-Net only on preprocessed MAPLES train rows. Manifest audit confirmed 122 MAPLES train rows, sample images are already 512x512 preprocessed paths. Dry-run split was train 104 / val 18, best val mDice only 0.0090 at epoch 1, early stop epoch 9. MAPLES test threshold sweep failed (best mDice 0.0039, best union IoU 0.0056); IDRiD transfer also stayed weak (mDice 0.0222, union IoU 0.0382). Decision: MAPLES failure was not solved by target-only fine-tuning on the current 512 preprocessed data.
- **Sprint boundary correction**: Phase 4-G is Sprint 5 work. Sprint 4 closes with v31 active deployment and Phase 4-E/F diagnostics. TJDR/DDR_SEG integration, MAPLES ROI correction, v8b standalone evidence baseline, v8b evidence classifier, and v31+v8b late fusion are Sprint 5 records. FGADR is excluded from the default path because the access procedure is too heavy. Candidate sources should be limited to datasets with clear access/license before training, such as Retinal-Lesions or another locally provided 4-channel mask source.
- **Phase 4-G v8 DDR segmentation completed, then corrected**: Original `seg_evidence_v8_ddrseg_tjdr` used misaligned MAPLES ROI masks and is diagnostic only. Corrected rerun `seg_evidence_v8b_ddrseg_tjdr_maplesfix` uses the same DDR_SEG/TJDR/IDRiD/MAPLES composite data after ROI restoration. Mask-valid counts: DDR_SEG 532 / TJDR 448 / MAPLES 122 / IDRiD 54. Best epoch 38, val mDice 0.3388. Evaluation: IDRiD best mDice 0.4151 / union IoU 0.3903, TJDR best mDice 0.3788 / union IoU 0.3149, DDR_SEG best mDice 0.3945 / union IoU 0.2880, MAPLES best mDice 0.2928 / union IoU 0.2121. `seg_evidence_v5b_maples_fda_tjdr_maplesfix` was also rerun as the strongest prior MAPLES-rejected candidate; it improved MAPLES to mDice 0.1595 / union IoU 0.1385 but remains below v8b. Decision: v8b became the Phase 4-G best standalone evidence baseline and is now used inside `v31_v8b_fusion_v2`.
- **Phase 4-G G-4 completed**: `drscreen/cli/lesion_evidence_classifier.py` was added for diagnostic-only scalar classifiers over v8b lesion maps. Variants tested: all train domains AUROC 0.8828, classification-domain-only AUROC 0.8479, APTOS-only AUROC 0.8725, and C-grid AUROC 0.8942 on DDR external_test. Best G-4 still underperforms v31 AUROC 0.9160, so grounded classifier re-entry through scalar lesion evidence alone is not promoted. G-5 late fusion was then completed and later superseded by `v31_v8b_fusion_features_hflip_v2`.
- **v27 MIL attention (discarded)**: train XAI AUC-IoU 0.0119 — random baseline(0.0260) 이하. MIL attention 방향 폐기.
- **CAM Phase-0 gate**: CAM/Layer-CAM research path 기준으로는 test split 전 모델 FAIL이었다. center_gaussian+2σ=0.1089 임계값은 CAM 계열 단기 목표로 현실적으로 달성 불가 수준이었다. 현재 active evidence는 v8b lesion segmentation metric으로 별도 평가한다.
- **IDRiD contamination**: XAI eval(`A. Segmentation`)과 분류 학습(`B. Disease Grading`)은 파일은 다르나 동일 환자 포함(patient-level overlap). file-level contamination 없음. 상세 내용은 아래 섹션 참조.
- **Phase-0 gate 기준 재정의**: 기존 2σ threshold는 모든 모델이 달성 불가 수준임을 확인. `eval_xai_iou.py --gate-sigma`는 콘솔 gate 출력의 sigma multiplier를 바꾸지만, 저장 JSON의 `phase0_gate` 필드는 여전히 canonical 2σ 기준을 기록한다. 현재 test split 기준: 2σ=0.1089 (v37b/v39 best=0.0625, FAIL), 1σ=0.0728 (v37b/v39 FAIL), ~0.5σ=0.0547 (v37b/v39 PASS). 2σ는 aspirational 기준으로 유지, 현실적 진척도 측정은 콘솔 출력의 1σ 또는 절대값 0.05를 함께 확인하는 방식이 맞다.
- **DiceBCELoss**: `drscreen/train/loss.py`에 구현 완료. `engine.py:196-209`에 wiring. `seg_loss_type: dice_bce` config로 활성화.
- **v32 구조 상태**: v32 artifact는 4채널 seg_head train 평가만 존재하고 XAI 수치가 낮다. 현재 코드(`drscreen/models/aux_seg.py`)의 4채널 gated classifier는 per-lesion sigmoid + softmax weighted sum을 사용한다. 단, 단일 evidence map 생성용 `predict_seg_union()`은 아직 `amax(dim=1)` union을 사용한다.
- **MAPLES-DR**: 확보 완료. `data/raw/MAPLES-DR/AdditionalData/` (dataset_record.yaml, annotations/, `MESSIDOR-ROIs.csv` 포함). train 138장 / test 60장, 12종 biomarker 마스크. MAPLES pathology masks are ROI-space masks and must go through the ROI restoration in `drscreen/xai/iou.py` before train/eval use. `MAPLESMaskProvider` 구현 완료 (`drscreen/data/mask_providers.py`)는 MA/HE/EX/CWS 4채널 pathology mask 로딩 범위다. `eval_xai_iou.py --mask-provider maples`로 MAPLES clean-cohort CAM 평가는 가능하고, OD masking도 지원한다. 단, Phase 1 anatomy audit용 `attribution_in_anatomy_ratio`/`lesion_vs_confounder_ratio` 지표는 아직 구현되지 않았다.
- **Unit Tests**: regression tests exist under `tests/regression/`: payload contract, fusion contract, and preprocessing geometry. Current smoke command: `py -3.14 -m pytest tests/regression -q`.
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
- 단, 현재 모델들의 XAI 수치가 모두 낮은 수준(v37b/v39 AUPRC 0.1546, AUC-IoU 0.0625)이므로 오염에 의한 인플레이션이 아닌 실질적인 한계로 판단.

### 대응

- 현재: patient-level split 검증 툴링 없음. XAI eval 결과를 "참고 수치"로만 활용.
- 이상적: MAPLES-DR 데이터셋 사용 시 완전히 분리된 코호트에서 XAI eval 가능.

---

## 8. Phase-0 XAI Gate

### 정의

Phase-0 gate는 CAM/Layer-CAM 연구 경로에서 모델의 병변 localization 능력이 **center_gaussian baseline을 유의미하게 초과**하는지를 판별하는 threshold다. 현재 active `v31_v8b_fusion_quickqual_v1`의 병변 evidence는 CAM이 아니라 v8b lesion segmentation이므로 `xai_seg_mdice`, `xai_seg_union_iou`, `xai_auc_iou`를 별도로 본다.

```
gate = model_AUC_IoU > center_gaussian_mean + N × center_gaussian_std
```

- `N=2` (기존): aspirational 기준. test split 기준 threshold=0.1089. **CAM 계열 전 모델 FAIL.**
- `N=1` (완화): progress gate 후보. test split 기준 threshold=0.0728. **CAM 계열 전 모델 FAIL (v37b/v39 best=0.0625).**
- `N=0.5` (완화): test split 기준 threshold=0.0547. v37b/v39(0.0625) PASS.

### 현재 결론

- 2σ 기준은 CAM research path의 aspirational 목표로만 유지 (단기 달성 불가).
- CAM 단기 진척도 측정은 `--gate-sigma 1.0` 또는 절대값 0.05 기준 사용.
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

### Runtime Policy

- **Training only**: use Python 3.14. Prefer the wrapper below or call `py -3.14` explicitly. The training CLI refuses other Python versions to avoid accidentally using the deployment/runtime interpreter.
- **Deployment/runtime**: keep the existing backend/Docker runtime unchanged. This training policy does not modify deployment Docker files or inference service execution.
- Avoid bare `python -m drscreen.cli.train ...` in automation because PATH may resolve to a non-training Python.

### Training
```powershell
.\train.ps1 -Config configs/v9_fda.yaml
# equivalent:
py -3.14 -m drscreen.cli.train --config configs/v9_fda.yaml
```

### Evaluation (DDR external test)
```bash
python -m drscreen.cli.evaluate --config configs/v9_fda.yaml --split external_test
```

### Inference
```bash
python -m drscreen.cli.infer --config configs/v9_fda.yaml --image path/to/image.png
```
