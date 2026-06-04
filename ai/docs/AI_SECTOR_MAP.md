# AI Sector Map

This file splits the current `eye-project/ai` code and experiment artifacts into reviewable sectors.
Use it as the checklist for the loop: inspect one sector, compare it with docs, update docs, verify, then move to the next sector.

## Review Rule

- Do not physically move Python modules unless import paths and config references are refactored together.
- Physical experiment artifacts are already grouped by `artifacts/runs/<primary_group>/<run_id>/`.
- `configs/base.yaml` plus `artifacts/checkpoints/best.pt` define the active deployment.
- `docs/EXPERIMENT_REGISTRY.md` classifies experiment families and run outcomes.

## Sector 0 — Active Deployment Runtime

Purpose: currently deployed inference contract and active checkpoint alias.

Code/config:
- `configs/base.yaml`
- `artifacts/checkpoints/best.pt`
- `drscreen/infer/service.py`
- `drscreen/infer/pipeline.py`
- `drscreen/infer/late_fusion_features.py`
- `drscreen/models/fusion.py`
- `drscreen/models/build.py`
- `drscreen/settings.py`

Current active evidence:
- `configs/base.yaml`: `project.version: v31_v8b_fusion_quickqual_v2`
- `artifacts/checkpoints/best.pt`: `architecture=v31_v8b_fusion`, `optimal_threshold=0.08563088401268978`, `feature_schema[0]=v31_logit` (88 features; v31 collinearity refit)
- source classifier: `artifacts/runs/99_misc/v31_no_se_gated_quickqual_v1/checkpoints/best.pt`
- source segmenter: `artifacts/runs/99_misc/seg_evidence_v8b_quickqual_v1/checkpoints/best.pt`
- source fusion metrics: `artifacts/runs/99_misc/v31_v8b_late_fusion_quickqual_v1_v31rep/evaluations/v31_v8b_late_fusion_quickqual_v1_v31rep_metrics.json` (key `classification_domains:late_fusion:v31_logit`)
- runtime compact metrics: `artifacts/evaluations/external_test_v31_v8b_fusion_quickqual_v2_best_metrics.json`
- rollback (previous active, two-feature v1): `artifacts/checkpoints/best_pre_collinearity_refit_20260603.pt.bak`

Docs touched in this pass:
- `docs/AI_HANDOFF.md`
- `docs/EXPERIMENT_REGISTRY.md`
- `docs/INDEX.md`

## Sector 1 — Data, Manifest, Preprocessing, Masks

Purpose: raw/preprocessed data manifests, geometry, mask providers, QuickQual/circular/safezoom/contentcrop paths.

Code:
- `preprocess_images.py`
- `drscreen/data/datasets.py`
- `drscreen/data/manifest_builder.py`
- `drscreen/data/mask_providers.py`
- `drscreen/data/transforms.py`
- `drscreen/cli/build_manifest.py`
- `drscreen/cli/diagnose_color_canonicalization.py`
- `drscreen/cli/diagnose_resize_path_skew.py`
- `drscreen/cli/diagnose_selection_sanity.py`
- `drscreen/cli/diagnose_v8b_mask_quality.py`

Experiment/document anchors:
- `.omc/plans/preprocessing_color_canonicalization_plan.md`
- `.omc/plans/preprocessing_safezoom_plan.md`
- `.omc/research/preprocessing_color/`
- `.omc/research/preprocessing_safezoom/`
- `artifacts/preprocess_debug/`

## Sector 2 — Core Training, Evaluation, Checkpointing

Purpose: shared classifier/segmenter training loops, selection, evaluation, and checkpoint policy.

Code:
- `drscreen/cli/train.py`
- `drscreen/cli/train_seg.py`
- `drscreen/train/runner.py`
- `drscreen/train/seg_runner.py`
- `drscreen/train/engine.py`
- `drscreen/train/evaluate.py`
- `drscreen/train/checkpointing.py`
- `drscreen/train/data_loader_factory.py`
- `drscreen/train/model_setup.py`
- `drscreen/utils/seed.py`

Current code facts:
- Classifier training CLI (`drscreen.cli.train`) and standalone segmenter training CLI (`drscreen.cli.train_seg`) both enforce Python 3.14.
- Classifier selection supports `train.selection_metric: val_auroc` and `external_calibration_auroc`.
- `val_auroc` runs use the default sensitivity floor 0.80 and may become manual `promotion_candidate`s.
- `external_calibration_auroc` runs select on `data.external_calibration_split` and skip global-best promotion because the metric is not comparable with legacy validation AUROC.
- Standalone lesion segmenter training filters mask-valid rows, splits them deterministically by `data.seg_val_fraction`, and selects `best.pt` by validation mDice.

Docs touched in this pass:
- `docs/AI_HANDOFF.md`

## Sector 3 — Classifier Lineage and CAM/XAI Research

Purpose: v1-v39 classifier lineage, attention ablation, Layer-CAM/CAM methods, decoder alignment.

Code:
- `drscreen/models/aux_seg.py`
- `drscreen/xai/evaluation.py`
- `drscreen/xai/perturbation.py`
- `eval_xai_iou.py`
- `eval_xai_maples.py`
- `sweep_xai_blocks.py`
- `visualize_heatmap_gt.py`

Artifact groups:
- `00_baselines_and_early`
- `01_ssl_lineage`
- `02_domain_generalization`
- `03_resolution_layercam`
- `04_lesion_supervision`
- `05_xai_attention_ablation`
- `06_xai_classifier_routing`
- `07_lesion_evidence`
- `08_xai_decoder_alignment`

Current code facts:
- `eval_xai_iou.py` supports CAM methods `gradcam`, `layercam`, `hirescam`, `gradcam++`, `scorecam`, `eigencam`, `ig`; perturbation methods `occlusion`, `rise`; and direct patch evidence `bagnet`.
- Multi-block fusion (`--target-blocks`) is for CAM methods only. It is incompatible with `--methods`, perturbation methods, and direct evidence.
- `--mask-provider maples` uses MAPLES-DR masks, but multi-method comparison mode is IDRiD-only.
- Metrics are Pointing Game, pixel AUPRC, AUC-IoU, and top-N% union/per-lesion IoU. Baselines are random, center Gaussian, and retina uniform.
- `--gate-sigma` affects the printed Phase-0 gate; saved JSON keeps the canonical 2-sigma `phase0_gate`.

Docs touched in this pass:
- `docs/AI_HANDOFF.md`

## Sector 4 — Standalone Lesion Evidence Segmentation

Purpose: classifier-independent lesion segmentation evidence path, TJDR/DDR_SEG/MAPLES/IDRiD mask learning.

Code:
- `eval_seg_evidence.py`
- `drscreen/cli/lesion_evidence_classifier.py`
- `drscreen/cli/diagnose_maples_masks.py`
- `visualize_lesion_pred_gt.py`

Artifact group:
- `09_evidence_segmentation`
- QuickQual-line active source segmenter runs are currently still under `99_misc` until a later physical migration.

Key runs:
- `seg_evidence_v8b_ddrseg_tjdr_maplesfix`
- `seg_evidence_v8b_quickqual_v1`
- `seg_evidence_v8b_contentcrop_v1`
- `seg_evidence_v8b_safezoom_v1`
- `seg_evidence_v8b_swa`
- `seg_evidence_v9_gin`
- `seg_evidence_v10_adverin`

Current code facts:
- `LesionSegEvidence` supports `encoder: resnet50` with U-Net-style decoder and `encoder: deeplabv3_resnet50`.
- `eval_seg_evidence.py` supports `--mask-provider idrid|maples|tjdr|ddr_seg`, threshold-0.5 evaluation, and threshold sweeps.
- Evaluation metrics are mDice, mIoU, union Dice, union IoU, and per-class MA/HE/EX/SE Dice/IoU.
- If an evaluation manifest uses offline preprocessed paths (`processed*` prefixes), eval applies geometry-aware mask alignment instead of assuming raw image geometry.
- Active deployment uses `seg_evidence_v8b_quickqual_v1` inside `v31_v8b_fusion_quickqual_v1`; circular-era standalone best remains `seg_evidence_v8b_ddrseg_tjdr_maplesfix`.

Docs touched in this pass:
- No additional docs changes required beyond Sector 0/1 active-source clarification already applied to `AI_HANDOFF.md`, `EXPERIMENT_REGISTRY.md`, and `SPRINT5_Devlog.md`.

## Sector 5 — Grounded Classifier and Fusion Diagnostics

Purpose: shortcut-free classifier diagnostics, DFR/BagNet/CBM, v31+v8b fusion packaging, fusion complementarity.

Code:
- `drscreen/cli/dfr_relearn.py`
- `drscreen/cli/diagnose_shortcut_audit.py`
- `drscreen/cli/diagnose_v31_lesion_probe.py`
- `drscreen/cli/late_fusion_classifier.py`
- `drscreen/cli/build_fusion_checkpoint.py`
- `drscreen/cli/eval_fusion_full.py`
- `drscreen/cli/diagnose_fusion_complementarity.py`
- `drscreen/models/fusion.py`

Artifact group:
- `10_grounded_classifier`

Plans/research:
- `.omc/plans/domain_overfit_mitigation_plan.md`
- `.omc/plans/v31_v8b_fusion_improvement_plan.md`
- `.omc/plans/fusion_complementarity_plan.md`
- `.omc/research/fusion_complementarity/`

Current code facts:
- `V31V8bFusion.forward()` returns the classifier logits path; `predict_fusion_score()` runs the classifier and segmenter, extracts lesion scalar features, then applies the stored numpy scaler/logistic-regression meta-classifier.
- `drscreen/cli/late_fusion_classifier.py` fits StandardScaler + LogisticRegression on the configured train split. The DDR calibration split is used for threshold selection unless a diagnostic explicitly refits the meta-classifier on calibration data.
- `drscreen/cli/diagnose_fusion_complementarity.py` implements Phase 0/Phase C feature-cache, complementarity, calibration-fit, residual-weighting, and paired-bootstrap diagnostics.

Current result:
- `.omc/research/fusion_complementarity/phase0_power_ceiling.json`: v31/v8b complementarity ceiling is limited (Q-statistic 0.888; v8b corrects 37% of v31 errors; both-wrong 1,326).
- `.omc/research/fusion_complementarity/phase_c_calfit_ablation.json`: `calfit_none` improves holdout AUROC over active train-fit fusion (0.940840 vs 0.934129), but sensitivity drops (0.8010 vs 0.8234) and residual-weighting variants underperform `calfit_none`. This leaves active deployment unchanged at `v31_v8b_fusion_quickqual_v1`.

## Sector 6 — Miscellaneous / Legacy / Staging

Purpose: reproducibility reruns, temporary diagnostics, old scripts, and staging aliases that do not yet have a canonical group.

Artifact group:
- `99_misc`

Examples:
- `v31_no_se_gated_quickqual_v1`
- `seg_evidence_v8b_quickqual_v1`
- `v31_v8b_late_fusion_quickqual_v1`
- QuickQual 1024/v2/v3 diagnostics
- `seg_evidence_v8b_repro_seed43/44/45`
- `seg_evidence_v8b_repro_seed43_compat`
- `seg_evidence_v8b_repro_seed43_geometryfix`

These are not automatically disposable. Some `99_misc` runs are current active sources until a later migration assigns a primary group; others are diagnostic-only reruns retained for reproducibility and regression analysis.
