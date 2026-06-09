# Experiment Registry

This file is the canonical classification index for existing `eye-project/ai` experiment artifacts.

It is intentionally kept because the project has many historical checkpoints, evaluation JSONs, and migrated run folders. `AI_HANDOFF.md` is the current architecture handoff, while this registry answers "which run belongs to which experiment family, where are its artifacts, and why was it kept or rejected?"

## Current Active Snapshot

| Item | Current value | Source |
|---|---|---|
| Active deployment | `v31_v8b_fusion_quickqual_v2` (v31 collinearity refit, promoted 2026-06-03) | `configs/base.yaml` + `artifacts/checkpoints/best.pt` |
| Runtime architecture | `v31_v8b_fusion` | `configs/base.yaml` |
| Classifier source | `v31_no_se_gated_quickqual_v1` inside the fusion checkpoint | `artifacts/runs/99_misc/v31_no_se_gated_quickqual_v1/checkpoints/best.pt` |
| Evidence source | `seg_evidence_v8b_quickqual_v1` inside the fusion checkpoint | `artifacts/runs/99_misc/seg_evidence_v8b_quickqual_v1/checkpoints/best.pt` |
| Meta-classifier | numeric StandardScaler + LogisticRegression embedded in the composite checkpoint; single `v31_logit` feature + v8b lesion features (88 features total; redundant `v31_probability` dropped) | `artifacts/checkpoints/best.pt` |
| Deployment threshold | `0.08563088401268978` (calibration-selected to match v1 sensitivity 0.8234) | `artifacts/evaluations/external_test_v31_v8b_fusion_quickqual_v2_best_metrics.json` |
| DDR holdout performance | AUROC 0.9360, Sens 0.8316, Spec 0.9070, Acc 0.8693, F1 0.8641 | `artifacts/evaluations/external_test_v31_v8b_fusion_quickqual_v2_best_metrics.json` |
| Source fusion metric artifact | `classification_domains:late_fusion:v31_logit` | `artifacts/runs/99_misc/v31_v8b_late_fusion_quickqual_v1_v31rep/evaluations/v31_v8b_late_fusion_quickqual_v1_v31rep_metrics.json` |
| Active checkpoint metadata | `architecture=v31_v8b_fusion`, `optimal_threshold=0.08563088401268978`, `feature_schema[0]=v31_logit` (88 features) | `artifacts/checkpoints/best.pt` |
| MAPLES/TJDR/DDR_SEG lesion evidence | MAPLES mDice 0.2492 / union IoU 0.1939; TJDR mDice 0.3493 / union IoU 0.3054; DDR_SEG mDice 0.3536 / union IoU 0.2619 | `artifacts/runs/99_misc/seg_evidence_v8b_quickqual_v1/evaluations/` |
| Previous active rollback | `v31_v8b_fusion_quickqual_v1` (two-feature, AUROC 0.9341, threshold 0.06) | `artifacts/checkpoints/best_pre_collinearity_refit_20260603.pt.bak` |

Historical artifacts have been physically migrated to the primary group for each run. Canonical storage now follows:

- `artifacts/runs/<primary_group>/<run_id>/checkpoints/`
- `artifacts/runs/<primary_group>/<run_id>/evaluations/`
- `artifacts/runs/<primary_group>/<run_id>/logs/`

The resolver in `drscreen/settings.py` keeps legacy checkpoint references readable when a migrated file exists under `artifacts/runs/`.

## Rules

- `run_id` remains the existing version name, e.g. `v24_multitask`.
- `group` is the research question or experiment family.
- A run may belong to multiple groups. In that case, the primary group is the physical storage location and secondary groups reference that location.
- `artifacts/checkpoints/best.pt` is the fixed active deployment checkpoint alias.
- Active deployment is determined by `configs/base.yaml` plus the physical checkpoint currently placed at `artifacts/checkpoints/best.pt`, not by this registry alone.
- `deployment_candidates` is a virtual decision group, not a physical `artifacts/runs/<group>/` directory.

## Physical Artifact Groups

| Group | Purpose | Runs |
|---|---|---|
| `00_baselines_and_early` | Initial baselines and early supervised attempts | `effnet_1shot`, `v3`, `v4`, `v5` |
| `01_ssl_lineage` | SSL lineage, SSL contamination checks, focal variants | `v4.1`, `v4b`, `v4b_alpha_only`, `v6`, `v6_alpha_only`, `v6_gamma_only` |
| `02_domain_generalization` | Messidor inclusion, FDA, SWAD, IBN, CORAL | `v7_messidor_train`, `v8_mixstyle`, `v9_fda`, `v10_swad`, `v11_fda_swad`, `v12_fda_imagenet`, `v13_fda_swad`, `v14_ibn`, `v15_fda_a10`, `v16_focal_g1`, `v17_focal_g2`, `v18_focal_g3`, `v19_swad_focal_g2`, `v20_coral` |
| `03_resolution_layercam` | 512px training and Layer-CAM deployment experiments | `v7_512_messidor_train`, `v7_512_messidor_train_contentcrop_v1`, `v7_512_messidor_train_safezoom_v1`, `v17_512_focal_g2`, `v21_512_focal_g2`, `v21_512_layercam` |
| `04_lesion_supervision` | Auxiliary lesion mask supervision and lesion-aware heads | `v24_multitask`, `v25_multitask_l1`, `v26_multitask_l3`, `v27_mil_attention` |
| `05_xai_attention_ablation` | Matched XAI attention ablation and block sweeps | `v24_multitask`, `v28_no_attention`, `v29_with_attention` |
| `06_xai_classifier_routing` | Lesion gate routing into classifier pooling path | `v30_gated_pooling` |
| `07_lesion_evidence` | SE/ECA 제거 + gated pooling 유지 대조군(v31), synchronized mask-transform rerun/seed repeat, per-lesion routing 시리즈(v32~v35), v31 shortcut audit | `v31_no_se_gated`, `v31_syncfix_rerun`, `v31_syncfix_seed43`, `v31_syncfix_seed44`, `v31_no_se_gated_contentcrop_v1`, `v31_no_se_gated_contentcrop_bbretrain_v1`, `v31_no_se_gated_safezoom_v1`, `v31_no_se_gated_safezoom_bbretrain_v1`, `v32_lesion_seg_evidence`, `v33_per_lesion_routing`, `v34_calibrated_routing`, `v35_warmstart_routing` |
| `08_xai_decoder_alignment` | U-Net auxiliary decoder, CAM alignment, MAPLES-inclusive lesion supervision, aux-loss sweep, two-stage decoder fallback | `v36_xai_multi`, `v37_xai_multi_maples`, `v37b_xai_unet_only`, `v37c_xai_maples_r1plus`, `v37b_aux03`, `v37b_aux04`, `v37b_aux05`, `v38_xai_coral`, `v39_unet_2stage` |
| `09_evidence_segmentation` | Classifier-independent lesion segmentation evidence scaffold and TJDR/DDR-segmentation/Retinal-Lesions-style large-mask segmenter work. FGADR is excluded from the active path because access is too heavy | `seg_evidence_v1`, `seg_evidence_v2_focal_tversky`, `seg_evidence_v2_geomfix_retrain`, `seg_evidence_v3_tjdr`, `seg_evidence_v4_deeplab_tjdr`, `seg_evidence_v5_maples_fda_tjdr`, `seg_evidence_v5b_maples_fda_tjdr_maplesfix`, `seg_evidence_v6_maples_finetune_tjdr`, `seg_evidence_v7_maples_only`, `seg_evidence_v8_ddrseg_tjdr`, `seg_evidence_v8b_ddrseg_tjdr_maplesfix`, `seg_evidence_v8b_contentcrop_v1`, `seg_evidence_v8b_safezoom_v1`, `seg_evidence_v8b_swa`, `seg_evidence_v9_gin`, `seg_evidence_v10_adverin` |
| `10_grounded_classifier` | Shortcut-free classifier redesign diagnostics and future grounded architectures | `v31_dfr_v1`, `bagnet_v1_p33_r256`, `bagnet_v1_p65_r512`, `cbm_v1_stage1`, `cbm_v1`, `v8b_evidence_classifier_v1`, `v8b_evidence_classifier_clsdomains_v1`, `v8b_evidence_classifier_aptos_v1`, `v8b_evidence_classifier_grid_v1`, `v31_v8b_late_fusion_sweep_v1`, `v31_v8b_fusion_v2`, `v31_v8b_late_fusion_features_v2`, `v31_v8b_fusion_features_v2`, `v31_v8b_fusion_features_hflip_v2`, `v31_v8b_fusion_features_hflip_recalc_v2`, `v31_v8b_late_fusion_contentcrop_v1`, `v31_v8b_fusion_contentcrop_v1`, `v31_v8b_late_fusion_safezoom_v1`, `v31_v8b_fusion_safezoom_v1`, `v31_v8b_late_fusion_safezoom_bbretrain_v1`, `v41_ampmix`, `v42_coral_baseline`, `v42_rsc_coral` |
| `99_misc` | Temporary/legacy miscellaneous artifact bucket. Some current QuickQual active-source runs still live here pending a later physical migration | `v31_no_se_gated_quickqual_v1`, `seg_evidence_v8b_quickqual_v1`, `v31_v8b_late_fusion_quickqual_v1`, `seg_evidence_v8b_repro_seed43/44/45`, `seg_evidence_v8b_repro_seed43_compat`, `seg_evidence_v8b_repro_seed43_geometryfix` |

## Virtual Decision Groups

| Group | Purpose | Runs |
|---|---|---|
| `deployment_candidates` | Runs relevant to deployment decisions | `v17_focal_g2`, `v17_512_focal_g2`, `v21_512_layercam`, `v24_multitask`, `v28_no_attention`, `v30_gated_pooling`, `v31_no_se_gated`, `v31_v8b_fusion_v2`, `v31_v8b_fusion_features_hflip_v2`, `v31_v8b_fusion_quickqual_v1` |

## Run Registry

| Run | Primary Group | Secondary Groups | Config | Checkpoint Dir | Classification Artifacts | XAI Artifacts | Status |
|---|---|---|---|---|---|---|---|
| `effnet_1shot` | `00_baselines_and_early` | - | `effnetb5_1shot.yaml` | yes | external | - | completed |
| `v3` | `00_baselines_and_early` | - | - | yes | external, test | - | completed |
| `v4` | `00_baselines_and_early` | `01_ssl_lineage` | - | - | external, test | - | historical |
| `v4.1` | `01_ssl_lineage` | - | - | yes | external | - | completed |
| `v4b` | `01_ssl_lineage` | - | - | yes | external | - | completed |
| `v4b_alpha_only` | `01_ssl_lineage` | - | `v4b_alpha_only.yaml` | yes | external | - | completed |
| `v5` | `00_baselines_and_early` | - | - | yes | external | - | completed |
| `v6` | `01_ssl_lineage` | - | `v6_focal.yaml` | yes | external, test | - | completed |
| `v6_alpha_only` | `01_ssl_lineage` | - | `v6_alpha_only.yaml` | yes | external, test | - | completed |
| `v6_gamma_only` | `01_ssl_lineage` | - | `v6_gamma_only.yaml` | yes | external, test | - | completed |
| `v7_messidor_train` | `02_domain_generalization` | - | `v7_messidor_train.yaml` | yes | external | - | completed |
| `v8_mixstyle` | `02_domain_generalization` | - | `v8_mixstyle.yaml` | yes | external | - | discarded |
| `v9_fda` | `02_domain_generalization` | - | `v9_fda.yaml` | yes | external | - | completed |
| `v10_swad` | `02_domain_generalization` | - | `v10_swad.yaml` | yes | external | - | held |
| `v11_fda_swad` | `02_domain_generalization` | - | `v11_fda_swad.yaml` | yes | external | - | discarded |
| `v12_fda_imagenet` | `02_domain_generalization` | - | `v12_fda_imagenet.yaml` | yes | external | - | discarded |
| `v13_fda_swad` | `02_domain_generalization` | - | `v13_fda_swad.yaml` | yes | external | - | discarded |
| `v14_ibn` | `02_domain_generalization` | - | `v14_ibn.yaml` | yes | external | - | discarded |
| `v15_fda_a10` | `02_domain_generalization` | - | `v15_fda_a10.yaml` | yes | external | - | discarded |
| `v16_focal_g1` | `02_domain_generalization` | - | `v16_focal_g1.yaml` | yes | external | - | completed |
| `v17_focal_g2` | `02_domain_generalization` | `deployment_candidates` | `v17_focal_g2.yaml` | yes | external | - | completed |
| `v18_focal_g3` | `02_domain_generalization` | - | `v18_focal_g3.yaml` | yes | external | - | discarded |
| `v19_swad_focal_g2` | `02_domain_generalization` | - | `v19_swad_focal_g2.yaml` | yes | external | - | discarded |
| `v20_coral` | `02_domain_generalization` | - | `v20_coral.yaml` | yes | external | - | completed |
| `v7_512_messidor_train` | `03_resolution_layercam` | - | `v7_512_messidor_train.yaml` | yes | external, test | - | completed |
| `v7_512_messidor_train_contentcrop_v1` | `03_resolution_layercam` | - | `v7_512_messidor_train_contentcrop_v1.yaml` | yes | external | - | preprocessing diagnostic backbone baseline; DDR AUROC 0.7861, not promoted |
| `v7_512_messidor_train_safezoom_v1` | `03_resolution_layercam` | - | `v7_512_messidor_train_safezoom_v1.yaml` | yes | external | - | preprocessing diagnostic backbone baseline; DDR AUROC 0.8872, better than contentcrop backbone but below active lineage, not promoted |
| `v17_512_focal_g2` | `03_resolution_layercam` | `deployment_candidates` | - | yes | external, test | - | completed |
| `v21_512_focal_g2` | `03_resolution_layercam` | - | - | yes | external, test | - | completed |
| `v21_512_layercam` | `03_resolution_layercam` | `deployment_candidates` | `v21_512_layercam.yaml` | yes | external | train XAI, block sweep | completed |
| `v24_multitask` | `04_lesion_supervision` | `05_xai_attention_ablation`, `deployment_candidates` | `v24_multitask.yaml` | yes | external, test | method compare, block sweep, seg head | completed |
| `v25_multitask_l1` | `04_lesion_supervision` | - | `v25_multitask_l1.yaml` | yes | test | default XAI | discarded |
| `v26_multitask_l3` | `04_lesion_supervision` | - | `inactive/v26_multitask_l3.yaml.inactive` | no | none | none | inactive config only |
| `v27_mil_attention` | `04_lesion_supervision` | - | `v27_mil_attention.yaml` | yes | test | MIL attention XAI (test, train) | discarded for XAI |
| `v28_no_attention` | `05_xai_attention_ablation` | `deployment_candidates` | `v28_no_attention.yaml` | yes | external, test | block sweep | completed; previous deployment candidate |
| `v29_with_attention` | `05_xai_attention_ablation` | - | `v29_with_attention.yaml` | yes | external | pending | classification done; XAI pending |
| `v30_gated_pooling` | `06_xai_classifier_routing` | `deployment_candidates` | `v30_gated_pooling.yaml` | yes | external | block sweep | completed classifier-routing baseline |
| `v31_no_se_gated` | `07_lesion_evidence` | `deployment_candidates` | `v31_no_se_gated.yaml` | yes | external | train/test XAI block4, shortcut audit | previous active and current fusion base classifier; val AUROC 0.9993, DDR AUROC 0.9160; D5-D7 shortcut audit supports domain/style shortcut reliance |
| `v31_syncfix_rerun` | `07_lesion_evidence` | - | `v31_syncfix_rerun.yaml` | yes | external | IDRiD/MAPLES block4 + seg_head XAI | rerun after classifier aux_seg image/mask sync fix; DDR AUROC 0.9082 and IDRiD/MAPLES XAI regressed vs active v31 — not promoted |
| `v31_syncfix_seed43` | `07_lesion_evidence` | - | `v31_syncfix_seed43.yaml` | yes | external | IDRiD Grad-CAM/Layer-CAM | syncfix seed repeat; DDR AUROC 0.8999, XAI not promoted |
| `v31_syncfix_seed44` | `07_lesion_evidence` | - | `v31_syncfix_seed44.yaml` | yes | external | IDRiD Grad-CAM/Layer-CAM | syncfix seed repeat; DDR AUROC 0.9176 but XAI regressed and threshold shifted to 0.29 — not promoted |
| `v31_no_se_gated_contentcrop_v1` | `07_lesion_evidence` | - | `v31_no_se_gated_contentcrop_v1.yaml` | yes | external | not promotion-gated | preprocessing diagnostic; contentcrop classifier DDR AUROC 0.9091, below v31 base 0.9160 and active hflip fusion 0.9431 |
| `v31_no_se_gated_contentcrop_bbretrain_v1` | `07_lesion_evidence` | - | `v31_no_se_gated_contentcrop_bbretrain_v1.yaml` | yes | external | not promotion-gated | preprocessing diagnostic with contentcrop backbone retrain; DDR AUROC 0.8939, not promoted |
| `v31_no_se_gated_safezoom_v1` | `07_lesion_evidence` | - | `v31_no_se_gated_safezoom_v1.yaml` | yes | external | not promotion-gated | preprocessing diagnostic; safezoom classifier DDR AUROC 0.8837, not promoted |
| `v31_no_se_gated_safezoom_bbretrain_v1` | `07_lesion_evidence` | - | `v31_no_se_gated_safezoom_bbretrain_v1.yaml` | yes | external | not promotion-gated | preprocessing diagnostic with safezoom backbone retrain; DDR AUROC 0.8911, not promoted |
| `v32_lesion_seg_evidence` | `07_lesion_evidence` | - | `v32_lesion_seg_evidence.yaml` | yes | none | train XAI seg_head | completed; val AUROC 0.9992, not promoted |
| `v33_per_lesion_routing` | `07_lesion_evidence` | - | `v33_per_lesion_routing.yaml` | yes | external | test XAI block4 | completed; val AUROC 0.9980, DDR AUROC 0.9131; per-lesion routing AUC-IoU best (0.0557) but classification < v31 — not promoted |
| `v34_calibrated_routing` | `07_lesion_evidence` | - | `v34_calibrated_routing.yaml` | yes | external | test XAI block4 | completed; val AUROC 0.9989, DDR AUROC 0.9129; PG best (0.5185) but classification < v31 — not promoted |
| `v35_warmstart_routing` | `07_lesion_evidence` | - | `v35_warmstart_routing.yaml` | yes | external | test XAI block4 | completed; val AUROC 0.9992, DDR AUROC 0.9081; per-lesion routing AUPRC best (0.1537) but DDR regressed — warmstart 역효과 확인, 4ch trade-off 구조적 확정 |
| `v36_xai_multi` | `08_xai_decoder_alignment` | - | `v36_xai_multi.yaml` | yes | external | not required after DDR gate fail | discarded; DDR AUROC 0.9076 < 0.9100 gate |
| `v37_xai_multi_maples` | `08_xai_decoder_alignment` | - | `v37_xai_multi_maples.yaml` | yes | external | IDRiD/MAPLES block4 XAI | discarded; DDR AUROC 0.9103 passes minimum gate but < v31, optimal thr 0.15, IDRiD IoU20 0.0663, MAPLES AUPRC 0.0136 |
| `v37b_xai_unet_only` | `08_xai_decoder_alignment` | - | `v37b_xai_unet_only.yaml` | yes | external | IDRiD/MAPLES block4 XAI | diagnostic; removed CAM alignment. DDR AUROC 0.9200, opt thr 0.27, IDRiD IoU20 0.0816; MAPLES AUPRC 0.0161 still below v31 |
| `v37c_xai_maples_r1plus` | `08_xai_decoder_alignment` | - | `v37c_xai_maples_r1plus.yaml` | yes | external | IDRiD/MAPLES block4 XAI | diagnostic; MAPLES R0 rows excluded from mask supervision. DDR AUROC 0.9188, opt thr 0.31, but IDRiD IoU20 0.0643 and MAPLES AUPRC 0.0127 remain regressed |
| `v37b_aux03` | `08_xai_decoder_alignment` | - | `v37b_aux03.yaml` | yes | external | IDRiD/MAPLES block4 XAI | discarded; DDR AUROC 0.9203 passes, but IDRiD IoU20 0.0487 and MAPLES AUPRC 0.0094 regress |
| `v37b_aux04` | `08_xai_decoder_alignment` | - | `v37b_aux04.yaml` | yes | external | skipped by DDR gate | discarded; DDR AUROC 0.9147 but Sens@Opt 0.766 below guard |
| `v37b_aux05` | `08_xai_decoder_alignment` | - | `v37b_aux05.yaml` | yes | external | skipped by DDR gate | discarded; DDR AUROC 0.9129 but Sens@Opt 0.770 below guard |
| `v38_xai_coral` | `08_xai_decoder_alignment` | - | `v38_xai_coral.yaml` | no completed artifact | none | none | planned/blocked pending v37 diagnostics |
| `v39_unet_2stage` | `08_xai_decoder_alignment` | - | `v39_unet_2stage.yaml` | yes | external | IDRiD/MAPLES block4 + seg_head XAI | completed; frozen v37b classifier preserved DDR/XAI but did not improve Layer-CAM; seg_head direct evidence also regressed, not promoted |
| `seg_evidence_v1` | `09_evidence_segmentation` | - | `seg_evidence_v1.yaml` | yes | none | IDRiD/MAPLES segmentation eval | completed scaffold; standalone ResNet50+U-Net segmenter failed low-data baseline (best val mDice 0.00335, IDRiD test mDice 0.00129, MAPLES test mDice 0.00142), not product evidence |
| `seg_evidence_v2_focal_tversky` | `09_evidence_segmentation` | - | `seg_evidence_v2_focal_tversky.yaml` | yes | none | IDRiD/MAPLES segmentation eval + threshold sweep | completed diagnostic; synchronized image/mask transform + Focal Tversky+BCE improved over v1, but it was trained before the offline-image/raw-mask geometry fix. Aligned re-eval only: IDRiD mDice 0.0335 / union IoU 0.0886, MAPLES mDice 0.0088 / union IoU 0.0148 — not product evidence |
| `seg_evidence_v2_geomfix_retrain` | `09_evidence_segmentation` | - | `seg_evidence_v2_geomfix_retrain.yaml` | yes | none | IDRiD/MAPLES segmentation eval + threshold sweep | geometry-fix rerun of v2 conditions; best val mDice 0.0071, IDRiD best union IoU 0.0603, MAPLES best union IoU 0.0055. Geometry fix alone does not rescue the low-data v2 segmenter |
| `seg_evidence_v3_tjdr` | `09_evidence_segmentation` | - | `seg_evidence_v3_tjdr.yaml` | yes | none | IDRiD/MAPLES/TJDR segmentation eval | completed after mask-geometry fix; best val mDice 0.2482, IDRiD mDice 0.2055 / union IoU 0.2209, TJDR mDice 0.3524 / union IoU 0.3490, MAPLES mDice 0.0051 / union IoU 0.0071. Data leverage helps IDRiD/TJDR but not MAPLES generalization |
| `seg_evidence_v4_deeplab_tjdr` | `09_evidence_segmentation` | - | `seg_evidence_v4_deeplab_tjdr.yaml` | yes | none | IDRiD/MAPLES/TJDR segmentation eval + threshold sweep | completed after manual early stop; DeepLabV3-ResNet50 improved IDRiD (mDice 0.2445 / union IoU 0.2727 at threshold 0.5) but regressed TJDR and still failed MAPLES gate (best MAPLES mDice 0.0121), not promoted |
| `seg_evidence_v5_maples_fda_tjdr` | `09_evidence_segmentation` | - | `seg_evidence_v5_maples_fda_tjdr.yaml` | yes | none | IDRiD/MAPLES/TJDR segmentation eval + threshold sweep | completed with MAPLES-target FDA; IDRiD union IoU improved to 0.3068 and TJDR partly recovered, but MAPLES best mDice only 0.0141, not promoted |
| `seg_evidence_v6_maples_finetune_tjdr` | `09_evidence_segmentation` | - | `seg_evidence_v6_maples_finetune_tjdr.yaml` | yes | none | IDRiD/MAPLES/TJDR segmentation eval + threshold sweep | completed; v5 best warm-start + MAPLES-heavy domain sampling. MAPLES best mDice improved only 0.0141 -> 0.0165 and remains below gate; not promoted |
| `seg_evidence_v7_maples_only` | `09_evidence_segmentation` | - | `seg_evidence_v7_maples_only.yaml` | yes | none | MAPLES/IDRiD segmentation eval + threshold sweep | completed; MAPLES-only 122-row specialist failed MAPLES test (best mDice 0.0039, union IoU 0.0056), not promoted |
| `seg_evidence_v8_ddrseg_tjdr` | `09_evidence_segmentation` | - | `seg_evidence_v8_ddrseg_tjdr.yaml` | yes | none | IDRiD/MAPLES/TJDR/DDR_SEG segmentation eval + threshold sweep | completed; DDR_SEG added to composite masks. IDRiD/TJDR/DDR_SEG improved, but MAPLES best mDice 0.0103 / union IoU 0.0102 remains far below gate, not promoted |
| `seg_evidence_v8b_ddrseg_tjdr_maplesfix` | `09_evidence_segmentation` | - | `seg_evidence_v8b_ddrseg_tjdr_maplesfix.yaml` | yes | none | IDRiD/MAPLES/TJDR/DDR_SEG segmentation eval + threshold sweep | completed after MAPLES ROI coordinate fix; current best lesion evidence candidate. MAPLES best mDice 0.2928 / union IoU 0.2121, IDRiD best mDice 0.4151 / union IoU 0.3903 |
| `seg_evidence_v5b_maples_fda_tjdr_maplesfix` | `09_evidence_segmentation` | - | `seg_evidence_v5b_maples_fda_tjdr_maplesfix.yaml` | yes | none | IDRiD/MAPLES/TJDR segmentation eval + threshold sweep | rerun of strongest prior MAPLES-rejected candidate after ROI fix; MAPLES recovered to best mDice 0.1595 / union IoU 0.1385 but remains below v8b |
| `seg_evidence_v8b_contentcrop_v1` | `09_evidence_segmentation` | - | `seg_evidence_v8b_contentcrop_v1.yaml` | yes | none | IDRiD/MAPLES/TJDR/DDR_SEG segmentation eval + threshold sweep | preprocessing diagnostic; best val mDice 0.3288, MAPLES mDice 0.3694 improved vs v8b but IDRiD 0.2410 and DDR_SEG 0.3362 regressed, not promoted |
| `seg_evidence_v8b_safezoom_v1` | `09_evidence_segmentation` | - | `seg_evidence_v8b_safezoom_v1.yaml` | yes | none | IDRiD/MAPLES segmentation eval | preprocessing diagnostic; best val mDice 0.3514, MAPLES mDice 0.3655 improved vs v8b but IDRiD mDice 0.3085 regressed and TJDR/DDR_SEG were not evaluated, not promoted |
| `seg_evidence_v8b_swa` | `09_evidence_segmentation` | - | `seg_evidence_v8b_swa.yaml` | yes | none | IDRiD/MAPLES/TJDR/DDR_SEG segmentation eval + threshold sweep | B2 SWA diagnostic for domain-overfit mitigation; best val mDice 0.2476, IDRiD 0.3536, MAPLES 0.1753, TJDR 0.3541, DDR_SEG 0.3169. All test domains regressed vs v8b baseline, not promoted |
| `seg_evidence_v9_gin` | `09_evidence_segmentation` | - | `seg_evidence_v9_gin.yaml` | yes | none | IDRiD/MAPLES/TJDR/DDR_SEG segmentation eval + threshold sweep | B1 GIN diagnostic for domain-overfit mitigation; best val mDice 0.2252, IDRiD 0.3197, MAPLES 0.1705, TJDR 0.3471, DDR_SEG 0.3073. All test domains regressed vs v8b baseline, not promoted |
| `seg_evidence_v10_adverin` | `09_evidence_segmentation` | - | `seg_evidence_v10_adverin.yaml` | yes | none | IDRiD/MAPLES/TJDR/DDR_SEG segmentation eval + threshold sweep | B3 AdverIN diagnostic for domain-overfit mitigation; best val mDice 0.2205, IDRiD 0.3501, MAPLES 0.1875, TJDR 0.3370, DDR_SEG 0.3122. All test domains regressed vs v8b baseline, not promoted |
| `seg_evidence_v8b_repro_seed43/44/45` | `99_misc` | - | `seg_evidence_v8b_repro_seed43.yaml`, `seg_evidence_v8b_repro_seed44.yaml`, `seg_evidence_v8b_repro_seed45.yaml` | yes | none | IDRiD/MAPLES/TJDR/DDR_SEG segmentation eval + threshold sweep | Initial B-G0 current-env deterministic reproducibility gate. Python 3.14 / torch 2.9.1+cu130 / RTX 5070 Ti. Mean best val mDice 0.2522 ± 0.0448 vs old v8b 0.3388; later superseded by geometryfix diagnosis showing legacy `processed/images` alignment had regressed to contentcrop geometry |
| `seg_evidence_v8b_repro_seed43_compat` | `99_misc` | - | `seg_evidence_v8b_repro_seed43_compat.yaml` | yes | none | IDRiD/MAPLES/TJDR/DDR_SEG segmentation eval + threshold sweep | Deterministic-off compatibility control for B-G0. Best val mDice 0.1982, IDRiD 0.2959, MAPLES 0.1035, TJDR 0.3041, DDR_SEG 0.2688. It showed deterministic flags alone were not the cause, but was also affected by the same legacy geometry regression |
| `seg_evidence_v8b_repro_seed43_geometryfix` | `99_misc` | - | `seg_evidence_v8b_repro_seed43_geometryfix.yaml` | yes | none | IDRiD/MAPLES/TJDR/DDR_SEG segmentation eval + threshold sweep | Reproduction rerun after restoring `processed/images -> circular` mask/preprocess geometry. Best val mDice 0.3330 vs old v8b 0.3388; test best mDice IDRiD 0.3867, MAPLES 0.2772, TJDR 0.3923, DDR_SEG 0.3889. Confirms the initial reproduction failure was primarily a geometry regression, not environment alone |
| `v31_no_se_gated_quickqual_v1` | `99_misc` | `deployment_candidates` | `v31_no_se_gated_quickqual_v1.yaml` | yes | external_holdout | none | current active fusion classifier source. QuickQual-line holdout AUROC 0.9096, optimal threshold 0.26, Sens@Opt 0.7682, Spec@Opt 0.9134 |
| `seg_evidence_v8b_quickqual_v1` | `99_misc` | `deployment_candidates` | `seg_evidence_v8b_quickqual_v1.yaml` | yes | none | IDRiD/MAPLES/TJDR/DDR_SEG segmentation eval | current active fusion lesion-evidence source. Threshold-0.5 aggregate metrics: MAPLES mDice 0.2492 / union IoU 0.1939, TJDR mDice 0.3493 / union IoU 0.3054, DDR_SEG mDice 0.3536 / union IoU 0.2619. IDRiD mDice 0.1318 is recorded but treated as unreliable for active evidence selection |
| `v31_v8b_late_fusion_quickqual_v1` | `99_misc` | `deployment_candidates` | `v31_v8b_late_fusion_quickqual_v1.yaml` | no checkpoint | external_holdout | QuickQual-line v8b lesion evidence | previous active fusion meta source (v1, two-feature); superseded as active by the v31 collinearity refit `v31_v8b_late_fusion_quickqual_v1_v31rep` (v2) on 2026-06-03. v1 `classification_domains:late_fusion` holdout AUROC 0.9341, threshold 0.06, Sens 0.8234, Spec 0.9086 |
| `v31_dfr_v1` | `10_grounded_classifier` | - | `drscreen/cli/dfr_relearn.py` + `configs/base.yaml` | yes | external | D5/D6/D7 shortcut audit | diagnostic failed; DFR reduced D7 matched shortcut ratio but DDR AUROC fell to 0.8641 and Sens@Opt to 0.6554, so last-layer reweighting is not deployable |
| `bagnet_v1_p33_r256` | `10_grounded_classifier` | - | `grounded_bagnet_v1_p33_r256.yaml` | yes | external | not pursued after DDR hard fail | diagnostic failed; Sparse BagNet-33 at 256px reached DDR AUROC 0.6293 and Sens@Opt 0.4731, far below v31 |
| `bagnet_v1_p65_r512` | `10_grounded_classifier` | - | `grounded_bagnet_v1_p65_r512.yaml` | yes | external | IDRiD/MAPLES patch-logit evidence | diagnostic failed; Sparse BagNet-65 at 512px reached DDR AUROC 0.6552 and patch-logit evidence stayed at random/center-baseline level |
| `cbm_v1_stage1` | `10_grounded_classifier` | - | `cbm_v1_stage1.yaml` | yes | none | entropy gate | diagnostic warmup completed; normalized concept entropy 0.9983, redundant-solution gate passed |
| `cbm_v1` | `10_grounded_classifier` | - | `cbm_v1.yaml` | yes | external | IDRiD/MAPLES concept maps, seg-head XAI, D5/D6/D7 shortcut audit | diagnostic failed; DDR AUROC 0.9268 passed, but best-threshold IDRiD mDice 0.0217 and MAPLES mDice 0.0046 failed localization gates |
| `v8b_evidence_classifier_v1` | `10_grounded_classifier` | - | `v8b_evidence_classifier_v1.yaml` | no checkpoint | external | v8b lesion evidence scalar classifier | diagnostic failed; v8b lesion-map features trained on all train domains reached DDR AUROC 0.8828, below active v31 0.9160 |
| `v8b_evidence_classifier_clsdomains_v1` | `10_grounded_classifier` | - | `v8b_evidence_classifier_clsdomains_v1.yaml` | no checkpoint | external | v8b lesion evidence scalar classifier | diagnostic failed; classification-domain-only fit reached DDR AUROC 0.8479 |
| `v8b_evidence_classifier_aptos_v1` | `10_grounded_classifier` | - | `v8b_evidence_classifier_aptos_v1.yaml` | no checkpoint | external | v8b lesion evidence scalar classifier | diagnostic failed; APTOS-only fit reached DDR AUROC 0.8725 |
| `v8b_evidence_classifier_grid_v1` | `10_grounded_classifier` | - | `v8b_evidence_classifier_grid_v1.yaml` | no checkpoint | external | v8b lesion evidence scalar classifier | best G-4 diagnostic but not promoted; C-grid best reached DDR AUROC 0.8942, Sens 0.7639, Spec 0.8674, still below v31 |
| `v31_v8b_late_fusion_sweep_v1` | `10_grounded_classifier` | - | `v31_v8b_late_fusion_sweep_v1.yaml` | no checkpoint | external_holdout | v31 score + v8b lesion evidence late fusion | source sweep for deployed fusion. Formal DDR 20% calibration / 80% holdout policy selected threshold 0.38; holdout AUROC 0.9403, Acc 0.8678, Sens 0.8118, Spec 0.9238, F1 0.8599 |
| `v31_v8b_fusion_v2` | `10_grounded_classifier` | `deployment_candidates` | `v31_v8b_fusion_v2.yaml` | yes | external_holdout, latency probe, service smoke, full raw-live holdout | v8b lesion segmentation evidence | previous active deployment alias. Packages v31 classifier + v8b segmenter + numeric StandardScaler/LogReg meta-classifier; threshold 0.38. 2026-05-26 active raw-image runtime validation on 10,018 DDR holdout images reached AUROC 0.9401 with XAI errors 0. Rolled back checkpoint backup: `artifacts/checkpoints/best_pre_features_hflip_v2_20260527.pt.bak` |
| `v31_v8b_late_fusion_features_v2` | `10_grounded_classifier` | - | `v31_v8b_late_fusion_features_v2.yaml` | no checkpoint | external_holdout | v31 score + v8b extended lesion evidence features | completed staging meta retrain; append-only feature expansion reached AUROC 0.9413 at threshold 0.40 |
| `v31_v8b_fusion_features_v2` | `10_grounded_classifier` | - | `v31_v8b_fusion_features_v2.yaml` | staging | external_holdout, service smoke | v8b lesion segmentation evidence | staging checkpoint `artifacts/checkpoints/staging/v31_v8b_fusion_features_v2.pt`; AUROC 0.9413, Sens 0.8078, Spec 0.9232; gate pass but below hflip variant |
| `v31_v8b_fusion_features_hflip_v2` | `10_grounded_classifier` | `deployment_candidates` | `v31_v8b_fusion_features_hflip_v2.yaml` | yes | external_holdout, service smoke, latency probe, full raw-live holdout | v8b lesion segmentation evidence | previous active deployment alias as of 2026-05-27; same checkpoint as features_v2 plus hflip Option A meta-probability averaging. Formal holdout AUROC 0.9431, threshold 0.3931, Sens 0.8124, Spec 0.9230. Raw-live 10,018-image holdout AUROC 0.9427, Sens 0.8176, Spec 0.9176, XAI errors 0 |
| `v31_v8b_fusion_features_hflip_recalc_v2` | `10_grounded_classifier` | - | `v31_v8b_fusion_features_hflip_recalc_v2.yaml` | staging | external_holdout, service smoke | v8b lesion segmentation evidence | diagnostic only; hflip Option B recalculates fusion features from the averaged segmentation map. AUROC 0.9424, Sens 0.8122, Spec 0.9230; no-TTA보다 높지만 Option A보다 낮아 not promoted |
| `v31_v8b_late_fusion_contentcrop_v1` | `10_grounded_classifier` | - | `v31_v8b_late_fusion_contentcrop_v1.yaml` | no checkpoint | external_holdout | contentcrop v8b lesion evidence | preprocessing diagnostic; contentcrop late fusion reached AUROC 0.9280, threshold 0.23, Sens 0.8152, Spec 0.9106, below active hflip fusion 0.9431 |
| `v31_v8b_fusion_contentcrop_v1` | `10_grounded_classifier` | - | `v31_v8b_fusion_contentcrop_v1.yaml` | yes | external | contentcrop v8b lesion evidence | packaged runtime diagnostic; external AUROC 0.9091, not promoted |
| `v31_v8b_late_fusion_safezoom_v1` | `10_grounded_classifier` | - | `v31_v8b_late_fusion_safezoom_v1.yaml` | no checkpoint | external_holdout | safezoom v8b lesion evidence | preprocessing diagnostic; safezoom late fusion reached AUROC 0.9295, threshold 0.05, Sens 0.8166, Spec 0.8999. Slightly above contentcrop late fusion but below active fusion |
| `v31_v8b_fusion_safezoom_v1` | `10_grounded_classifier` | - | `v31_v8b_fusion_safezoom_v1.yaml` | checkpoint only | none | safezoom v8b lesion evidence | packaged diagnostic checkpoint exists, but the recorded decision uses `v31_v8b_late_fusion_safezoom_v1_metrics.json`; no active promotion |
| `v31_v8b_late_fusion_safezoom_bbretrain_v1` | `10_grounded_classifier` | - | `v31_v8b_late_fusion_safezoom_bbretrain_v1.yaml` | no checkpoint | external_holdout | safezoom v8b lesion evidence | preprocessing diagnostic with safezoom backbone-retrained classifier; AUROC 0.9204, threshold 0.18, Sens 0.8288, Spec 0.8921, not promoted |
| `v41_ampmix` | `10_grounded_classifier` | - | `v41_ampmix.yaml` | yes | external_holdout | IDRiD XAI, D5-D7 shortcut audit | A2 AmpMix diagnostic for domain-overfit mitigation; external_holdout AUROC 0.9027, opt threshold 0.40, Sens 0.7882, Spec 0.8496, D5 domain AUROC 0.9886. Regressed vs v31 base and active fusion, not promoted |
| `v42_coral_baseline` | `10_grounded_classifier` | - | `v42_coral_baseline.yaml` | yes | external_calibration selection, external_holdout | IDRiD XAI, D5-D7 shortcut audit | A3 CORAL diagnostic for domain-overfit mitigation; selected by external_calibration AUROC 0.9196 and reached external_holdout AUROC 0.9203. Slightly above v31 base 0.9160 but far below active hflip fusion 0.9431; D5 domain AUROC 0.9855, not promoted |
| `v42_rsc_coral` | `10_grounded_classifier` | - | `v42_rsc_coral.yaml` | yes | external_calibration selection, external_holdout | IDRiD XAI, D5-D7 shortcut audit | A3 RSC+CORAL diagnostic for domain-overfit mitigation; selected by external_calibration AUROC 0.9143 and reached external_holdout AUROC 0.9201. Slightly above v31 base but far below active fusion; D5 domain AUROC 0.9957, not promoted |

## Config / Helper Inventory

This table covers active runtime configs, helper configs, inactive configs, and completed configs that still need explicit handling notes.

| Config | Role | Current Classification |
|---|---|---|
| `base.yaml` | Active runtime/deployment config | not a run |
| `convnext_tiny_challenger.yaml` | Architecture challenger config | no completed artifact in current registry |
| `resnet50_baseline.yaml` | Architecture baseline config | no completed artifact in current registry |
| `messidor_eval.yaml` | Evaluation helper config | not a training run |
| `inactive/ssl_simclr_pretrain.yaml.inactive` | SSL pretraining config | inactive because the referenced SSL output artifact is not present |
| `inactive/v4_ssl_finetune_bce.yaml.inactive` | SSL fine-tune config | inactive because the referenced SSL backbone/checkpoint artifacts are not present |
| `inactive/v4_ssl_finetune_focal.yaml.inactive` | SSL focal fine-tune config | inactive because the referenced SSL backbone/checkpoint artifacts are not present |
| `inactive/v26_multitask_l3.yaml.inactive` | Lesion supervision λ=3.0 planned config | inactive config only; no checkpoint/evaluation artifact present |
| `v29_with_attention.yaml` | Attention-on control run | checkpoint and DDR external_test metric are stored under `05_xai_attention_ablation/v29_with_attention/`; XAI pending |
| `base_v3.yaml` | Phase 0 selection-gate base config | research helper config for domain-overfit mitigation; uses `manifest_with_val_mixed.csv` and external-calibration selection policy |
| `v41_ampmix.yaml` | A2 AmpMix classifier domain-generalization diagnostic | completed; external_holdout AUROC 0.9027 and D5 domain AUROC 0.9886, not promoted |
| `v42_coral_baseline.yaml` | A3 CORAL classifier domain-generalization diagnostic | completed; external_holdout AUROC 0.9203 and D5 domain AUROC 0.9855, not promoted |
| `v42_rsc_coral.yaml` | A3 RSC+CORAL classifier domain-generalization diagnostic | completed; external_holdout AUROC 0.9201 and D5 domain AUROC 0.9957, not promoted |
| `v8b_evidence_classifier_v1.yaml` | Phase 4-G G-4 scalar classifier over v8b lesion evidence | diagnostic completed; not promoted |
| `v8b_evidence_classifier_clsdomains_v1.yaml` | Phase 4-G G-4 classification-domain-only scalar classifier over v8b evidence | diagnostic completed; not promoted |
| `v8b_evidence_classifier_aptos_v1.yaml` | Phase 4-G G-4 APTOS-only scalar classifier over v8b evidence | diagnostic completed; not promoted |
| `v8b_evidence_classifier_grid_v1.yaml` | Phase 4-G G-4 C-grid scalar classifier over v8b evidence | best G-4 diagnostic; DDR AUROC 0.8942, below v31 |
| `v31_v8b_late_fusion_sweep_v1.yaml` | Phase 4-G G-5 v31 score + v8b lesion evidence late fusion | source sweep for `v31_v8b_fusion_v2`; formal DDR holdout AUROC 0.9403 at threshold 0.38 |
| `v31_v8b_fusion_v2.yaml` | Previous v31 + v8b score-level fusion runtime config | superseded by `v31_v8b_fusion_features_hflip_v2`; rollback checkpoint saved at `artifacts/checkpoints/best_pre_features_hflip_v2_20260527.pt.bak` |
| `v31_v8b_late_fusion_features_v2.yaml` | v31+v8b late-fusion extended-feature meta retrain | completed staging source; not active |
| `v31_v8b_fusion_features_v2.yaml` | staging runtime config for extended-feature fusion checkpoint | promotion gate pass, but lower AUROC than hflip variant |
| `v31_v8b_fusion_features_hflip_v2.yaml` | Previous active runtime config for extended-feature fusion + hflip Option A | promoted on 2026-05-27, then superseded by QuickQual active on 2026-05-29; threshold 0.3931 preserves active sensitivity in that rollback line |
| `v31_v8b_fusion_features_hflip_recalc_v2.yaml` | staging runtime config for extended-feature fusion + hflip Option B | diagnostic only; threshold 0.3852 preserves active sensitivity but AUROC is below hflip Option A |
| `v31_no_se_gated_quickqual_v1.yaml` | QuickQual-line v31 classifier source config | active fusion source classifier under `99_misc`; holdout AUROC 0.9096 |
| `seg_evidence_v8b_quickqual_v1.yaml` | QuickQual-line v8b lesion evidence source config | active fusion source segmenter under `99_misc`; MAPLES mDice 0.2492 / union IoU 0.1939 |
| `v31_v8b_late_fusion_quickqual_v1.yaml` | QuickQual-line v31+v8b late-fusion meta source config | v1 (two-feature) fusion meta source under `99_misc`; superseded as active by the v31 collinearity refit (v2) on 2026-06-03. v1 holdout AUROC 0.9341, threshold 0.06 |
| `v7_512_messidor_train_contentcrop_v1.yaml` | contentcrop preprocessing backbone diagnostic | completed; DDR AUROC 0.7861, not promoted |
| `v7_512_messidor_train_safezoom_v1.yaml` | safezoom preprocessing backbone diagnostic | completed; DDR AUROC 0.8872, not promoted |
| `v31_no_se_gated_contentcrop_v1.yaml` | contentcrop v31 classifier diagnostic | completed; DDR AUROC 0.9091, not promoted |
| `v31_no_se_gated_contentcrop_bbretrain_v1.yaml` | contentcrop v31 classifier with contentcrop backbone retrain | completed; DDR AUROC 0.8939, not promoted |
| `v31_no_se_gated_safezoom_v1.yaml` | safezoom v31 classifier diagnostic | completed; DDR AUROC 0.8837, not promoted |
| `v31_no_se_gated_safezoom_bbretrain_v1.yaml` | safezoom v31 classifier with safezoom backbone retrain | completed; DDR AUROC 0.8911, not promoted |
| `v31_v8b_late_fusion_contentcrop_v1.yaml` | contentcrop v31 + contentcrop v8b late fusion diagnostic | completed; AUROC 0.9280, below active fusion |
| `v31_v8b_fusion_contentcrop_v1.yaml` | packaged contentcrop fusion runtime diagnostic | completed; not active |
| `v31_v8b_late_fusion_safezoom_v1.yaml` | safezoom v31 + safezoom v8b late fusion diagnostic | completed; AUROC 0.9295, below active fusion |
| `v31_v8b_fusion_safezoom_v1.yaml` | packaged safezoom fusion runtime diagnostic | checkpoint exists; not active |
| `v31_v8b_late_fusion_safezoom_bbretrain_v1.yaml` | safezoom late fusion with backbone-retrained classifier | completed; AUROC 0.9204, not promoted |
| `v31_v8b_fusion_safezoom_bbretrain_v1.yaml` | packaged safezoom bbretrain fusion runtime config | config exists, but no completed runtime eval artifact found in current registry |
| `v36_xai_multi.yaml` | U-Net decoder + CAM alignment | completed; DDR gate fail |
| `v37_xai_multi_maples.yaml` | U-Net decoder + CAM alignment + MAPLES train masks | completed; calibration shift and XAI regression |
| `v37b_xai_unet_only.yaml` | v37 ablation without CAM alignment | completed; calibration recovered but MAPLES XAI still weak |
| `v37c_xai_maples_r1plus.yaml` | v37 branch A with MAPLES R1+ mask-supervised rows only | completed; R0 wiring fixed but XAI regression persists |
| `v37b_aux03.yaml` | Phase 4-D λ_aux_seg=0.3 sweep | completed; DDR pass but IDRiD/MAPLES XAI regression |
| `v37b_aux04.yaml` | Phase 4-D λ_aux_seg=0.4 sweep | completed; Sens@Opt gate fail |
| `v37b_aux05.yaml` | Phase 4-D λ_aux_seg=0.5 sweep | completed; Sens@Opt gate fail |
| `v39_unet_2stage.yaml` | frozen v37b classifier + decoder-only fallback | completed; v37b-equivalent Layer-CAM, seg_head direct evidence weak |
| `seg_evidence_v1.yaml` | standalone classifier-independent lesion segmentation evidence scaffold | completed; Python 3.14 `train_seg` run, IDRiD 54 + MAPLES 122 mask-valid rows only, failed mDice targets |
| `seg_evidence_v2_focal_tversky.yaml` | synchronized mask augmentation + Focal Tversky segmentation evidence diagnostic | completed; trained before offline-image/raw-mask geometry fix, so use aligned re-eval only as diagnostic |
| `seg_evidence_v2_geomfix_retrain.yaml` | v2 conditions rerun after mask-geometry fix | completed; same IDRiD+MAPLES R1+ manifest and Focal Tversky+BCE settings as v2, but still failed low-data segmentation targets |
| `seg_evidence_v3_tjdr.yaml` | TJDR data-leverage segmentation evidence config | completed after mask-geometry fix; uses `manifest_with_maples_tjdr_preprocessed.csv` and composite IDRiD/MAPLES/TJDR masks |
| `seg_evidence_v4_deeplab_tjdr.yaml` | Phase 4-G stronger encoder segmentation evidence config | completed after manual early stop; DeepLabV3-ResNet50 baseline, not promoted |
| `seg_evidence_v5_maples_fda_tjdr.yaml` | Phase 4-G MAPLES domain-generalization segmentation evidence config | completed; ResNet50+U-Net with MAPLES-target FDA, not promoted |
| `seg_evidence_v6_maples_finetune_tjdr.yaml` | Phase 4-G MAPLES-heavy fine-tune config | completed; v5 warm-start with MAPLES upweighting, not promoted |
| `seg_evidence_v7_maples_only.yaml` | MAPLES-only specialist segmentation diagnostic | completed; target-only 512px MAPLES training failed MAPLES test, not promoted |
| `seg_evidence_v8_ddrseg_tjdr.yaml` | Phase 4-G no-FGADR DDR segmentation data-leverage config | completed; uses `manifest_with_maples_tjdr_ddrseg_preprocessed.csv`, not promoted because MAPLES remains weak |
| `seg_evidence_v8b_ddrseg_tjdr_maplesfix.yaml` | v8 rerun after MAPLES ROI coordinate fix | completed; Phase 4-G current best standalone lesion evidence baseline and active evidence module inside `v31_v8b_fusion_v2` |
| `seg_evidence_v5b_maples_fda_tjdr_maplesfix.yaml` | v5 MAPLES-target FDA rerun after MAPLES ROI coordinate fix | completed; improves MAPLES but remains weaker than v8b |
| `seg_evidence_v8b_contentcrop_v1.yaml` | contentcrop v8b evidence segmenter diagnostic | completed; MAPLES improved but IDRiD/DDR_SEG regressed, not promoted |
| `seg_evidence_v8b_safezoom_v1.yaml` | safezoom v8b evidence segmenter diagnostic | completed; MAPLES improved but IDRiD regressed and full four-dataset eval was incomplete, not promoted |
| `seg_evidence_v8b_swa.yaml` | B2 SWA segmentation domain-generalization diagnostic | completed; all tested mDice metrics regressed vs v8b baseline, not promoted |
| `seg_evidence_v9_gin.yaml` | B1 GIN segmentation domain-generalization diagnostic | completed; all tested mDice metrics regressed vs v8b baseline, not promoted |
| `seg_evidence_v10_adverin.yaml` | B3 AdverIN segmentation domain-generalization diagnostic | completed; all tested mDice metrics regressed vs v8b baseline, not promoted |
| `grounded_bagnet_v1_p33_r256.yaml` | Phase 4-F v3 G3 Sparse BagNet-33 diagnostic | completed; DDR hard fail, not promoted |
| `grounded_bagnet_v1_p65_r512.yaml` | Phase 4-F v3 G3 Sparse BagNet-65 diagnostic | completed; DDR and localization gates failed, not promoted |
| `cbm_v1_stage1.yaml` | Phase 4-F v3 G2 CBM concept-head warmup | completed; mask-valid 188 rows, entropy gate pass |
| `cbm_v1.yaml` | Phase 4-F v3 G2 Concept Bottleneck diagnostic | completed; DDR pass but concept-map localization failed |
| `v31_no_se_gated.yaml` | True no-attention gated-pooling control | completed; removes ECA/Spatial/SE via `attention_mode: none`, Dice+BCE seg loss |
| `v31_syncfix_rerun.yaml` | v31 architecture rerun after synchronized image/mask transform fix | completed; `SegmentationManifestDataset` dry-run confirmed, but DDR/XAI regressed. v31 remains only the fusion base classifier now |
| `v32_lesion_seg_evidence.yaml` | Per-lesion segmentation evidence candidate | completed; 4-channel IDRiD MA/HE/EX/SE provider, Dice+BCE seg loss |
| `drscreen/cli/dfr_relearn.py` | Phase 4-F v3 G1 DFR diagnostic runner | completed; freezes v31 backbone/gated pooling and installs group-balanced logistic-regression weights into final classifier |
| `drscreen/models/sparse_bagnet.py` | Phase 4-F v3 G3 Sparse BagNet diagnostic model | completed; patch-logit local evidence path wired through existing training/inference/eval code |

## Log Inventory

Historical logs were migrated under the associated run's primary group.

| Path Pattern | Associated Group | Notes |
|---|---|---|
| `artifacts/runs/03_resolution_layercam/v7_512_messidor_train/logs/*` | `03_resolution_layercam` | 512px v7 train/no-crop logs |
| `artifacts/runs/03_resolution_layercam/v17_512_focal_g2/logs/*` | `03_resolution_layercam` | 512px v17 train/no-crop logs |
| `artifacts/runs/03_resolution_layercam/v21_512_layercam/logs/*` | `03_resolution_layercam` | v21 preprocessing/train/eval logs |
| `artifacts/runs/04_lesion_supervision/v24_multitask/logs/*` | `04_lesion_supervision` | v24 train/DDR/XAI log captures |

## External Test Summary

External test means the `external_test_*_best_metrics.json` artifact stored under each run's `artifacts/runs/<primary_group>/<run_id>/evaluations/` directory.

| Group | Run | Rows | AUROC | Optimal Thr | Sens@Opt | Spec@Opt |
|---|---|---:|---:|---:|---:|---:|
| `00_baselines_and_early` | `effnet_1shot` | 1200 | 0.823693 | 0.13 | 0.6544 | 0.9084 |
| `00_baselines_and_early` | `v3` | 1200 | 0.549526 | 0.94 | 0.2691 | 0.8535 |
| `00_baselines_and_early` | `v4` | 1200 | 0.760808 | 0.91 | 0.6346 | 0.7912 |
| `00_baselines_and_early` | `v5` | 1200 | 0.773423 | 0.61 | 0.5489 | 0.9212 |
| `01_ssl_lineage` | `v4.1` | 1200 | 0.801830 | 0.11 | 0.6376 | 0.8608 |
| `01_ssl_lineage` | `v4b` | 1200 | 0.678924 | 0.78 | 0.4404 | 0.8626 |
| `01_ssl_lineage` | `v4b_alpha_only` | 1200 | 0.726237 | 0.56 | 0.4801 | 0.8773 |
| `01_ssl_lineage` | `v6` | 1200 | 0.861590 | 0.59 | 0.7110 | 0.8919 |
| `01_ssl_lineage` | `v6_alpha_only` | 1200 | 0.869655 | 0.78 | 0.6667 | 0.9689 |
| `01_ssl_lineage` | `v6_gamma_only` | 1200 | 0.823071 | 0.34 | 0.6651 | 0.8993 |
| `02_domain_generalization` | `v7_messidor_train` | 12522 | 0.872454 | 0.09 | 0.7626 | 0.8417 |
| `02_domain_generalization` | `v8_mixstyle` | 12522 | 0.837086 | 0.31 | 0.6638 | 0.8797 |
| `02_domain_generalization` | `v9_fda` | 12522 | 0.882463 | 0.06 | 0.7498 | 0.8559 |
| `02_domain_generalization` | `v10_swad` | 12522 | 0.886270 | 0.05 | 0.7212 | 0.9033 |
| `02_domain_generalization` | `v11_fda_swad` | 12522 | 0.853910 | 0.31 | 0.7088 | 0.8936 |
| `02_domain_generalization` | `v12_fda_imagenet` | 12522 | 0.849791 | 0.05 | 0.6726 | 0.8886 |
| `02_domain_generalization` | `v13_fda_swad` | 12522 | 0.843620 | 0.05 | 0.6084 | 0.9443 |
| `02_domain_generalization` | `v14_ibn` | 12522 | 0.844502 | 0.08 | 0.6995 | 0.8468 |
| `02_domain_generalization` | `v15_fda_a10` | 12522 | 0.857904 | 0.05 | 0.6391 | 0.8884 |
| `02_domain_generalization` | `v16_focal_g1` | 12522 | 0.873815 | 0.18 | 0.7355 | 0.8559 |
| `02_domain_generalization` | `v17_focal_g2` | 12522 | 0.891052 | 0.42 | 0.7727 | 0.8564 |
| `02_domain_generalization` | `v18_focal_g3` | 12522 | 0.874663 | 0.29 | 0.7324 | 0.8546 |
| `02_domain_generalization` | `v19_swad_focal_g2` | 12522 | 0.883328 | 0.06 | 0.7719 | 0.8487 |
| `02_domain_generalization` | `v20_coral` | 12522 | 0.875376 | 0.20 | 0.7625 | 0.8519 |
| `03_resolution_layercam` | `v7_512_messidor_train` | 12522 | 0.904582 | 0.05 | 0.6357 | 0.9674 |
| `03_resolution_layercam` | `v17_512_focal_g2` | 12522 | 0.895191 | 0.11 | 0.7698 | 0.8680 |
| `03_resolution_layercam` | `v21_512_focal_g2` | 12522 | 0.871149 | 0.37 | 0.7327 | 0.8848 |
| `03_resolution_layercam` | `v21_512_layercam` | 12522 | 0.877540 | 0.54 | 0.7356 | 0.8982 |
| `04_lesion_supervision` | `v24_multitask` | 12522 | 0.845189 | 0.17 | 0.6742 | 0.9111 |
| `05_xai_attention_ablation` | `v28_no_attention` | 12522 | 0.892425 | 0.45 | 0.7481 | 0.9055 |
| `05_xai_attention_ablation` | `v29_with_attention` | 12522 | 0.862836 | 0.44 | 0.6985 | 0.8993 |
| `06_xai_classifier_routing` | `v30_gated_pooling` | 12522 | 0.913700 | 0.31 | 0.7840 | 0.9009 |
| `07_lesion_evidence` | `v31_no_se_gated` | 12522 | 0.916036 | 0.35 | 0.7983 | 0.8677 |
| `07_lesion_evidence` | `v31_syncfix_rerun` | 12522 | 0.908240 | 0.24 | 0.7639 | 0.8905 |
| `07_lesion_evidence` | `v31_syncfix_seed43` | 12522 | 0.899886 | 0.33 | 0.7550 | 0.8950 |
| `07_lesion_evidence` | `v31_syncfix_seed44` | 12522 | 0.917566 | 0.29 | 0.7896 | 0.9055 |
| `07_lesion_evidence` | `v33_per_lesion_routing` | 12522 | 0.913102 | 0.32 | 0.765 | 0.912 |
| `07_lesion_evidence` | `v34_calibrated_routing` | 12522 | 0.912859 | 0.51 | 0.772 | 0.908 |
| `07_lesion_evidence` | `v35_warmstart_routing` | 12522 | 0.908138 | 0.18 | 0.7932 | 0.8739 |
| `08_xai_decoder_alignment` | `v36_xai_multi` | 12522 | 0.907551 | 0.23 | 0.7574 | 0.8977 |
| `08_xai_decoder_alignment` | `v37_xai_multi_maples` | 12522 | 0.910284 | 0.15 | 0.7757 | 0.8848 |
| `08_xai_decoder_alignment` | `v37b_xai_unet_only` | 12522 | 0.919999 | 0.27 | 0.8223 | 0.8763 |
| `08_xai_decoder_alignment` | `v37c_xai_maples_r1plus` | 12522 | 0.918787 | 0.31 | 0.7826 | 0.9151 |
| `08_xai_decoder_alignment` | `v37b_aux03` | 12522 | 0.920277 | 0.41 | 0.7813 | 0.9050 |
| `08_xai_decoder_alignment` | `v37b_aux04` | 12522 | 0.914691 | 0.55 | 0.7660 | 0.9266 |
| `08_xai_decoder_alignment` | `v37b_aux05` | 12522 | 0.912937 | 0.31 | 0.7700 | 0.9122 |
| `08_xai_decoder_alignment` | `v39_unet_2stage` | 12522 | 0.919999 | 0.27 | 0.8223 | 0.8763 |
| `10_grounded_classifier` | `v31_dfr_v1` | 12522 | 0.864109 | 0.05 | 0.6554 | 0.9226 |
| `10_grounded_classifier` | `bagnet_v1_p33_r256` | 12522 | 0.629288 | 0.31 | 0.4731 | 0.7044 |
| `10_grounded_classifier` | `bagnet_v1_p65_r512` | 12522 | 0.655197 | 0.47 | 0.3950 | 0.8082 |
| `10_grounded_classifier` | `cbm_v1` | 12522 | 0.926782 | 0.21 | 0.8354 | 0.8770 |
| `10_grounded_classifier` | `v8b_evidence_classifier_grid_v1` | 12522 | 0.894188 | 0.56 | 0.7639 | 0.8674 |
| `10_grounded_classifier` | `v31_v8b_late_fusion_sweep_v1` | 10018 | 0.940255 | 0.38 | 0.8118 | 0.9238 |
| `10_grounded_classifier` | `v31_v8b_fusion_v2` | 10018 | 0.940255 | 0.38 | 0.8118 | 0.9238 |

## Internal Test Summary

| Run | Rows | AUROC | Optimal Thr | Sens@Opt | Spec@Opt |
|---|---:|---:|---:|---:|---:|
| `v17_512_focal_g2` | 469 | 0.991780 | 0.83 | 0.9449 | 0.9742 |
| `v21_512_focal_g2` | 469 | 0.991362 | 0.68 | 0.9788 | 0.9185 |
| `v24_multitask` | 469 | 0.991989 | 0.86 | 0.9237 | 0.9871 |
| `v25_multitask_l1` | 469 | 0.927093 | 0.75 | 0.9110 | 0.9356 |
| `v27_mil_attention` | 469 | 0.988216 | 0.63 | 0.9449 | 0.9528 |
| `v28_no_attention` | 469 | 0.992344 | 0.74 | 0.9449 | 0.9657 |
| `v3` | 469 | 0.972740 | - | - | - |
| `v4` | 469 | 0.982596 | 0.93 | 0.9068 | 0.9700 |
| `v6` | 469 | 0.991889 | 0.93 | 0.9449 | 0.9700 |
| `v6_alpha_only` | 469 | 0.990971 | 0.95 | 0.9788 | 0.9185 |
| `v6_gamma_only` | 469 | 0.993053 | 0.86 | 0.9449 | 0.9742 |
| `v7_512_messidor_train` | 469 | 0.989643 | 0.90 | 0.9534 | 0.9571 |

## XAI Summary (IDRiD)

All XAI rows below use IDRiD lesion masks.

| Run | Split | Target/Method | N | PG | AUPRC | AUC-IoU | IoU top-20 |
|---|---|---|---:|---:|---:|---:|---:|
| `v21_512_layercam` | train | default | 54 | 0.1111 | - | - | 0.0300 |
| `v21_512_layercam` | train | block2 | 54 | 0.1111 | - | - | 0.0271 |
| `v21_512_layercam` | train | block3 | 54 | 0.0556 | - | - | 0.0254 |
| `v21_512_layercam` | train | block4 | 54 | 0.0185 | - | - | 0.0284 |
| `v24_multitask` | test | default/block6 Layer-CAM | 27 | 0.0370 | 0.0390 | 0.0098 | 0.0321 |
| `v24_multitask` | test | block2 | 27 | 0.0000 | 0.0366 | 0.0084 | 0.0268 |
| `v24_multitask` | test | block3 | 27 | 0.0000 | 0.0351 | 0.0077 | 0.0236 |
| `v24_multitask` | test | block4 | 27 | 0.0000 | 0.0337 | 0.0044 | 0.0219 |
| `v24_multitask` | test | block5 | 27 | 0.0370 | 0.0375 | 0.0120 | 0.0298 |
| `v24_multitask` | test | seg_head | 27 | 0.0000 | - | - | 0.0318 |
| `v25_multitask_l1` | test | default | 27 | 0.0000 | - | - | 0.0254 |
| `v27_mil_attention` | test | MIL attention | 27 | 0.0000 | 0.0407 | 0.0183 | 0.0263 |
| `v28_no_attention` | test | default/block6 Layer-CAM | 27 | 0.1852 | 0.0831 | 0.0506 | 0.0627 |
| `v28_no_attention` | test | block2 | 27 | 0.4815 | 0.0979 | 0.0193 | 0.0472 |
| `v28_no_attention` | test | block3 | 27 | 0.1111 | 0.0885 | 0.0262 | 0.0585 |
| `v28_no_attention` | test | block4 | 27 | 0.4444 | 0.1253 | 0.0374 | 0.0741 |
| `v28_no_attention` | test | block5 | 27 | 0.1481 | 0.0920 | 0.0460 | 0.0651 |
| `v30_gated_pooling` | test | block4 Layer-CAM | 27 | 0.3704 | 0.1311 | 0.0443 | 0.0788 |
| `v30_gated_pooling` | test | seg_head | 27 | - | - | - | 0.0669 |
| `v27_mil_attention` | train | MIL attention | 54 | 0.0000 | 0.0286 | 0.0119 | 0.0183 |
| `v28_no_attention` | train | default Layer-CAM | 54 | 0.1111 | 0.0811 | 0.0473 | 0.0497 |
| `v31_no_se_gated` | train | block4 Layer-CAM | 54 | 0.3333 | 0.1174 | 0.0491 | 0.0601 |
| `v31_no_se_gated` | test | block4 Layer-CAM | 27 | 0.3704 | 0.1409 | 0.0496 | 0.0785 |
| `v31_no_se_gated` | test | block4 Grad-CAM | 27 | 0.2222 | 0.1404 | 0.0555 | 0.0827 |
| `v31_syncfix_rerun` | test | block4 Layer-CAM | 27 | 0.4815 | 0.1215 | 0.0394 | 0.0600 |
| `v31_syncfix_rerun` | test | block4 Grad-CAM | 27 | 0.4074 | 0.1328 | 0.0475 | 0.0629 |
| `v31_syncfix_rerun` | test | seg_head | 27 | 0.1481 | 0.0890 | 0.0412 | 0.0642 |
| `v31_syncfix_seed43` | test | block4 Grad-CAM | 27 | 0.4444 | 0.1547 | 0.0618 | 0.0613 |
| `v31_syncfix_seed43` | test | block4 Layer-CAM | 27 | 0.4444 | 0.1439 | 0.0480 | 0.0661 |
| `v31_syncfix_seed44` | test | block4 Grad-CAM | 27 | 0.2593 | 0.1142 | 0.0418 | 0.0590 |
| `v31_syncfix_seed44` | test | block4 Layer-CAM | 27 | 0.2593 | 0.1121 | 0.0348 | 0.0571 |
| `v33_per_lesion_routing` | test | block4 Layer-CAM | 27 | 0.4074 | 0.1478 | 0.0557 | 0.0799 |
| `v34_calibrated_routing` | test | block4 Layer-CAM | 27 | **0.5185** | 0.1492 | 0.0543 | 0.0769 |
| `v35_warmstart_routing` | test | block4 Layer-CAM | 27 | 0.4074 | 0.1537 | 0.0525 | 0.0796 |
| `v32_lesion_seg_evidence` | train | seg_head | 54 | 0.2222 | 0.0538 | 0.0208 | 0.0364 |
| `v37_xai_multi_maples` | test | block4 Layer-CAM | 27 | 0.3333 | 0.1230 | 0.0442 | 0.0663 |
| `v37_xai_multi_maples` | test | seg_head | 27 | 0.0370 | 0.0458 | 0.0173 | 0.0366 |
| `v37b_xai_unet_only` | test | block4 Layer-CAM | 27 | 0.3704 | **0.1546** | **0.0625** | **0.0816** |
| `v37c_xai_maples_r1plus` | test | block4 Layer-CAM | 27 | 0.2593 | 0.1179 | 0.0431 | 0.0643 |
| `v37b_aux03` | test | block4 Layer-CAM | 27 | 0.4074 | 0.0977 | 0.0313 | 0.0487 |
| `v39_unet_2stage` | test | block4 Layer-CAM | 27 | 0.3704 | **0.1546** | **0.0625** | **0.0816** |
| `v39_unet_2stage` | test | seg_head | 27 | 0.0000 | 0.0442 | 0.0158 | 0.0387 |
| `bagnet_v1_p65_r512` | test | patch logits | 27 | 0.1111 | 0.0372 | 0.0309 | 0.0262 |
| `cbm_v1` | test | seg_head/concept union | 27 | 0.1111 | 0.0583 | 0.0385 | 0.0432 |

## XAI Summary (MAPLES-DR)

Clean-cohort eval on MESSIDOR images with MAPLES-DR pixel-level annotations (MA/HE/EX/CWS).
No training data overlap. Script: `eval_xai_maples.py`.

| Run | Split | Target/Method | N | PG | AUPRC | AUC-IoU | IoU top-20 |
|---|---|---|---:|---:|---:|---:|---:|
| `v31_no_se_gated` | test | block4 Layer-CAM | 60 | 0.0500 | 0.0172 | 0.0051 | 0.0113 |
| `v31_no_se_gated` | test | block4 + OD mask | 60 | 0.0500 | 0.0173 | 0.0052 | 0.0113 |
| `v31_syncfix_rerun` | test | block4 Layer-CAM | 60 | 0.0000 | 0.0125 | 0.0031 | 0.0067 |
| `v31_syncfix_rerun` | test | seg_head | 60 | 0.0000 | 0.0102 | 0.0048 | 0.0076 |
| `v35_warmstart_routing` | test | block4 Layer-CAM | 60 | 0.0500 | 0.0166 | 0.0053 | 0.0098 |
| `v35_warmstart_routing` | test | block4 + OD mask | 60 | 0.0500 | 0.0167 | 0.0053 | 0.0099 |
| `v37_xai_multi_maples` | test | block4 Layer-CAM | 60 | 0.0167 | 0.0136 | 0.0037 | 0.0086 |
| `v37_xai_multi_maples` | test | seg_head | 60 | 0.0000 | 0.0069 | 0.0024 | 0.0052 |
| `v37b_xai_unet_only` | test | block4 Layer-CAM | 60 | 0.0000 | 0.0161 | 0.0058 | 0.0113 |
| `v37c_xai_maples_r1plus` | test | block4 Layer-CAM | 60 | 0.0000 | 0.0127 | 0.0039 | 0.0084 |
| `v37b_aux03` | test | block4 Layer-CAM | 60 | 0.0000 | 0.0094 | 0.0026 | 0.0061 |
| `v39_unet_2stage` | test | block4 Layer-CAM | 60 | 0.0000 | 0.0161 | 0.0058 | 0.0113 |
| `v39_unet_2stage` | test | seg_head | 60 | 0.0000 | 0.0069 | 0.0032 | 0.0039 |
| `bagnet_v1_p65_r512` | test | patch logits | 60 | 0.0167 | 0.0082 | 0.0053 | 0.0061 |
| `cbm_v1` | test | seg_head/concept union | 60 | 0.0000 | 0.0168 | 0.0085 | 0.0102 |

**해석**: IDRiD XAI 수치(v31: AUPRC 0.1409) 대비 약 10× 하락. v31 vs v35 차이 소멸.
IDRiD XAI 수치는 학습 도메인 편향에 의한 과대평가였음. 현 아키텍처는 외부 코호트 병변 로컬라이제이션 능력이 거의 없음.
OD masking 효과: AUPRC +0.0001 (측정 노이즈 수준) — OD는 CAM confound 아님. MAPLES-DR 저성능은 도메인 일반화 실패가 원인.

## Segmentation Evidence Summary

Standalone segmentation evidence metrics use direct model mask output rather than post-hoc CAM. Current numbers below are from aligned mask geometry evaluation, where GT masks receive the same circular-crop/pad/resize geometry as the evaluated image preprocessing.

| Run | Eval set | N | mDice | mIoU | Union Dice | Union IoU | Notes |
|---|---|---:|---:|---:|---:|---:|---|
| `seg_evidence_v2_focal_tversky` | IDRiD test | 27 | 0.0335 | 0.0186 | 0.1583 | 0.0886 | trained before mask-geometry fix; diagnostic only |
| `seg_evidence_v2_focal_tversky` | MAPLES test | 60 | 0.0088 | 0.0050 | 0.0262 | 0.0148 | trained before mask-geometry fix; diagnostic only |
| `seg_evidence_v2_geomfix_retrain` | IDRiD test | 27 | 0.0183 | 0.0095 | 0.1104 | 0.0603 | v2 conditions retrained after mask-geometry fix; best union-IoU threshold 0.50 |
| `seg_evidence_v2_geomfix_retrain` | MAPLES test | 60 | 0.0040 | 0.0020 | 0.0108 | 0.0055 | v2 conditions retrained after mask-geometry fix; best union-IoU threshold 0.45 |
| `seg_evidence_v3_tjdr` | IDRiD test | 27 | 0.2055 | 0.1317 | 0.3535 | 0.2209 | trained after mask-geometry fix |
| `seg_evidence_v3_tjdr` | MAPLES test | 60 | 0.0051 | 0.0028 | 0.0130 | 0.0071 | domain generalization still failed |
| `seg_evidence_v3_tjdr` | TJDR test | 113 | 0.3524 | 0.2713 | 0.4634 | 0.3490 | trained after mask-geometry fix |
| `seg_evidence_v4_deeplab_tjdr` | IDRiD test | 27 | 0.2445 | 0.1603 | 0.4217 | 0.2727 | DeepLabV3 stronger encoder; threshold 0.5 |
| `seg_evidence_v4_deeplab_tjdr` | MAPLES test | 60 | 0.0096 | 0.0054 | 0.0227 | 0.0126 | still below MAPLES gate |
| `seg_evidence_v4_deeplab_tjdr` | TJDR test | 113 | 0.2543 | 0.1860 | 0.3335 | 0.2358 | lower than v3 on TJDR |
| `seg_evidence_v5_maples_fda_tjdr` | IDRiD test | 27 | 0.2458 | 0.1625 | 0.4585 | 0.3068 | MAPLES-target FDA; threshold 0.5 |
| `seg_evidence_v5_maples_fda_tjdr` | MAPLES test | 60 | 0.0114 | 0.0066 | 0.0241 | 0.0133 | still below MAPLES gate |
| `seg_evidence_v5_maples_fda_tjdr` | TJDR test | 113 | 0.3108 | 0.2265 | 0.3975 | 0.2852 | recovered vs v4 but below v3 |
| `seg_evidence_v6_maples_finetune_tjdr` | IDRiD test | 27 | 0.2144 | 0.1345 | 0.4041 | 0.2574 | v5 warm-start + MAPLES-heavy sampler; threshold 0.5 |
| `seg_evidence_v6_maples_finetune_tjdr` | MAPLES test | 60 | 0.0134 | 0.0079 | 0.0308 | 0.0175 | small MAPLES gain, still below gate |
| `seg_evidence_v6_maples_finetune_tjdr` | TJDR test | 113 | 0.2816 | 0.1965 | 0.3472 | 0.2373 | regressed vs v5/v3 at threshold 0.5 |
| `seg_evidence_v7_maples_only` | IDRiD test | 27 | 0.0222 | 0.0118 | 0.0717 | 0.0382 | MAPLES-only specialist transfer; best threshold 0.5 |
| `seg_evidence_v7_maples_only` | MAPLES test | 60 | 0.0039 | 0.0020 | 0.0106 | 0.0054 | target-only training still failed MAPLES; best mDice threshold 0.4 |
| `seg_evidence_v8_ddrseg_tjdr` | IDRiD test | 27 | 0.3154 | 0.2088 | 0.4925 | 0.3324 | DDR_SEG data leverage; threshold 0.5 |
| `seg_evidence_v8_ddrseg_tjdr` | MAPLES test | 60 | 0.0086 | 0.0048 | 0.0151 | 0.0081 | still below MAPLES gate; threshold 0.5 |
| `seg_evidence_v8_ddrseg_tjdr` | TJDR test | 113 | 0.3633 | 0.2756 | 0.4314 | 0.3200 | improved vs v3/v5/v6 at threshold 0.5 |
| `seg_evidence_v8_ddrseg_tjdr` | DDR_SEG test | 225 | 0.3513 | 0.2380 | 0.4094 | 0.2724 | DDR lesion-segmentation held-out test; threshold 0.5 |
| `seg_evidence_v8b_ddrseg_tjdr_maplesfix` | IDRiD test | 27 | 0.4145 | 0.2828 | 0.5535 | 0.3893 | MAPLES ROI fix rerun; threshold 0.5 |
| `seg_evidence_v8b_ddrseg_tjdr_maplesfix` | MAPLES test | 60 | 0.2798 | 0.1871 | 0.3065 | 0.1993 | MAPLES ROI fix rerun; threshold 0.5 |
| `seg_evidence_v8b_ddrseg_tjdr_maplesfix` | TJDR test | 113 | 0.3759 | 0.2813 | 0.4266 | 0.3129 | MAPLES ROI fix rerun; threshold 0.5 |
| `seg_evidence_v8b_ddrseg_tjdr_maplesfix` | DDR_SEG test | 225 | 0.3932 | 0.2665 | 0.4244 | 0.2838 | MAPLES ROI fix rerun; threshold 0.5 |
| `seg_evidence_v5b_maples_fda_tjdr_maplesfix` | IDRiD test | 27 | 0.2703 | 0.1848 | 0.4940 | 0.3280 | v5 MAPLES-target FDA rerun; threshold 0.5 |
| `seg_evidence_v5b_maples_fda_tjdr_maplesfix` | MAPLES test | 60 | 0.0996 | 0.0675 | 0.1626 | 0.1049 | v5 MAPLES-target FDA rerun; threshold 0.5 |
| `seg_evidence_v5b_maples_fda_tjdr_maplesfix` | TJDR test | 113 | 0.3307 | 0.2465 | 0.4459 | 0.3308 | v5 MAPLES-target FDA rerun; threshold 0.5 |

Threshold sweep for `seg_evidence_v3_tjdr`: IDRiD best threshold 0.05 (mDice 0.2419 / union IoU 0.2674), TJDR best union-IoU threshold 0.50 (mDice 0.3524 / union IoU 0.3490), MAPLES best threshold 0.05 but still weak (mDice 0.0070 / union IoU 0.0091). This keeps MAPLES failure classified as domain/representation generalization, not threshold calibration.

Threshold sweep for `seg_evidence_v4_deeplab_tjdr`: IDRiD best mDice threshold 0.25 (mDice 0.2460 / union IoU 0.2736), IDRiD best union-IoU threshold 0.35 (mDice 0.2456 / union IoU 0.2739), TJDR best union-IoU threshold 0.65 (mDice 0.2535 / union IoU 0.2364), MAPLES best threshold 0.05 but still weak (mDice 0.0121 / union IoU 0.0159). DeepLabV3 improves IDRiD but does not solve MAPLES generalization and is worse than v3 on TJDR.

Threshold sweep for `seg_evidence_v5_maples_fda_tjdr`: IDRiD best union-IoU threshold 0.50 (mDice 0.2458 / union IoU 0.3068), TJDR best union-IoU threshold 0.75 (mDice 0.3126 / union IoU 0.2962), MAPLES best threshold 0.05 but still weak (mDice 0.0141 / union IoU 0.0183). FDA improves IDRiD/TJDR relative to v4 but does not solve MAPLES.

Threshold sweep for `seg_evidence_v6_maples_finetune_tjdr`: IDRiD best union-IoU threshold 0.95 (mDice 0.2450 / union IoU 0.3033), TJDR best union-IoU threshold 0.95 (mDice 0.3098 / union IoU 0.2933), MAPLES best threshold 0.05 but still weak (mDice 0.0165 / union IoU 0.0201). MAPLES-heavy fine-tune produces only a small MAPLES gain and slightly regresses IDRiD/TJDR versus v5.

Threshold sweep for `seg_evidence_v7_maples_only`: MAPLES best mDice threshold 0.40 (mDice 0.0039 / union IoU 0.0054), MAPLES best union-IoU threshold 0.50 (mDice 0.0035 / union IoU 0.0056), IDRiD best threshold 0.50 (mDice 0.0222 / union IoU 0.0382). Target-only MAPLES fine-tuning on the current 512px preprocessed data does not solve MAPLES lesion evidence.

Threshold sweep for `seg_evidence_v2_geomfix_retrain`: IDRiD best mDice threshold 0.40 (mDice 0.0257 / union IoU 0.0352), IDRiD best union-IoU threshold 0.50 (mDice 0.0183 / union IoU 0.0603), MAPLES best mDice threshold 0.40 (mDice 0.0040 / union IoU 0.0054), MAPLES best union-IoU threshold 0.45 (mDice 0.0040 / union IoU 0.0055). This rerun confirms that the v2 low-data segmenter remains weak even after mask-geometry correction.

Threshold sweep for `seg_evidence_v8_ddrseg_tjdr`: IDRiD best mDice threshold 0.20 (mDice 0.3182 / union IoU 0.3383), IDRiD best union-IoU threshold 0.15 (union IoU 0.3386); TJDR best mDice threshold 0.10 (mDice 0.3679), TJDR best union-IoU threshold 0.50 (union IoU 0.3200); DDR_SEG best mDice threshold 0.30 (mDice 0.3523), DDR_SEG best union-IoU threshold 0.55 (union IoU 0.2724); MAPLES best threshold 0.05 (mDice 0.0103 / union IoU 0.0102). DDR segmentation improves in-domain and IDRiD/TJDR evidence but does not solve MAPLES generalization.

Threshold sweep for `seg_evidence_v8b_ddrseg_tjdr_maplesfix`: IDRiD best mDice/union-IoU threshold 0.40 (mDice 0.4151 / union IoU 0.3903); MAPLES best mDice threshold 0.20 (mDice 0.2928), best union-IoU threshold 0.10 (union IoU 0.2121); TJDR best mDice threshold 0.40 (mDice 0.3788), best union-IoU threshold 0.75 (union IoU 0.3149); DDR_SEG best mDice threshold 0.55 (mDice 0.3945), best union-IoU threshold 0.75 (union IoU 0.2880). This is the current strongest standalone lesion evidence run.

Threshold sweep for `seg_evidence_v5b_maples_fda_tjdr_maplesfix`: IDRiD best mDice/union-IoU threshold 0.05 (mDice 0.2990 / union IoU 0.3426); MAPLES best threshold 0.05 (mDice 0.1595 / union IoU 0.1385); TJDR best mDice threshold 0.05 (mDice 0.3535), best union-IoU threshold 0.45 (union IoU 0.3315). It confirms the MAPLES ROI fix matters, but v5b remains below v8b.

**Mask-geometry caveat**: Earlier pixel-mask-supervised conclusions for decoder/seg_head/standalone segmentation runs are confounded unless retrained after the geometry fix. A second, MAPLES-specific caveat was found on 2026-05-21: MAPLES annotation PNGs are ROI-space masks and must be restored through `MESSIDOR-ROIs.csv`. Therefore MAPLES metrics for pre-ROI-fix segmentation runs (`seg_evidence_v3` through original `v8`) are diagnostic only. The v31 base classifier path is not affected because it is image-only classification + CAM. The current active `v31_v8b_fusion_quickqual_v2` deployment uses the QuickQual-line `seg_evidence_v8b_quickqual_v1` evidence module; the previous hflip/circular active used the post-fix `seg_evidence_v8b_ddrseg_tjdr_maplesfix` evidence module.

## XAI Method Comparison on v24

| Method | AUPRC | AUC-IoU | IoU top-20 | PG |
|---|---:|---:|---:|---:|
| Grad-CAM | 0.0380 | 0.0104 | 0.0210 | 0.1111 |
| Layer-CAM | 0.0390 | 0.0098 | 0.0321 | 0.0370 |
| Grad-CAM++ | 0.0364 | 0.0113 | 0.0316 | 0.0370 |
| Integrated Gradients | 0.0399 | 0.0039 | 0.0317 | 0.0370 |
| Score-CAM | 0.0382 | 0.0300 | 0.0265 | 0.0741 |
| Center Gaussian baseline | 0.0526 | - | 0.0436 | - |
| Random baseline | 0.0348 | - | 0.0282 | - |

## Current Interpretation

- **`v31_v8b_fusion_quickqual_v2` is the active deployment alias** in `configs/base.yaml` (v31 collinearity refit, promoted 2026-06-03), and `artifacts/checkpoints/best.pt` currently contains the composite QuickQual-line v31 classifier + QuickQual-line v8b segmenter + numeric meta-classifier checkpoint with a single `v31_logit` feature (88 features total). Formal DDR 20% calibration / 80% holdout metrics: AUROC **0.9360**, threshold **0.08563**, Sensitivity **0.8316**, Specificity **0.9070**, Accuracy **0.8693**, F1 **0.8641**. The immediate rollback is the two-feature `v31_v8b_fusion_quickqual_v1` (AUROC 0.9341, threshold 0.06), backed up at `artifacts/checkpoints/best_pre_collinearity_refit_20260603.pt.bak`.
- **2026-06-03 v2 collinearity refit promotion**: the meta-classifier dropped the redundant `v31_probability` column (`sigmoid(v31_logit)`, near-collinear) and now uses a single `v31_logit` feature plus the v8b lesion features (89 -> 88 features), which stabilized the coefficients and weakly improved AUROC. `v31_v8b_fusion_quickqual_v2` was promoted to active with a calibration-matched threshold **0.08563** to preserve the previous active sensitivity (0.8234). Source metrics: `artifacts/runs/99_misc/v31_v8b_late_fusion_quickqual_v1_v31rep/evaluations/v31_v8b_late_fusion_quickqual_v1_v31rep_metrics.json`, key `classification_domains:late_fusion:v31_logit`.
- **2026-05-29 QuickQual promotion**: `v31_v8b_fusion_quickqual_v1` aligns active AI runtime with backend QuickQual geometry. `configs/base.yaml` uses `infer.preprocess_mode: none`, `infer.tta_mode: none`, and `infer.threshold: 0.06`. Previous active `v31_v8b_fusion_features_hflip_v2` is preserved at `artifacts/checkpoints/best_pre_quickqual_v1_20260529.pt.bak`; older `v31_v8b_fusion_v2` is preserved at `artifacts/checkpoints/best_pre_features_hflip_v2_20260527.pt.bak`.
- **2026-05-31 fusion complementarity diagnostic**: `.omc/plans/fusion_complementarity_plan.md` Phase 0/Phase C were executed against the QuickQual active line. Phase 0 (`.omc/research/fusion_complementarity/phase0_power_ceiling.json`) showed low complementarity between v31 and v8b signals: Q-statistic 0.888, v8b correction of v31 errors 37%, both-wrong 1,326 on holdout. Phase C (`.omc/research/fusion_complementarity/phase_c_calfit_ablation.json`) showed unweighted calibration-fit (`calfit_none`) improves holdout AUROC over the active train-fit policy (0.940840 vs 0.934129; paired CI +0.00417 to +0.00923), but sensitivity drops from 0.8234 to 0.8010 and all residual-weighting variants underperform `calfit_none`. Therefore residual complementarity failed, B/A tracks stay on hold, and the then-active deployment remained `v31_v8b_fusion_quickqual_v1`.
- **2026-05-24 preprocessing result**: safezoom/contentcrop preprocessing variants are diagnostic only. Safezoom improved foreground preservation and reached late-fusion AUROC **0.9295**, slightly above contentcrop late fusion **0.9280**, but both were below the then-active circular fusion **0.9403**. Backbone-retrained safezoom late fusion fell to **0.9204**. Segmenters improved MAPLES mDice but regressed IDRiD and/or DDR_SEG, so those preprocessing variants were not promoted.
- **2026-05-25 domain-overfit mitigation result**: no active or staging candidate changed. `v42_coral_baseline` and `v42_rsc_coral` slightly improved over v31 base classifier AUROC 0.9160 (0.9203/0.9201) but remained far below active fusion AUROC 0.9403 and failed D5 domain-shortcut probes. `v41_ampmix` regressed to AUROC 0.9027. Segmentation DG runs `seg_evidence_v8b_swa`, `seg_evidence_v9_gin`, and `seg_evidence_v10_adverin` all regressed against `seg_evidence_v8b_ddrseg_tjdr_maplesfix`; therefore `fusion_v3` was not created.
- **2026-05-26 current-env v8b reproducibility result**: B-G0 reproduced v8b in the current Python 3.14 / torch 2.9.1+cu130 / RTX 5070 Ti stack with deterministic seed 43/44/45 configs. Old v8b `best_val_mdice=0.3388` was not reproduced: deterministic repeats averaged val mDice **0.2522 ± 0.0448**, IDRiD **0.3392 ± 0.0367**, MAPLES **0.1993 ± 0.0783**, TJDR **0.3389 ± 0.0221**, DDR_SEG **0.3069 ± 0.0317**. A deterministic-off seed43 control also stayed low (val **0.1982**, MAPLES **0.1035**), so deterministic settings alone do not explain the reproduction failure. Segmentation canonicalization experiments should not be promoted against the old v8b baseline until the current-env baseline is re-established or the original training stack is recovered.
- `v28_no_attention` is a previous deployment candidate and remains in the registry for attention-ablation/block-sweep comparison.
- `v30_gated_pooling` DDR AUROC 0.9137, test XAI AUPRC 0.1311 — v31이 DDR AUROC, AUPRC, AUC-IoU에서 앞서며 PG는 동일(0.3704), IoU top-20은 v30 0.0788 vs v31 0.0785로 사실상 동률이다. classifier-routing 설계 baseline으로만 유지.
- block4 lesion gate를 classifier pooling 경로에 곱하는 방식이 분류와 XAI 지표 모두 개선 — "분류 경로에 병변 위치 신호를 묶으면 XAI 정렬이 개선되는가" 가설에 긍정적.
- `v32_lesion_seg_evidence` (train split, seg_head 직접 출력): PG 0.2222, AUPRC 0.0538, AUC-IoU 0.0208. v32 artifact는 seg_head train 평가만 존재하고 제품 XAI 후보로 보기 어렵다. 현재 코드의 4채널 gated classifier는 per-lesion sigmoid + softmax weighted sum을 사용하며, 단일 evidence map 생성용 `predict_seg_union()`만 `amax(dim=1)` union을 사용한다.
- `v27_mil_attention` XAI 결과 (train): AUC-IoU 0.0119 — random baseline(0.0260) 이하. MIL attention은 spatial localization에 구조적으로 부적합. 방향 폐기.
- `v29_with_attention` XAI artifacts 미생성. 분류 AUROC 0.8629로 v28(0.8924) 대비 열위. ECA+CBAM spatial attention이 분류·XAI 모두 악화시킴. 방향 폐기.
- **CAM Phase-0 gate 전 모델 FAIL**: test split 기준 center_gaussian+2σ threshold=0.1089. CAM/Layer-CAM 계열 모델의 AUC-IoU는 임계값 미달이었다. 이 결론은 CAM research path에 대한 것이며, 현재 active evidence는 별도 v8b lesion segmentation metric(`xai_auc_iou`, `xai_seg_mdice`, `xai_seg_union_iou`)로 평가한다.
- `use_attention=false`는 ECA channel module을 유지하는 legacy 기준. `attention_mode: none`을 쓰는 v31만 SE/ECA/Spatial 계열을 IdentitySE로 대체하는 true no-attention 대조군.
- IDRiD contamination 주의: XAI eval은 `A. Segmentation` 이미지 사용, 분류 학습은 `B. Disease Grading` 이미지 사용 — 파일은 다르나 동일 환자 포함(patient-level overlap). file-level contamination은 없음.
- `v33_per_lesion_routing` (test split): PG 0.4074, AUPRC 0.1478, AUC-IoU 0.0557 — per-lesion routing 계열에서 AUC-IoU 기준 최고였다. per-lesion 독립 sigmoid + learnable weighted sum(softmax) 구조가 XAI 품질을 개선했음을 확인. 단, DDR AUROC 0.9131 < v31 0.9160으로 분류 소폭 하락해 당시 배포 미승격.
- **MAPLES-DR 확보 완료**: `data/raw/MAPLES-DR/AdditionalData/` (train 138 / test 60장, 12종 biomarker). `MAPLESMaskProvider` 구현 완료 (`drscreen/data/mask_providers.py`) 범위는 MA/HE/EX/CWS pathology mask 로딩이다. `eval_xai_iou.py --mask-provider maples`로 clean-cohort CAM 평가는 가능하다. 단, Phase 1 anatomy audit용 anatomy/lesion attribution ratio 지표는 아직 구현되지 않았다.
- `v34_calibrated_routing` (test split): PG **0.5185**, AUPRC 0.1492 — PG 기준 최고. lambda_aux_seg=0.3 변경으로 XAI 소폭 개선. 그러나 DDR AUROC 0.9129 < v31 0.9160, 분류 기준 미달로 당시 배포 미승격.
- `v35_warmstart_routing` (external_test): DDR AUROC 0.9081, optimal thr 0.18, Sens 0.7932, Spec 0.8739. test XAI AUPRC 0.1537 — per-lesion routing 계열 AUPRC 기준 최고였으나 AUC-IoU 0.0525, PG 0.4074로 v33/v34 대비 지표별 우위가 갈린다. 분류 회귀로 배포 미승격.
- **4ch per-lesion routing 구조 trade-off 최종 확정 (v33~v35)**: lambda 조정(v34), v31 warmstart(v35) 모두 DDR 회귀 미해소. v35 warmstart는 오히려 DDR AUROC 0.9081로 최저 — 4ch routing이 OOD 일반화를 구조적으로 희생. XAI 개선(AUPRC ↑)과 DDR 일반화(AUROC ↓)는 현 아키텍처에서 trade-off 관계.
- **실험 방향 전환**: 4ch per-lesion routing 추가 실험 중단. 당시 배포(v31)는 분류 최우선 기준으로 유지했고, 이후 Sprint 5에서 v31+v8b score-level fusion을 active deployment로 승격했다. XAI 개선은 분류에 영향 없는 방법 탐색으로 전환했다.
- **MAPLES-DR clean-cohort 확인 완료**: v31/v35 모두 PG 0.0500, AUPRC ~0.017, AUC-IoU ~0.005 — IDRiD 수치 대비 10× 하락. IDRiD XAI 수치는 학습 도메인 편향 과대평가. 현 아키텍처의 XAI 일반화 능력 부재 확인.
- **Anatomy-guided CAM masking 효과 없음**: OD 마스킹 후 AUPRC +0.0001 (노이즈 수준). OD가 CAM confound가 아님을 확인. MAPLES-DR 저성능은 도메인 일반화 실패가 근본 원인.
- **Phase 4 lesion-routing/CAM 실험 완료**: v31~v35 + MAPLES-DR clean-cohort + OD masking. XAI 개선을 위해서는 도메인 불변 feature 학습 등 아키텍처 수준 접근 필요. 당시에는 v31을 배포 상태로 유지했다.
- **v36/v37 decoder-alignment 실험 기록**: v36은 DDR AUROC 0.9076으로 폐기. v37은 MAPLES-inclusive manifest + U-Net aux decoder + CAM alignment로 학습했지만 DDR AUROC 0.9103으로 v31(0.9160) 미달, optimal threshold 0.15로 calibration이 크게 이동했다. IDRiD test XAI도 PG 0.3333, AUPRC 0.1230, AUC-IoU 0.0442, IoU top-20 0.0663으로 v31 baseline 미달. MAPLES test는 PG 0.0167, AUPRC 0.0136, AUC-IoU 0.0037, IoU top-20 0.0086으로 MAPLES train mask 추가 효과가 확인되지 않았다.
- **Phase 4-C 진단 완료**: D1에서 기존 MAPLES train manifest는 R0 12장이 `domain=MAPLES`, `valid=True`, union pixel mean 0.0000985로 들어가 빈 mask supervision을 제공함을 확인했다. D2에서 v37 seg_head 직접 출력은 IDRiD IoU top-20 0.0366, MAPLES AUPRC 0.0069로 decoder evidence 자체가 약했다. v37b(`lambda_cam_align=0`)는 DDR AUROC 0.9200, threshold 0.27, IDRiD IoU top-20 0.0816으로 회복했으나 MAPLES AUPRC 0.0161로 v31(0.0172) 미달. v37c(R1+ only MAPLES supervision)는 DDR AUROC 0.9188, threshold 0.31이지만 IDRiD IoU top-20 0.0643, MAPLES AUPRC 0.0127로 실패. 결론: R0 필터는 필요하지만 충분하지 않고, CAM alignment는 제거 대상이다.
- **Phase 4-D 완료**: `v37b_aux03/04/05` λ_aux_seg sweep과 `v39_unet_2stage` fallback을 실행했다. aux03은 DDR AUROC 0.9203으로 통과했지만 IDRiD IoU top-20 0.0487, MAPLES AUPRC 0.0094로 XAI가 회귀했다. aux04/aux05는 Sens@Opt 0.766/0.770으로 gate fail. v39는 DDR AUROC 0.9200, IDRiD IoU top-20 0.0816, MAPLES AUPRC 0.0161로 v37b와 동등하지만 개선은 없었다. decoder-only 학습은 freeze sanity check로는 유효했으나, `use_gated_pooling=false` 구조에서 seg_head가 classifier logit path에 연결되지 않아 Layer-CAM 개선 수단으로는 부적합하다. v39 seg_head 직접 출력도 IDRiD IoU top-20 0.0387, MAPLES AUPRC 0.0069로 낮아 decoder-as-evidence 분기도 폐기한다. 당시에는 v31을 배포 상태로 유지했다.
- **Phase 4-E Track 1 완료**: `occlusion`/`rise` perturbation attribution과 deletion/insertion faithfulness metric을 추가했다. v31 Occlusion grid16은 IDRiD test에서 AUPRC 0.0832, AUC-IoU 0.0498, IoU top-20 0.0588, PG 0.1481로 Layer-CAM block4보다 병변 정렬이 낮았다. MAPLES test도 AUPRC 0.0172, IoU top-20 0.0103으로 localization PASS 미달. 단 deletion AUC는 Occlusion 0.5971 vs Layer-CAM 0.7107로 Occlusion이 classifier confidence에 더 직접적인 영역을 찾았다. 결론은 **FAITHFULNESS_ONLY**: 평가 도구로 유지하되 제품 XAI/evidence로 승격하지 않는다. 다음은 독립 lesion segmentation evidence track.
- **Phase 4-E root-cause update**: Track 1 결과는 "attribution method만 약하다"보다 "classifier가 병변 위치가 아닌 shortcut feature에 의존한다"는 해석과 더 잘 맞는다. D5 domain probe, D6 lesion presence probe, D7 counterfactual style swap을 Phase 4-E plan에 추가했다. 제품 evidence는 classifier의 인과 설명이 아니라 별도 lesion candidate overlay로 정의한다.
- **Phase 4-E Track 2 scaffold 결과와 정정**: `seg_evidence_v1`은 classifier-independent ResNet50+U-Net 4ch segmenter scaffold다. 당시 로컬 mask-valid 데이터는 IDRiD 54 + MAPLES 122뿐이었다. 이후 점검에서 segmentation train/eval의 image-mask sync 문제와 offline-preprocessed image/raw mask geometry mismatch가 확인됐다. 따라서 v1/v2 및 v36~v39 mask-supervised 실패 해석은 confounded로 표시한다. `seg_evidence_v2_focal_tversky`는 synchronized transform은 고쳤지만 mask-geometry fix 전 학습물이므로, aligned re-eval 결과(IDRiD mDice 0.0335 / union IoU 0.0886, MAPLES mDice 0.0088 / union IoU 0.0148)만 diagnostic으로 사용한다.
- **Phase 4-F Step 0**: 후속 기본 경로는 encoder-first + data-access gate로 정리됐으나, 이후 grounded-classifier 진단과 data/representation leverage 흐름으로 재구성됐다. H13 shortcut이 SUPPORTED이고, segmentation evidence는 IDRiD/TJDR에서는 개선 가능하지만 MAPLES 일반화가 약하기 때문에 lesion-mask data access와 fundus/SAM-style encoder probe를 병행 검토했다. `.omc/research/phase4f_data_access.json` 기준 당시 FGADR/TJDR/RETFound/SAM은 로컬에 없었고, 이후 TJDR은 로컬 다운로드 및 provider 통합까지 완료됐다. 2026-05-21 기준 FGADR는 접근 절차 부담으로 active path에서 제외하고, DDR segmentation subset/Retinal-Lesions 등 대체 후보를 우선한다. Phase 4-F의 target mask class는 MA/HE/EX/SE 4채널이다.
- **Phase 4-F v3 S0 grounded-classifier prep**: 방향을 독립 segmenter-first에서 grounded classifier로 전환했다. `drscreen/cli/diagnose_v31_lesion_probe.py`와 `data/processed/lesion_concept_labels.csv`를 추가했고, `.omc/research/phase4f_v3_d12_v31_probe.json`에 D12 결과를 기록했다. D12-A IDRiD AUROC 0.9977, D12-B MAPLES+fallback AUROC 0.8965, D12-U pooled AUROC 0.9495. 단, D12-B full은 5 native MAPLES no-lesion + 115 Messidor fallback 정상으로 구성되므로 pure MAPLES decodability로 해석하면 안 된다. v31 DDR regression guard는 기존 active metric과 일치했다.
- **Phase 4-F v3 G1 DFR diagnostic**: `v31_dfr_v1`은 v31 backbone/gated pooling을 동결하고 최종 classifier만 4-group balanced set으로 재학습했다. Training set AUROC는 0.9984였지만 DDR external_test는 AUROC 0.8641, optimal threshold 0.05, Sens@Opt 0.6554로 gate fail. D7 matched non-lesion/lesion ratio는 1.4752x에서 0.8720x로 개선됐으나, D5 domain AUROC 0.9681과 D6 MAPLES lesion AUROC 0.4048은 변하지 않았다. 결론: last-layer reweighting만으로 shortcut-free classifier를 만들 수 없었다.
- **Phase 4-F v3 G3 Sparse BagNet diagnostic**: `sparse_bagnet` architecture와 `grounded_classifier` evidence path를 추가하고 `bagnet_v1_p33_r256`, `bagnet_v1_p65_r512`를 학습했다. p33은 DDR AUROC 0.6293, Sens@Opt 0.4731로 hard fail. p65는 DDR AUROC 0.6552, Sens@Opt 0.3950으로 p33보다 낫지만 v31(0.9160)과 비교 불가한 수준이다. p65 patch-logit evidence도 IDRiD IoU top-20 0.0262, MAPLES IoU top-20 0.0061로 center/random baseline 수준이다. 결론: receptive-field constraint만으로 shortcut-free DR classifier를 만들 수 없고, G3는 product/deployment 후보가 아니다. G2 CBM이 Phase 4-F의 마지막 grounded-classifier track이며, 실패 시 Phase 4-G는 더 강한 lesion-supervised/fundus-pretrained representation 중심으로 재설계해야 한다.
- **Phase 4-F v3 G2 CBM diagnostic**: `concept_bottleneck` architecture를 추가하고 `cbm_v1_stage1`, `cbm_v1`을 학습했다. Stage1 entropy gate는 통과했지만, 최종 CBM은 DDR AUROC 0.9268에도 concept-map localization이 실패했다(0.1~0.5 threshold sweep best: IDRiD mDice 0.0217, MAPLES mDice 0.0046). IDRiD seg-head IoU top-20도 0.0432로 center Gaussian 0.0436과 사실상 동일했다. D7 matched ratio는 1.1913x로 v31보다 개선됐으나 D5 domain AUROC 0.9870이 남아 있고, 병변 위치 정렬이 없으므로 product evidence 후보가 아니다.
- **Phase 4-F selection**: `.omc/research/phase4f_v3_selection.json` 기준 G1/G2/G3 모두 product gates를 통과하지 못했다. v31 active deployment 유지. 다음은 Phase 4-G(data/representation leverage)로 전환한다.
- **Phase 4-G G-1/G-2 TJDR integration and aligned retrain**: `.omc/research/phase4g_data_access_gate.json`은 초기 상태(TJDR/FGADR/RETFound/SAM/MedSAM 로컬 없음)를 기록한다. 이후 TJDR은 `data/raw/TJDR`에 확보 완료됐고, 최신 구조 감사 기준 `train/image` 448장, `train/annotation` 448장, `test/image` 113장, `test/annotation` 113장으로 총 561쌍이 1:1 매칭된다. `TJDRMaskProvider`와 `build_manifest --include-tjdr`를 구현했고, `preprocess_images.py`로 `manifest_with_maples_tjdr_preprocessed.csv`를 생성했다. 추가로 mask provider가 preprocessed image row에 대해 raw image 기준 circular-crop/pad/resize geometry를 mask에도 적용하도록 수정했다. `seg_evidence_v3_tjdr` aligned retrain 결과: best val mDice 0.2482, IDRiD test mDice 0.2055 / union IoU 0.2209, TJDR test mDice 0.3524 / union IoU 0.3490, MAPLES test mDice 0.0051 / union IoU 0.0071. Threshold sweep에서도 MAPLES best가 mDice 0.0070 / union IoU 0.0091에 그쳐 threshold 문제가 아님을 확인했다.
- **Phase 4-G G-2 stronger encoder baseline**: `seg_evidence_v4_deeplab_tjdr`는 DeepLabV3-ResNet50 baseline이다. epoch 11 best val mDice 0.1506 이후 epoch 36까지 best가 갱신되지 않아 수동 조기종료했고, 이후 `train_seg`에 `early_stopping_patience/min_delta`를 추가했다. Threshold 0.5 aligned eval: IDRiD mDice 0.2445 / union IoU 0.2727, MAPLES mDice 0.0096 / union IoU 0.0126, TJDR mDice 0.2543 / union IoU 0.2358. Threshold sweep에서도 MAPLES best는 mDice 0.0121 / union IoU 0.0159로 gate 0.05 미달. DeepLabV3는 IDRiD를 개선했지만 MAPLES 일반화와 TJDR 성능을 해결하지 못해 promotion 대상이 아니다.
- **Phase 4-G MAPLES-target FDA diagnostic**: `SegmentationFDAManifestDataset`와 `seg_evidence_v5_maples_fda_tjdr`를 추가했다. 학습 시 non-MAPLES samples를 MAPLES reference image의 low-frequency Fourier amplitude와 섞는다(`fda_target_domain=MAPLES`, `fda_probability=0.8`, `fda_alpha=0.05`). Best epoch 21, val mDice 0.2269, early stop epoch 29. Threshold 0.5 aligned eval: IDRiD mDice 0.2458 / union IoU 0.3068, MAPLES mDice 0.0114 / union IoU 0.0133, TJDR mDice 0.3108 / union IoU 0.2852. Threshold sweep MAPLES best는 mDice 0.0141 / union IoU 0.0183으로 gate 0.05 미달. FDA 단독으로 MAPLES cross-domain evidence gap을 해결하지 못했다.
- **Phase 4-G MAPLES-heavy fine-tune diagnostic**: `seg_evidence_v6_maples_finetune_tjdr`는 v5 best checkpoint에서 시작해 MAPLES sampling weight를 4.0으로 올린 low-LR fine-tune이다. Best epoch 2, val mDice 0.2145, early stop epoch 6. Threshold 0.5 aligned eval: IDRiD mDice 0.2144 / union IoU 0.2574, MAPLES mDice 0.0134 / union IoU 0.0175, TJDR mDice 0.2816 / union IoU 0.2373. Threshold sweep MAPLES best는 mDice 0.0165 / union IoU 0.0201로 v5보다 소폭 개선됐지만 gate 0.05에 크게 미달한다. IDRiD/TJDR best union IoU는 v5보다 소폭 낮아졌다. Fine-tune/upweighting만으로 MAPLES 일반화는 해결되지 않았다.
- **MAPLES-only specialist segmenter diagnostic**: `seg_evidence_v7_maples_only`는 MAPLES train 122장만 사용해 target-only 학습을 확인했다. Best val mDice 0.0090, MAPLES test best mDice 0.0039 / union IoU 0.0056로 실패했다. 현재 512px preprocessed data 안에서 target-only fine-tuning은 MAPLES lesion evidence의 해결책이 아니다.
- **Phase 4-G no-FGADR DDR segmentation result**: FGADR는 access burden 때문에 active path에서 제외했다. DDR segmentation subset을 `data/raw/ddr/lesion_segmentation`에 배치하고 `DDRSegMaskProvider`, `build_manifest --include-ddr-seg`, `seg_evidence_v8_ddrseg_tjdr.yaml`로 v8을 학습했다. Best val mDice 0.3019. IDRiD/TJDR/DDR_SEG evidence는 개선됐지만 MAPLES best mDice 0.0103 / union IoU 0.0102로 gate 0.05에 미달한다. v8은 not promoted; 이후 MAPLES ROI fix rerun인 v8b가 active fusion evidence module이 됐다.
- **Sprint boundary**: Sprint 4 closed at v31 active deployment plus Phase 4-E/F diagnostics. Phase 4-G is Sprint 5 work and includes TJDR/DDR_SEG integration, MAPLES ROI fix, v8b evidence baseline, v8b evidence classifier, v31+v8b late-fusion diagnostic, and `v31_v8b_fusion_v2` AI-side deployment packaging. Additional MA/HE/EX/SE 4-channel lesion-mask dataset expansion remains a Sprint 5 follow-up. FGADR remains excluded from the default path unless local source data and access terms are explicitly provided.
- **Phase 4-G deployment packaging**: `v31_v8b_fusion_v2` packaged the source `v31_v8b_late_fusion_sweep_v1` numeric meta-classifier into a composite checkpoint, then `v31_v8b_fusion_features_hflip_v2` superseded it with append-only lesion features and hflip Option A. The current active alias is `v31_v8b_fusion_quickqual_v2` (v31 collinearity refit of the QuickQual line), which packages the QuickQual-line classifier/segmenter/meta-classifier. `configs/base.yaml` uses `model.architecture: v31_v8b_fusion`, `infer.threshold: 0.08563088401268978`, `infer.tta_mode: none`, `infer.preprocess_mode: none`, `infer.use_meta_classifier: true`, and `evidence_type: lesion_segmentation`. Backend/frontend code was not changed.
