# Experiment Registry

This file is the canonical classification index for existing `eye-project/ai` experiment artifacts.

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

## Groups

| Group | Purpose | Runs |
|---|---|---|
| `00_baselines_and_early` | Initial baselines and early supervised attempts | `effnet_1shot`, `v3`, `v4`, `v5` |
| `01_ssl_lineage` | SSL lineage, SSL contamination checks, focal variants | `v4.1`, `v4b`, `v4b_alpha_only`, `v6`, `v6_alpha_only`, `v6_gamma_only` |
| `02_domain_generalization` | Messidor inclusion, FDA, SWAD, IBN, CORAL | `v7_messidor_train`, `v8_mixstyle`, `v9_fda`, `v10_swad`, `v11_fda_swad`, `v12_fda_imagenet`, `v13_fda_swad`, `v14_ibn`, `v15_fda_a10`, `v16_focal_g1`, `v17_focal_g2`, `v18_focal_g3`, `v19_swad_focal_g2`, `v20_coral` |
| `03_resolution_layercam` | 512px training and Layer-CAM deployment experiments | `v7_512_messidor_train`, `v17_512_focal_g2`, `v21_512_focal_g2`, `v21_512_layercam` |
| `04_lesion_supervision` | Auxiliary lesion mask supervision and lesion-aware heads | `v24_multitask`, `v25_multitask_l1`, `v26_multitask_l3`, `v27_mil_attention` |
| `05_xai_attention_ablation` | Matched XAI attention ablation and block sweeps | `v24_multitask`, `v28_no_attention`, `v29_with_attention` |
| `06_xai_classifier_routing` | Lesion gate routing into classifier pooling path | `v30_gated_pooling` |
| `07_lesion_evidence` | SE/ECA 제거 + gated pooling 유지 대조군(v31), per-lesion routing 시리즈(v32~v35) | `v31_no_se_gated`, `v32_lesion_seg_evidence`, `v33_per_lesion_routing`, `v34_calibrated_routing`, `v35_warmstart_routing` |
| `06_deployment_candidates` | Runs relevant to deployment decisions | `v17_focal_g2`, `v17_512_focal_g2`, `v21_512_layercam`, `v24_multitask`, `v28_no_attention`, `v30_gated_pooling`, `v31_no_se_gated` |

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
| `v17_focal_g2` | `02_domain_generalization` | `06_deployment_candidates` | `v17_focal_g2.yaml` | yes | external | - | completed |
| `v18_focal_g3` | `02_domain_generalization` | - | `v18_focal_g3.yaml` | yes | external | - | discarded |
| `v19_swad_focal_g2` | `02_domain_generalization` | - | `v19_swad_focal_g2.yaml` | yes | external | - | discarded |
| `v20_coral` | `02_domain_generalization` | - | `v20_coral.yaml` | yes | external | - | completed |
| `v7_512_messidor_train` | `03_resolution_layercam` | - | `v7_512_messidor_train.yaml` | yes | external, test | - | completed |
| `v17_512_focal_g2` | `03_resolution_layercam` | `06_deployment_candidates` | - | yes | external, test | - | completed |
| `v21_512_focal_g2` | `03_resolution_layercam` | - | - | yes | external, test | - | completed |
| `v21_512_layercam` | `03_resolution_layercam` | `06_deployment_candidates` | `v21_512_layercam.yaml` | yes | external | train XAI, block sweep | completed |
| `v24_multitask` | `04_lesion_supervision` | `05_xai_attention_ablation`, `06_deployment_candidates` | `v24_multitask.yaml` | yes | external, test | method compare, block sweep, seg head | completed |
| `v25_multitask_l1` | `04_lesion_supervision` | - | `v25_multitask_l1.yaml` | yes | test | default XAI | discarded |
| `v26_multitask_l3` | `04_lesion_supervision` | - | `inactive/v26_multitask_l3.yaml.inactive` | no | none | none | inactive config only |
| `v27_mil_attention` | `04_lesion_supervision` | - | `v27_mil_attention.yaml` | yes | test | MIL attention XAI (test, train) | discarded for XAI |
| `v28_no_attention` | `05_xai_attention_ablation` | `06_deployment_candidates` | `v28_no_attention.yaml` | yes | external, test | block sweep | completed; previous deployment candidate |
| `v29_with_attention` | `05_xai_attention_ablation` | - | `v29_with_attention.yaml` | yes | external | pending | classification done; XAI pending |
| `v30_gated_pooling` | `06_xai_classifier_routing` | `06_deployment_candidates` | `v30_gated_pooling.yaml` | yes | external | block sweep | completed classifier-routing baseline |
| `v31_no_se_gated` | `07_lesion_evidence` | `06_deployment_candidates` | `v31_no_se_gated.yaml` | yes | external | train/test XAI block4 | active deployment alias; val AUROC 0.9993, DDR AUROC 0.9160 |
| `v32_lesion_seg_evidence` | `07_lesion_evidence` | - | `v32_lesion_seg_evidence.yaml` | yes | none | train XAI seg_head | completed; val AUROC 0.9992, not promoted |
| `v33_per_lesion_routing` | `07_lesion_evidence` | - | `v33_per_lesion_routing.yaml` | yes | external | test XAI block4 | completed; val AUROC 0.9980, DDR AUROC 0.9131; AUC-IoU best (0.0557) but classification < v31 — not promoted |
| `v34_calibrated_routing` | `07_lesion_evidence` | - | `v34_calibrated_routing.yaml` | yes | external | test XAI block4 | completed; val AUROC 0.9989, DDR AUROC 0.9129; PG best (0.5185) but classification < v31 — not promoted |
| `v35_warmstart_routing` | `07_lesion_evidence` | - | `v35_warmstart_routing.yaml` | yes | external | test XAI block4 | completed; val AUROC 0.9992, DDR AUROC 0.9081; AUPRC best (0.1537) but DDR regressed — warmstart 역효과 확인, 4ch trade-off 구조적 확정 |

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
| `v31_no_se_gated.yaml` | True no-attention gated-pooling control | completed; removes ECA/Spatial/SE via `attention_mode: none`, Dice+BCE seg loss |
| `v32_lesion_seg_evidence.yaml` | Per-lesion segmentation evidence candidate | completed; 4-channel IDRiD MA/HE/EX/SE provider, Dice+BCE seg loss |

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
| `07_lesion_evidence` | `v31_no_se_gated` | 12522 | **0.916036** | **0.35** | **0.7983** | **0.8677** |
| `07_lesion_evidence` | `v33_per_lesion_routing` | 12522 | 0.913102 | 0.32 | 0.765 | 0.912 |
| `07_lesion_evidence` | `v34_calibrated_routing` | 12522 | 0.912859 | 0.51 | 0.772 | 0.908 |
| `07_lesion_evidence` | `v35_warmstart_routing` | 12522 | 0.908138 | 0.18 | 0.7932 | 0.8739 |

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

## XAI Summary

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
| `v33_per_lesion_routing` | test | block4 Layer-CAM | 27 | 0.4074 | 0.1478 | 0.0557 | 0.0799 |
| `v34_calibrated_routing` | test | block4 Layer-CAM | 27 | **0.5185** | 0.1492 | 0.0543 | 0.0769 |
| `v35_warmstart_routing` | test | block4 Layer-CAM | 27 | 0.4074 | **0.1537** | 0.0525 | 0.0796 |
| `v32_lesion_seg_evidence` | train | seg_head | 54 | 0.2222 | 0.0538 | 0.0208 | 0.0364 |

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

- **`v31_no_se_gated` is the active deployment alias** in `configs/base.yaml`, and `artifacts/checkpoints/best.pt` currently contains the `v31_no_se_gated` checkpoint. DDR AUROC 0.9160 (> v30 0.9137 > v28 0.8924). Optimal threshold 0.35, Sensitivity 0.798, Specificity 0.868. test-split XAI block4 Layer-CAM: PG **0.3704**, AUPRC **0.1409**, AUC-IoU **0.0496**, IoU top-20 **0.0785**. v31은 classification-best active deployment다.
- `v28_no_attention` is a previous deployment candidate and remains in the registry for attention-ablation/block-sweep comparison.
- `v30_gated_pooling` DDR AUROC 0.9137, test XAI AUPRC 0.1311 — v31이 DDR AUROC, AUPRC, AUC-IoU에서 앞서며 PG는 동일(0.3704), IoU top-20은 v30 0.0788 vs v31 0.0785로 사실상 동률이다. classifier-routing 설계 baseline으로만 유지.
- block4 lesion gate를 classifier pooling 경로에 곱하는 방식이 분류와 XAI 지표 모두 개선 — "분류 경로에 병변 위치 신호를 묶으면 XAI 정렬이 개선되는가" 가설에 긍정적.
- `v32_lesion_seg_evidence` (train split, seg_head 직접 출력): PG 0.2222, AUPRC 0.0538, AUC-IoU 0.0208. v32 artifact는 seg_head train 평가만 존재하고 제품 XAI 후보로 보기 어렵다. 현재 코드의 4채널 gated classifier는 per-lesion sigmoid + softmax weighted sum을 사용하며, 단일 evidence map 생성용 `predict_seg_union()`만 `amax(dim=1)` union을 사용한다.
- `v27_mil_attention` XAI 결과 (train): AUC-IoU 0.0119 — random baseline(0.0260) 이하. MIL attention은 spatial localization에 구조적으로 부적합. 방향 폐기.
- `v29_with_attention` XAI artifacts 미생성. 분류 AUROC 0.8629로 v28(0.8924) 대비 열위. ECA+CBAM spatial attention이 분류·XAI 모두 악화시킴. 방향 폐기.
- **Phase-0 gate 전 모델 FAIL**: test split 기준 center_gaussian+2σ threshold=0.1089. 모든 모델의 AUC-IoU(최고 v33 test 0.0557)가 임계값 미달. 2σ gate 기준 재조정 필요 (test 기준 후보: +1σ=0.0728 또는 절대값 0.05).
- `use_attention=false`는 ECA channel module을 유지하는 legacy 기준. `attention_mode: none`을 쓰는 v31만 SE/ECA/Spatial 계열을 IdentitySE로 대체하는 true no-attention 대조군.
- IDRiD contamination 주의: XAI eval은 `A. Segmentation` 이미지 사용, 분류 학습은 `B. Disease Grading` 이미지 사용 — 파일은 다르나 동일 환자 포함(patient-level overlap). file-level contamination은 없음.
- `v33_per_lesion_routing` (test split): PG 0.4074, AUPRC 0.1478, AUC-IoU **0.0557** — AUC-IoU 기준 최고. per-lesion 독립 sigmoid + learnable weighted sum(softmax) 구조가 XAI 품질을 개선했음을 확인. 단, DDR AUROC 0.9131 < v31 0.9160으로 분류 소폭 하락. v31 배포 유지.
- **MAPLES-DR 확보 완료**: `data/raw/MAPLES-DR/AdditionalData/` (train 138 / test 60장, 12종 biomarker). `MAPLESMaskProvider` 구현 완료 (`drscreen/data/mask_providers.py`) 범위는 MA/HE/EX/CWS pathology mask 로딩이다. 현재 `eval_xai_iou.py`에는 `--mask-provider maples`와 anatomy/lesion attribution ratio 지표가 없으므로, clean-cohort XAI eval 및 Phase 1 anatomy audit는 평가 CLI/metric wiring이 필요하다.
- `v34_calibrated_routing` (test split): PG **0.5185**, AUPRC 0.1492 — PG 기준 최고. lambda_aux_seg=0.3 변경으로 XAI 소폭 개선. 그러나 DDR AUROC 0.9129 < v31 0.9160, 분류 기준 미달 — v31 배포 유지.
- `v35_warmstart_routing` (external_test): DDR AUROC 0.9081, optimal thr 0.18, Sens 0.7932, Spec 0.8739. test XAI AUPRC **0.1537** — AUPRC 기준 최고이나 AUC-IoU 0.0525, PG 0.4074로 v33/v34 대비 지표별 우위가 갈린다. 분류 회귀로 배포 미승격.
- **4ch per-lesion routing 구조 trade-off 최종 확정 (v33~v35)**: lambda 조정(v34), v31 warmstart(v35) 모두 DDR 회귀 미해소. v35 warmstart는 오히려 DDR AUROC 0.9081로 최저 — 4ch routing이 OOD 일반화를 구조적으로 희생. XAI 개선(AUPRC ↑)과 DDR 일반화(AUROC ↓)는 현 아키텍처에서 trade-off 관계.
- **실험 방향 전환**: 4ch per-lesion routing 추가 실험 중단. 현재 배포(v31)는 분류 최우선 기준으로 유지. XAI 개선은 분류에 영향 없는 방법 탐색으로 전환하되, (1) anatomy-guided CAM masking과 (2) MAPLES-DR clean-cohort XAI eval은 먼저 평가 CLI/metric wiring이 필요하다.
- 다음 단계: MAPLES-DR 기반 v31/v35 XAI eval 구현 및 측정 (clean-cohort, IDRiD contamination 없음).
