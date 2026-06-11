# Cleanup & Restructure Proposal (검토용 리포트)

작성: 2026-06-04. **이 문서는 제안/리포트이며, 실제 삭제·이동은 수행하지 않았습니다.**
사용자 승인 후에만 개별 항목을 진행합니다.

핵심 결론을 먼저 요약하면:

- **Dead code는 거의 없습니다.** 흔한 미사용 import/변수는 0건(ruff), 모듈 단위 미참조도
  연구 스캐폴딩 4건(약 250줄) 뿐이며 전부 배포 경로 밖입니다.
- **Dead data는 위험합니다.** 지울 수 있는 `artifacts/runs`(59 GB)는 git 미추적(로컬 전용,
  복구 불가)이고 active source/rollback을 포함합니다. 자동 삭제는 하지 않습니다.
- **디렉토리 구조는 제약이 강합니다.** `project_root = config_path.parents[1]` 가정과 고정
  배포 경로 계약 때문에 config 중첩·artifacts 이동은 불가하며, 안전한 개선은 패키징 추가와
  루트 스크립트 정리 정도입니다.

---

## 1. Dead Code 후보 리포트

### 1.1 방법

- `ruff --select F401,F811,F841` (미사용 import/재정의/지역변수): **0건**.
- `vulture --min-confidence 60`: 24건 후보 → 각 항목을 `grep`으로 실제 참조 검증.
- 검증 기준: 정의 외 참조가 0이고, 동적 import/`getattr` 디스패치/CLI 진입점/테스트/config가
  사용하지 않는 경우에만 "미참조"로 분류.

### 1.2 진짜 미참조 (제거 후보, 그러나 의도적 스캐폴딩일 수 있음)

| 항목 | 위치 | 규모 | 성격 |
|---|---|---|---|
| `anatomy.py` 전체 (`locate_od_fovea`, `AnatomyLandmarks`, `_retina_mask`, `_as_rgb`) | `drscreen/data/anatomy.py` | ~100줄 | OD/중심와 검출. OD-anchored 특징(Problem 3)용 스캐폴딩이나 아직 미배선 |
| `manifest_variants.py` 전체 (`build_/write_shadow_validation_manifest`, `summarize_manifest_variant`) | `drscreen/data/manifest_variants.py` | ~105줄 | shadow-validation manifest 생성기. 패키지/테스트/CLI 미사용 |
| `MAPLESMaskProvider` | `drscreen/data/mask_providers.py:613` | ~50줄 | docs상 "구현 완료"이나 실제 eval 경로는 `load_maples_masks`를 직접 사용 → 클래스는 미참조 |
| `_seg_logits_to_gate` | `drscreen/models/aux_seg.py:264` | ~4줄 | private staticmethod, 호출처 없음 |
| `SPRINT2_KEYS` | `visualize_metrics.py:37` | 1줄 | 미사용 상수 |
| `MANIFEST_PATH`, `OUTPUT_ROOT` | `preprocess_images.py:28-29` | 2줄 | config로 대체된 모듈 상수(추정) — 확인 필요 |

**중요 경고:** 위 `anatomy.py` / `MAPLESMaskProvider` / `manifest_variants.py`는 `docs/AI_HANDOFF.md`에
"구현 완료된 능력"으로 기록돼 있습니다. 즉 **버려진 코드가 아니라 '계획됐으나 아직 배선 안 된'
스캐폴딩**일 수 있습니다. 제거 전 작성자(=사용자) 확인이 필요합니다.

### 1.3 vulture가 잡았으나 dead가 아닌 것 (오탐, 제거 금지)

| 항목 | 이유 |
|---|---|
| `payload.py`의 `quality`, `quality_warning`, `quality_grade`, `quality_grade_confidence`, `should_block` | 백엔드 페이로드 **계약 필드**. `asdict`로 직렬화됨. 테스트가 고정 검증 |
| `profiles.py`의 `num_params`, `gflops`, `gradcam_target_layer`, `rationale` | ModelProfile 정보 필드. `to_dict()`로 노출 |
| `service.py`의 `predict_image_bytes` | 바이트 입력 추론 **공개 API**(백엔드가 호출) |
| `service.py`의 `saved` (dataclass 필드) | SingleImagePrediction 데이터 계약 |
| `od_anchored_feature_names` | `tests/regression/test_fusion_contract.py`가 사용 → dead 아님 |
| `gradcam.py`의 `gi` (line 57, 291) | PyTorch backward hook의 위치 인자(grad_input). 제거 불가 |
| `concept_bottleneck._latest_concept_map_logits` | 연구 모델 내부 상태, accessor 경유 사용 |

### 1.4 "연구/미배포" 코드는 dead가 아님

BagNet(`sparse_bagnet`)·CBM(`concept_bottleneck`)·MIL(`mil_attention`)은 배포되지 않았지만
각각 **13 / 13 / 29개 파일 + 4개 config**에서 참조됩니다(`build.py` 디스패치, `model_setup`,
`profiles`, eval 스크립트, 테스트, configs). 이는 **의도적 연구 자산**이며 dead code가 아닙니다.
제거하려면 관련 config·테스트·디스패치 분기까지 함께 지워야 하는 **기능 제거 결정**입니다.

### 1.5 권장

1. 안전한 즉시 정리: `_seg_logits_to_gate`, `SPRINT2_KEYS`, `MANIFEST_PATH`/`OUTPUT_ROOT`
   (배포 경로 밖, 명백한 leftover) — 테스트 게이트 후 제거 가능. **승인 시 진행.**
2. 보류/확인 필요: `anatomy.py`, `manifest_variants.py`, `MAPLESMaskProvider` — 계획된 기능
   스캐폴딩 여부를 확인한 뒤 결정.
3. 연구 모듈(BagNet/CBM/MIL): 제거는 별도 "기능 제거" 결정으로 분리.

---

## 2. Dead Data 삭제 후보 인벤토리

### 2.1 git 추적 현황 (복구 가능성)

| 영역 | 추적 여부 | 삭제 시 복구 |
|---|---|---|
| `drscreen/**` 코드 | 추적 | git으로 복구 가능 |
| `artifacts/checkpoints/best.pt` | **추적**(예외 화이트리스트) | 복구 가능 |
| `artifacts/checkpoints/*.bak`, `staging_*`, `*.pt` | gitignore | **복구 불가(로컬 전용)** |
| `artifacts/runs/**` (59 GB) | gitignore | **복구 불가(로컬 전용)** |
| `data/**` | gitignore | **복구 불가(로컬 전용)** |
| `artifacts/evaluations/*` (선별 JSON 제외) | gitignore | 복구 불가 |

→ **용량의 핵심인 `artifacts/runs`(59 GB)는 전부 로컬 전용입니다. 한 번 지우면 끝입니다.**

### 2.2 용량 분포

```
artifacts/runs            59 GB  (116 runs / 12 groups)
  09_evidence_segmentation 13 GB (16)
  99_misc                  12 GB (19)  ← active source 포함
  02_domain_generalization 7.3 GB (15)
  10_grounded_classifier   6.7 GB (23)
  07_lesion_evidence       6.5 GB (12)
  08_xai_decoder_alignment 3.9 GB (9)
  01_ssl_lineage           3.4 GB (6)
  03_resolution_layercam   3.2 GB (6)
  04_lesion_supervision    1.6 GB (3)
  00_baselines_and_early   1.4 GB (4)
  05_xai_attention_ablation 1.1 GB (2)
  06_xai_classifier_routing 534 MB (1)
artifacts/checkpoints      2.3 GB  (best.pt + .bak rollback + staging)
artifacts/quickqual        57 MB
artifacts/heatmaps         56 MB   (재생성 가능 추론 산출물)
artifacts/preprocess_debug 15 MB
```

### 2.3 절대 삭제 금지 (active deployment + 재구성 소스 + 롤백)

- `artifacts/checkpoints/best.pt` — **현재 배포 모델 별칭**
- `99_misc/v31_no_se_gated_quickqual_v1` — active **분류기 소스**
- `99_misc/seg_evidence_v8b_quickqual_v1` — active **분할기 소스**
- `99_misc/v31_v8b_late_fusion_quickqual_v1_v31rep` — active **fusion metrics 소스**
- `99_misc/v31_v8b_fusion_quickqual_v1` — 즉시 롤백(v1)
- `checkpoints/best_pre_collinearity_refit_20260603.pt.bak` (즉시 롤백)
  / `best_pre_quickqual_v1_20260529.pt.bak` / `best_pre_features_hflip_v2_20260527.pt.bak` (심층 롤백)
- `07_lesion_evidence/v31_no_se_gated` (circular-era 기준 분류기, 비교용)
- `09_evidence_segmentation/seg_evidence_v8b_ddrseg_tjdr_maplesfix` (circular-era standalone best)

### 2.4 재현성/회귀용 — 유지 권장 (레지스트리가 명시)

- `99_misc/seg_evidence_v8b_repro_seed43/44/45` 및 `*_compat` / `*_geometryfix` — 재현성 검증
- `99_misc`의 1024 / quickqual v2·v3 진단 run

### 2.5 Superseded/Discarded — **검토 후보** (레지스트리상 미승격, 단 재현성 보존 대상일 수 있음)

> 아래는 docs/EXPERIMENT_REGISTRY 기준 '폐기/미승격'으로 기록된 run입니다. 그래도 로컬 전용이라
> 삭제는 복구 불가이며, 일부는 회귀 분석용으로 의도적으로 남겨둔 것입니다. **삭제는 직접 결정하세요.**

- `02_domain_generalization`: `v8_mixstyle`, `v11_fda_swad`, `v12_fda_imagenet`, `v13_fda_swad`,
  `v14_ibn`, `v15_fda_a10`, `v18_focal_g3`, `v20_coral`, `ensemble_v9v10`
- `08_xai_decoder_alignment`: `v36_xai_multi`(즉시 폐기), `v37_xai_multi_maples`, `v37b_aux04/05`,
  `v37c_xai_maples_r1plus`, `v38_xai_coral`, `v39_unet_2stage`
- `09_evidence_segmentation`: `seg_evidence_v1`(실패), `v2_focal_tversky`, `v2_geomfix_retrain`,
  `v3_tjdr`, `v4_deeplab_tjdr`, `v5_maples_fda_tjdr`, `v6_maples_finetune_tjdr`, `v7_maples_only`,
  `v8_ddrseg_tjdr`(misaligned, v8b로 대체), `v9_gin`, `v10_adverin`
- `10_grounded_classifier`: `bagnet_v1_p33_r256`, `bagnet_v1_p65_r512`, `cbm_v1_stage1`, `cbm_v1`,
  `v31_dfr_v1`(게이트 실패), `v41_ampmix`, `v42_coral_baseline`, `v42_rsc_coral`,
  contentcrop/safezoom fusion 변형들
- `04_lesion_supervision`: `v25_multitask_l1`, `v27_mil_attention`
- `05_xai_attention_ablation`: `v29_with_attention`

### 2.6 안전하게 즉시 정리 가능 (무위험)

- 재생성 캐시: `__pycache__/`, `.ruff_cache/`, `.pytest_cache/`
- 추론 부산물(원할 때 재생성): `artifacts/heatmaps/`(56 MB), `artifacts/predictions/`(396 KB)
  — 단, 보존 정책 확인 후.

### 2.7 권장

1. 무위험 캐시 정리부터(승인 시 즉시).
2. §2.5 목록을 직접 검토해 삭제할 run을 지정 → 항목별 확인 후 제거(복구 불가 재경고).
3. §2.3/§2.4는 보존.

---

## 3. 디렉토리 구조 제안서

### 3.1 현 구조

```
ai/
  AGENTS.md                         # 패키징 파일 없음 (pyproject/setup/requirements 부재)
  train.ps1
  *.py  (루트 15개)                 # eval_*, visualize_*, diagnose_*, preprocess_images, ...
  configs/  (*.yaml 110개, flat)
  drscreen/  (패키지, 9개 서브패키지)
  archive/retfound/  (폐기 실험, git 추적)
  artifacts/  (체크포인트/runs/평가, 대부분 gitignore)
  data/  (gitignore)
  docs/  tests/  .omc/
```

### 3.2 식별된 문제점

1. **루트 네임스페이스 오염**: 15개 스크립트가 `ai/` 루트에 흩어짐.
2. **루트 스크립트의 이중 성격**: `eval_seg_evidence.py`는 단순 실행 스크립트가 아니라
   **사실상 공유 모듈**이다 — `drscreen/cli/diagnose_v8b_mask_quality.py`와 루트
   `diagnose_v8b_serve_skew.py`가 `from eval_seg_evidence import ...`로 import한다.
   즉 루트 스크립트를 그냥 옮기면 import가 깨진다.
3. **configs 110개 flat**: 실험 계열(domain_gen / lesion_evidence / fusion / seg_evidence / xai /
   ssl)이 한 폴더에 섞여 있다.
4. **패키징 부재**: `pip install -e .` 불가. `ai/` 디렉토리에서 `py -3.14 -m drscreen.cli.X`로만 실행.

### 3.3 강한 제약 (반드시 보존)

- **고정 배포 경로 계약**: `infer.checkpoint_path: artifacts/checkpoints/best.pt`는 의도적으로 고정.
  `artifacts/` 구조는 옮기면 안 된다.
- **`project_root = config_path.parents[1]` 가정**: `cli/train.py`, `cli/evaluate.py`,
  `cli/pipeline.py`, `infer/service.py`가 모두 config의 부모의 부모를 project_root로 본다.
  즉 **config는 정확히 `ai/configs/X.yaml`(루트 1단계 아래)에 있어야 한다.**
  → **configs를 `ai/configs/family/X.yaml`로 중첩하면 project_root가 `ai/configs/`로 잘못 잡혀
  전 학습/추론이 깨진다.** config 중첩은 이 로직을 먼저 고치지 않는 한 불가.
- 체크포인트 내부에 저장된 config의 상대 경로, 110개 config의 manifest/이미지 경로.

### 3.4 제안 (위험도별)

**Tier A — 저위험·고가치 (권장)**
- `pyproject.toml` 추가: 의존성 명시 + `console_scripts` 진입점으로 `py -3.14 -m ...`를 대체,
  `pip install -e .`로 어디서나 실행 가능하게. (코드 이동 없음, 경로 계약 불변.)
- 공유돼버린 `eval_seg_evidence.py`의 재사용 헬퍼를 패키지로 승격(`drscreen/eval/seg_evidence.py`)
  하고, 루트는 얇은 CLI 래퍼만 남김. 그러면 import 결합이 정리된다.

**Tier B — 중위험 (참조 갱신 필요)**
- 순수 실행 스크립트(`visualize_*`, `prepare_messidor`, `sweep_xai_blocks`, 루트 `diagnose_*`)를
  `ai/scripts/`로 이동. **단** §3.2-2의 import 결합(2곳)과 `preprocess_images`를 참조하는 config 1건을
  먼저 갱신하고 `pytest tests/regression` 통과 확인.

**Tier C — 비권장 / 보류**
- configs 중첩: §3.3의 `parents[1]` 제약 때문에 **권장하지 않음**. 정말 필요하면 project_root
  해석을 (config 기준이 아니라) 명시적 루트 탐색으로 먼저 리팩터링해야 하며, 회귀 위험이 큼.
- `artifacts/` 재배치: 배포 계약 위반. **금지.**
- `archive/retfound/`: git 추적이라 복구 가능. 별도 보관소로 옮기거나 제거 가능하나 가치 낮음.

### 3.5 제안 목표 레이아웃 (Tier A+B만 반영한 예시)

```
ai/
  pyproject.toml            # 신규: 패키징 + console_scripts
  configs/  (*.yaml flat 유지 — parents[1] 제약)
  drscreen/
    eval/                   # 신규: eval_seg_evidence 등 공유 평가 로직 승격
    ...(기존 서브패키지)
  scripts/                  # 신규: 순수 실행 스크립트(visualize_*, diagnose_* 등) 이동
  artifacts/  data/  docs/  tests/   # 불변
  archive/                  # 유지 또는 별도 보관
```

### 3.6 권장 진행 순서 (승인 시)

1. (Tier A) `pyproject.toml` 추가 → 회귀 테스트 통과 확인. 코드 이동 0.
2. (Tier A) `eval_seg_evidence` 공유 로직을 `drscreen/eval/`로 승격 + import 2곳 갱신 → 테스트.
3. (Tier B) 나머지 순수 스크립트 `scripts/`로 이동 + 참조 갱신 → 테스트.
4. config 중첩(Tier C)은 별도 결정.

---

## 4. 전제 vs 실태 요약

| 요청 전제 | 실태 |
|---|---|
| "dead code가 많다" | 흔한 dead code 0건. 미참조는 연구 스캐폴딩 ~250줄(배포 밖) |
| "dead data가 많다" | 59 GB 있으나 로컬 전용·복구 불가·active/롤백 포함 → 자동 삭제 위험 |
| "디렉토리 구조 리팩 필요" | 강한 제약(parents[1], 배포 경로) → 안전 개선은 패키징+스크립트 정리로 한정 |

다음 단계는 위 각 Tier/항목 중 **승인하신 것만** 진행합니다.
