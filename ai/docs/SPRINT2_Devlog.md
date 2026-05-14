# SPRINT 2 개발 요약

## 1. 개요
- **프로젝트 명**: eye-project (AI 파트: drscreen)
- **목표**: Sprint 1에서 확인된 외부 도메인 일반화 한계와 threshold bias를 개선하고, 평가 데이터셋을 Messidor에서 DDR로 전환하여 더 큰 외부 검증 체계를 수립
- **기간**: SPRINT 2 주요 실험 이력 (2026.04.09 ~ 2026.04.17, v10 SWAD BN 재평가 2026.04.21 반영)
- **대상 범위**: RETFound 폐기, v4/v4.1 계보 정정, SSL 오염 검증, v7_messidor_train ~ v14_ibn 도메인 일반화 실험

> 주의: Sprint 2 종료 기준 best는 `v9_fda`였으나, 이후 Sprint 3에서 `v9_fda`가 새 전처리로 재학습되어 현재 `external_test_v9_fda_best_metrics.json`의 수치는 Sprint 2 원본 기록과 다르다. 이 문서는 Sprint 2 이력 요약이므로 `DEVLOG.md`의 Sprint 2 기록을 기준으로 정리하고, 현재 아티팩트로 직접 재확인 가능한 값은 근거 파일에 별도 표기한다.

## 2. 주요 개발 및 성과 (Milestones)

### 2.1 외부 평가 체계 재정의 및 v7 기준선 수립
- 기존 외부 평가셋이던 **Messidor 1,200장**을 학습 데이터에 편입하고, 외부 테스트를 **DDR 12,522장**으로 교체.
- `v7_messidor_train`에서 APTOS + IDRiD + Messidor 기반 학습 후 DDR external_test를 평가.
- DDR 기준선 성능은 AUROC **0.8725**, optimal threshold **0.09**로 확인됨.
- AUROC는 기존 Messidor 기준 `v6_alpha_only`와 비슷한 수준까지 도달했지만, threshold가 0.09로 낮아 DDR 도메인에서 이상 확률을 낮게 산출하는 편향이 남음.

### 2.2 RETFound 백본 교체 실험 및 폐기
- EfficientNet-B5 한계를 보완하기 위해 안저 foundation model인 **RETFound ViT-Large** 백본을 실험.
- `retfound_v1`(BCE)과 `retfound_v2`(Focal) 모두 Messidor AUROC가 각각 **0.6722**, **0.6611**로 기존 `v6_alpha_only`의 **0.8697**보다 크게 낮음.
- RETFound SimCLR + LLRD + Focal 재시도도 Messidor AUROC **0.728** 수준에 그쳐 폐기.
- 관련 코드는 `archive/retfound/`로 이동되었고, 공유 모델 빌드 경로에서는 제거됨.

### 2.3 v4/v4.1 계보 정정 및 SSL 오염 검증
- `v4/best.pt`와 `v4.1/best.pt`가 동일 학습 실행의 결과이며, `v4.1`은 SSL backbone(`artifacts/ssl/backbone_best.pt`)에서 시작한 supervised fine-tune으로 재확인됨.
- Sprint 1의 `v6_alpha_only` 성능에는 Messidor가 비레이블 SSL 단계에 포함된 영향이 있었음을 검증.
- Messidor를 SSL에서 제외한 `v4b_alpha_only`는 Messidor AUROC **0.7262**로, `v6_alpha_only`의 **0.8697**보다 **-0.1435** 낮음.
- 결론적으로 `v6_alpha_only`의 Messidor 성능은 순수 외부 일반화라기보다 Messidor 도메인 노출 효과를 포함한 값으로 판단됨.

### 2.4 Domain Adaptation 배제 및 Domain Generalization 전략 확립
- DANN 등 타깃 도메인 DDR을 학습에 사용하는 Domain Adaptation 계열은 외부 테스트 격리 원칙을 훼손하므로 기각.
- 이후 실험은 소스 도메인(APTOS + IDRiD + Messidor)만 사용하는 **Domain Generalization(DG)** 전략으로 제한.
- DG 우선순위는 FDA, SWAD, FDA+SWAD, IBN 계열로 정리됨.
- 학습 기반 복원 모델(SwinIR, Real-HAT, SUPIR)은 병변 과평활화 또는 hallucination 위험으로 즉시 도입하지 않기로 결정.

### 2.5 DG 실험 시리즈 (v8 ~ v14)
- **v8_mixstyle**: optimal threshold는 0.09 -> 0.31로 개선됐으나 AUROC가 **0.8371**로 하락해 폐기.
- **v9_fda**: 저주파 Fourier 진폭 교환으로 도메인 색조 편향을 완화. Sprint 2 기록 기준 DDR AUROC **0.8812**, optimal threshold **0.19**로 개선되어 Sprint 2 종료 기준 best로 지정.
- **v10_swad**: BN 재보정 버그 수정 후 DDR AUROC **0.8863**으로 높았으나 optimal threshold **0.05**로 악화되어 배포 부적합.
- **v11_fda_swad / v13_fda_swad**: FDA와 SWAD를 결합했으나 AUROC가 각각 **0.8539**, **0.8436**으로 하락. FDA의 epoch별 스타일 노이즈와 SWAD의 flat minima 가정이 충돌한 것으로 분석.
- **v12_fda_imagenet**: v7 backbone 없이 ImageNet 초기화 + FDA만으로 학습했으나 AUROC **0.8498**, threshold **0.05**로 실패. 누적 fine-tuning 계보가 DG 성능의 전제임을 확인.
- **v14_ibn**: shallow block BN을 IBN-a로 교체했으나 AUROC **0.8445**, threshold **0.08**로 하락. InstanceNorm이 진단에 필요한 국소 texture까지 약화한 것으로 판단.

### 2.6 파이프라인 및 코드 변경
- `manifest_builder.py`, `build_manifest.py`: Messidor 학습 편입 및 DDR external_test 구성을 위한 manifest 생성 기능 확장.
- `drscreen/data/transforms.py`: FDA용 `fda_mix(source, reference, alpha)` 구현.
- `drscreen/data/datasets.py`: `FDAManifestDataset` 추가, 도메인별 참조 이미지 샘플링 및 FDA 적용 지원.
- `drscreen/train/engine.py`: `SWADBuffer` 추가.
- `drscreen/train/runner.py`: FDA dataset 분기, SWAD 통합, BN 재보정 로직 수정, IBN/FDA 설정 전달 보강.
- `drscreen/models/build.py`: IBN-a 주입 로직 추가.
- `drscreen/infer/service.py`: Grad-CAM 실패 시 `xai_error_code="XAI_001"` 반환 및 IBN 설정 전달 보강.

## 3. 주요 성능 지표

### 3.1 Sprint 1 best와 SSL 오염 검증

| 모델 | 평가 데이터셋 | AUROC | Sensitivity@0.5 | Specificity@0.5 | 비고 |
|---|---|---:|---:|---:|---|
| `v6_alpha_only` | Messidor | **0.8697** | 0.7049 | 0.9103 | Sprint 1 종료 기준 best, Messidor SSL 노출 영향 포함 |
| `v4b_alpha_only` | Messidor | 0.7262 | 0.5015 | 0.8462 | Messidor-free SSL + Focal 재학습 |

### 3.2 DDR external_test 기준 DG 실험

| 버전 | 핵심 기법 | DDR AUROC | Optimal threshold | Sensitivity@optimal | Specificity@optimal | 판정 |
|---|---|---:|---:|---:|---:|---|
| `v7_messidor_train` | Messidor 학습 편입 | 0.8725 | 0.09 | 0.7626 | 0.8417 | DDR 기준선 |
| `v8_mixstyle` | MixStyle | 0.8371 | 0.31 | 0.6638 | 0.8797 | 폐기 |
| `v9_fda` | FDA | **0.8812** | **0.19** | 0.7353 | 0.8894 | Sprint 2 종료 기준 best |
| `v10_swad` | SWAD | 0.8863 | 0.05 | 0.7212 | 0.9033 | AUROC 우위, threshold 악화로 배포 보류 |
| `v11_fda_swad` | FDA + SWAD | 0.8539 | 0.31 | 0.7088 | 0.8936 | 폐기 |
| `v12_fda_imagenet` | FDA + ImageNet 초기화 | 0.8498 | 0.05 | 0.6726 | 0.8886 | 폐기 |
| `v13_fda_swad` | FDA + SWAD BN 수정 | 0.8436 | 0.05 | 0.6084 | 0.9443 | 폐기 |
| `v14_ibn` | IBN-Net | 0.8445 | 0.08 | 0.6995 | 0.8468 | 폐기 |

> `v9_fda` 행은 Sprint 2 원본 DEVLOG 기록 기준이다. 현재 저장된 `external_test_v9_fda_best_metrics.json`은 Sprint 3 재학습 후 수치(AUROC 0.8825, optimal threshold 0.06)이므로 Sprint 2 원본 결과의 직접 검증 근거로 사용하지 않는다.

## 4. 결론 및 향후 과제
- **성과**: Sprint 2는 Messidor 의존 평가를 DDR 외부 테스트로 전환하고, 순수 DG 전략을 기준으로 FDA, SWAD, IBN 등 주요 후보를 검증했다. 최종적으로 `v9_fda`가 AUROC와 threshold 균형 기준에서 Sprint 2 종료 best로 지정됐다.
- **핵심 학습**:
  - Messidor SSL 노출은 실제 성능 향상에 영향을 주었으며, 기존 Messidor AUROC는 순수 일반화 성능으로 보기 어렵다.
  - DDR을 외부 테스트로 격리하기 위해 DANN 등 DA 계열은 배제해야 한다.
  - MixStyle, RETFound, IBN, FDA+SWAD 조합은 현재 데이터와 모델 구조에서 실효성이 낮거나 역효과가 확인됐다.
  - FDA는 잘 수렴된 EfficientNet 계보 위에서 도메인 색조 편향을 줄이는 데 가장 실용적이었다.
- **남은 과제**:
  - `v9_fda`도 optimal threshold가 0.19에 머물러 threshold bias가 완전히 해결되지는 않음.
  - DDR sensitivity@optimal은 0.7353으로 임상 스크리닝 목표(일반적으로 0.80 이상, 더 엄격하게는 0.90 이상)에 미달.
  - CORAL, temperature scaling, calibration split 구성 등 후속 Sprint에서 확률 보정과 도메인 불변 표현 학습을 계속 검토해야 함.
  - DDR을 학습에 포함하려면 새로운 독립 외부 테스트셋 확보가 선행되어야 함.

## 5. 근거 파일
- `ai/docs/AI_HANDOFF.md`: 현재 기준 문서. `DEVLOG.md`는 역사 로그로 취급한다.
- `ai/docs/DEVLOG.md`: Sprint 2 상세 변경 이력. v7~v14, RETFound, v4/v4.1 정정, SSL 오염 검증, DG 전략 결정 기록.
- `ai/artifacts/runs/02_domain_generalization/v7_messidor_train/evaluations/external_test_v7_messidor_train_best_metrics.json`
- `ai/artifacts/runs/02_domain_generalization/v8_mixstyle/evaluations/external_test_v8_mixstyle_best_metrics.json`
- `ai/artifacts/runs/02_domain_generalization/v10_swad/evaluations/external_test_v10_swad_best_metrics.json`
- `ai/artifacts/runs/02_domain_generalization/v11_fda_swad/evaluations/external_test_v11_fda_swad_best_metrics.json`
- `ai/artifacts/runs/02_domain_generalization/v12_fda_imagenet/evaluations/external_test_v12_fda_imagenet_best_metrics.json`
- `ai/artifacts/runs/02_domain_generalization/v13_fda_swad/evaluations/external_test_v13_fda_swad_best_metrics.json`
- `ai/artifacts/runs/02_domain_generalization/v14_ibn/evaluations/external_test_v14_ibn_best_metrics.json`
- `ai/artifacts/runs/01_ssl_lineage/v4b_alpha_only/evaluations/external_test_v4b_alpha_only_best_metrics.json`
- `ai/artifacts/runs/01_ssl_lineage/v6_alpha_only/evaluations/external_test_v6_alpha_only_best_metrics.json`
- `ai/archive/retfound/artifacts/evaluations/external_test_retfound_v1_best_metrics.json`
- `ai/archive/retfound/artifacts/evaluations/external_test_retfound_v2_best_metrics.json`
- `ai/archive/retfound/artifacts/evaluations/external_test_retfound_simclr_ft_best_metrics.json`
- `ai/archive/retfound/artifacts/evaluations/external_test_retfound_simclr_ft_llrd_focal_best_metrics.json`

---
**[SPRINT 2 종료]**
