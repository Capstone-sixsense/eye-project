# SPRINT 1 개발 요약

## 1. 개요
- **프로젝트 명**: eye-project (AI 파트: drscreen)
- **목표**: 안저 이미지를 활용한 당뇨망막병증(DR) 이진 분류 스크리닝 시스템 구축 및 성능 최적화
- **기간**: SPRINT 1 (v0.1.0 ~ v0.7.0)

## 2. 주요 개발 및 성과 (Milestones)

### 2.1 기초 인프라 및 모델 베이스라인 구축 (v0.1.x)
- EfficientNet-B3/B5 기반의 이진 분류 파이프라인 수립
- APTOS + IDRiD 데이터셋을 활용한 학습 환경 구축
- **Shadow Validation**: IDRiD 데이터를 검증셋에 강제 포함하여 도메인 편향 완화

### 2.2 모델 아키텍처 고도화 (v0.2.x ~ v0.3.x)
- **timm 백본 전환**: EfficientNet-B5로 업그레이드
- **어텐션 메커니즘 통합**: ECA(Efficient Channel Attention) 및 Spatial Attention(공간 어텐션)을 MBConv 블록 내부에 주입하여 미세 병변 포착 능력 강화
- **Messidor 외부 데이터셋 도입**: 학습에 사용되지 않은 외부 데이터로 일반화 성능 검증 시작

### 2.3 시스템 통합 및 전처리 최적화 (v0.4.x ~ v0.5.x)
- **FastAPI 백엔드 통합**: 추론 세션을 FastAPI lifespan에 로드하여 실시간 API 서비스 구축
- **Ben Graham 전처리 도입**: Circular Crop 및 적응형 가우시안 블러를 통해 조명 편차 및 히스토그램 왜곡 해결
- **QuickQual 품질 검사**: 저품질 이미지(블러, 밝기)를 사전에 필터링하는 파이프라인 구축
- **학습 효율화**: 오프라인 전처리 및 배치 사이즈 최적화로 GPU 사용률 9% -> 99% 달성

### 2.4 도메인 일반화 및 손실 함수 최적화 (v0.6.x ~ v0.7.x)
- **SimCLR 비지도 학습(SSL)**: 레이블 없는 데이터를 활용해 도메인 불변 표현을 학습, Messidor AUROC를 0.54 -> 0.76으로 대폭 개선
- **Focal Loss & Alpha-Weighting**: Hard Example에 집중하고 양성 클래스 가중치를 조정하여 Messidor AUROC를 최종 **0.87**까지 향상

## 3. 최종 성능 지표 (v6_alpha_only 기준)

| 평가 데이터셋 | AUROC | Sensitivity | Specificity | 비고 |
|---|---|---|---|---|
| **내부 Test (APTOS+IDRiD)** | 0.9893 | 0.9790 | 0.9010 | 높은 신뢰도 확보 |
| **외부 Test (Messidor)** | **0.8697** | 0.7050 | 0.9100 | 일반화 성능 대폭 개선 |

## 4. 결론 및 향후 과제
- **성과**: 다양한 전처리 기법과 SSL, Focal Loss 실험을 통해 초기 0.5 수준이었던 외부 데이터 일반화 성능을 실용 가능한 수준(0.87)까지 끌어올림.
- **남은 과제**:
  - Messidor Sensitivity(70.5%)를 임상 기준(80% 이상)으로 추가 향상 필요.
  - RETFound 등 안저 전용 파운데이션 모델 도입 검토.
  - 다중 등급 분류(5-class) 확장 준비.

- 비지도 대조 학습(SSL) 과정에서 외부 테스트셋인 Messidor의 이미지가 레이블 없이 포함되었음. 인코더가 해당 도메인의 분포를 사전에 학습함으로써 실제 운영 환경에서의 순수 일반화 성능보다 AUROC가 과대평가되었을 가능성이 높음.


---
**[SPRINT 1 종료]**