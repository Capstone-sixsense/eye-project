# -*- mode: python ; coding: utf-8 -*-
#
# PyInstaller 스펙 파일 — eye-project 백엔드 (Windows 배포용)
#
# 실행 방법 (Windows PowerShell, D:\eye-project 기준):
#   pyinstaller windows\backend.spec
#
# 결과물: windows\dist\eye_backend\ 폴더
#   eye_backend.exe      ← FastAPI 서버 실행파일
#   configs\             ← AI YAML 설정 파일들
#   models\              ← QuickQual .pkl 모델
#   (torch, cv2 등 DLL들...)

from pathlib import Path

ROOT = Path(SPEC).resolve().parent.parent  # windows/ 의 부모 = 프로젝트 루트

a = Analysis(
    [str(ROOT / 'backend' / 'main.py')],

    # PyInstaller가 import를 탐색할 경로.
    # PYTHONPATH=/ai (Docker) 와 동일한 역할 → drscreen 패키지를 찾을 수 있게 됨.
    pathex=[
        str(ROOT / 'backend'),
        str(ROOT / 'ai'),
    ],

    binaries=[],

    # 소스 코드가 아닌 데이터 파일 (YAML, pkl 모델 등).
    # 형식: (원본 경로, 번들 내 상대 경로)
    datas=[
        # AI YAML 설정 파일 전체 → exe 옆 configs/ 폴더에 복사
        (str(ROOT / 'ai' / 'configs'), 'configs'),
        # QuickQual SVM 모델 → exe 옆 models/ 폴더에 복사
        (str(ROOT / 'backend' / 'models' / 'quickqual_dn121_512.pkl'), 'models'),
    ],

    # PyInstaller 정적 분석으로 찾지 못하는 동적 import 목록.
    #
    # drscreen/models/build.py 의 build_model() 함수는 model_name 에 따라
    # if-블록 안에서 조건부 import를 수행한다:
    #   if model_name == "lesion_seg_evidence":
    #       from drscreen.models.seg_evidence import LesionSegEvidence
    # 이런 패턴은 정적 분석으로 감지되지 않으므로 수동 명시가 필요하다.
    hiddenimports=[
        # --- drscreen 추론 경로 ---
        'drscreen',
        'drscreen.settings',
        'drscreen.infer',
        'drscreen.infer.service',
        'drscreen.infer.pipeline',
        'drscreen.infer.payload',
        'drscreen.infer.late_fusion_features',
        # --- drscreen 모델 (build.py if-블록 동적 import) ---
        'drscreen.models',
        'drscreen.models.build',
        'drscreen.models.profiles',
        'drscreen.models.aux_seg',
        'drscreen.models.mil_attention',
        'drscreen.models.sparse_bagnet',
        'drscreen.models.concept_bottleneck',
        'drscreen.models.seg_evidence',
        'drscreen.models.fusion',
        # --- drscreen 데이터 전처리 ---
        'drscreen.data',
        'drscreen.data.transforms',
        'drscreen.data.datasets',
        'drscreen.data.manifest_builder',
        'drscreen.data.manifest_variants',
        'drscreen.data.mask_providers',
        # --- drscreen XAI ---
        'drscreen.xai',
        'drscreen.xai.gradcam',
        'drscreen.xai.iou',
        'drscreen.xai.evaluation',
        'drscreen.xai.faithfulness',
        'drscreen.xai.perturbation',
        'drscreen.xai.seg_metrics',
        # --- drscreen 유틸리티 ---
        'drscreen.utils',
        'drscreen.utils.checkpoint',
        'drscreen.utils.logging',
        'drscreen.utils.seed',
        # --- drscreen 학습 (추론 시에도 일부 참조됨) ---
        'drscreen.train',
        'drscreen.train.model_setup',
        'drscreen.train.metrics',
        'drscreen.train.evaluate',
        'drscreen.train.loss',
        'drscreen.train.checkpointing',
        # --- 외부 패키지 ---
        'timm',
        'timm.layers',
        'timm.layers.cbam',
        'timm.models',
        'cv2',
        'sklearn',
        'sklearn.svm',
        'sklearn.utils._cython_blas',
        'sklearn.neighbors._partition_nodes',
        'joblib',
        'albumentations',
        'captum',
        'captum.attr',
        'pytorch_grad_cam',
    ],

    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],

    # 추론에 불필요한 대용량 패키지 제거 → 번들 크기 절감.
    excludes=[
        'matplotlib',
        'IPython',
        'notebook',
        'jupyter',
        'tkinter',
        # pandas, scipy: drscreen.data.datasets 에서 사용 → 제거 금지
    ],

    noarchive=False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,   # one-dir 모드: DLL들을 폴더에 따로 둠
    name='eye_backend',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,                # UPX 압축 (없으면 False로)
    console=True,            # 초기 배포 시 True → 오류 로그 콘솔에 표시
    icon=None,               # 아이콘 파일 경로 (예: str(ROOT / 'windows' / 'icon.ico'))
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='eye_backend',      # 결과: windows/dist/eye_backend/ 폴더
)
