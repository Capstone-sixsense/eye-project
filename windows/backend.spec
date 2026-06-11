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
    [str(ROOT / 'windows' / 'backend_entry.py')],

    # PyInstaller가 import를 탐색할 경로.
    # PYTHONPATH=/ai (Docker) 와 동일한 역할 → drscreen 패키지를 찾을 수 있게 됨.
    pathex=[
        str(ROOT / 'windows'),
        str(ROOT / 'backend'),
        str(ROOT / 'ai'),
    ],

    binaries=[],

    # 소스 코드가 아닌 데이터 파일 (YAML, pkl 모델 등).
    # 형식: (원본 경로, 번들 내 상대 경로)
    datas=[
        # AI YAML 설정 파일 전체
        (str(ROOT / 'ai' / 'configs'), 'configs'),
        # QuickQual SVM 모델
        (str(ROOT / 'backend' / 'models' / 'quickqual_dn121_512.pkl'), 'models'),
        # AI 분류 모델 checkpoint (base.yaml: infer.checkpoint_path = artifacts/checkpoints/best.pt)
        (str(ROOT / 'ai' / 'artifacts' / 'checkpoints' / 'best.pt'), 'artifacts/checkpoints'),
        # 배포 성능 지표 및 XAI aggregate metrics
        (str(ROOT / 'ai' / 'artifacts' / 'evaluations'), 'artifacts/evaluations'),
        # DenseNet121 seed cache. Runtime hook copies this into writable AppData
        # and creates snapshots/{hash}/model.safetensors there.
        (str(ROOT / 'backend' / '.cache' / 'huggingface' / 'hub' /
             'models--timm--densenet121.tv_in1k' / 'blobs' /
             'c894c6d9caa317a8ca1942986dee7a16a86c77734a4d691d2abe05389cfef358'),
         '.cache/huggingface/hub/models--timm--densenet121.tv_in1k/blobs'),
        (str(ROOT / 'backend' / '.cache' / 'huggingface' / 'hub' /
             'models--timm--densenet121.tv_in1k' / 'refs' / 'main'),
         '.cache/huggingface/hub/models--timm--densenet121.tv_in1k/refs'),
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
        'sklearn.pipeline',
        'sklearn.preprocessing',
        'sklearn.calibration',
        'joblib',
        'albumentations',
        'albumentations.pytorch',   # drscreen/data/transforms.py: from albumentations.pytorch import ToTensorV2
        # captum, pytorch_grad_cam: 코드에서 사용하지 않으므로 제거
        # sklearn._cython_blas, _partition_nodes: scikit-learn 1.2.2에 존재하지 않으므로 제거
        # uvicorn: windows/backend_entry.py가 FastAPI app을 실행할 때 import
        'uvicorn',
        'uvicorn.lifespan',
        'uvicorn.lifespan.on',
        'uvicorn.protocols',
        'uvicorn.protocols.http',
        'uvicorn.protocols.http.auto',
        'uvicorn.protocols.websockets',
        'uvicorn.protocols.websockets.auto',
        'uvicorn.loops',
        'uvicorn.loops.auto',
    ],

    hookspath=[],
    hooksconfig={},
    runtime_hooks=[str(ROOT / 'windows' / 'hooks' / 'rthook_torch_home.py')],

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
    upx=False,               # PyTorch DLL은 UPX 압축 시 손상될 수 있어 비활성화
    console=True,            # 초기 배포 시 True → 오류 로그 콘솔에 표시
    icon=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,               # 동일한 이유로 비활성화
    upx_exclude=[],
    name='eye_backend',      # 결과: windows/dist/eye_backend/ 폴더
)
