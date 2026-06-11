# -*- mode: python ; coding: utf-8 -*-
#
# PyInstaller 스펙 파일 — EyeProject 런처 (Windows 배포용)
#
# 실행 방법 (Windows PowerShell, D:\eye-project 기준):
#   pyinstaller windows\launcher.spec
#
# 결과물: windows\dist\EyeProject.exe (단일 파일)

from pathlib import Path

ROOT = Path(SPEC).resolve().parent.parent

a = Analysis(
    [str(ROOT / 'windows' / 'launcher.py')],
    pathex=[str(ROOT / 'windows')],
    binaries=[],
    datas=[],
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
)

pyz = PYZ(a.pure)

# one-file 모드: 런처는 작은 스크립트이므로 단일 exe가 적합
exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='EyeProject',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    console=False,   # 런처 콘솔 창 숨김 (오류는 MessageBox로 표시)
    icon=None,       # 아이콘 파일 경로 (예: str(ROOT / 'windows' / 'icon.ico'))
)
