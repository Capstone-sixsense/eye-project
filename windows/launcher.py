"""
EyeProject Windows 런처

백엔드(FastAPI)를 먼저 기동하고, 준비되면 Flutter UI를 열고,
UI 종료 시 백엔드도 함께 종료합니다.
"""
from __future__ import annotations

import ctypes
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

BASE_DIR = Path(sys.executable).parent

BACKEND_EXE  = BASE_DIR / "eye_backend" / "eye_backend.exe"
FRONTEND_EXE = BASE_DIR / "eye_frontend" / "eye_project.exe"
BACKEND_HOST = "127.0.0.1"
BACKEND_PORT = 8000
HEALTH_URL   = f"http://{BACKEND_HOST}:{BACKEND_PORT}/health"

# AI 모델 로드에 시간이 걸리므로 충분히 여유 있게 설정
STARTUP_TIMEOUT = 120


def _msgbox(title: str, message: str, *, error: bool = False) -> None:
    """Windows 메시지박스를 표시한다. tkinter 의존성 없이 ctypes 사용."""
    icon = 0x10 if error else 0x40  # MB_ICONERROR / MB_ICONINFORMATION
    ctypes.windll.user32.MessageBoxW(0, message, title, icon)


def _port_in_use(port: int) -> bool:
    """포트가 이미 사용 중인지 확인한다."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("127.0.0.1", port)) == 0


def _wait_for_backend(timeout: int) -> bool:
    """/health 가 200을 반환할 때까지 1초 간격으로 폴링한다."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(HEALTH_URL, timeout=2) as r:
                if r.status == 200:
                    return True
        except Exception:
            pass
        time.sleep(1)
    return False


def main() -> None:
    # 1. 필수 실행파일 존재 확인
    for exe in (BACKEND_EXE, FRONTEND_EXE):
        if not exe.exists():
            _msgbox(
                "EyeProject 오류",
                f"필수 파일을 찾을 수 없습니다:\n{exe}\n\n"
                "프로그램을 다시 설치해 주세요.",
                error=True,
            )
            sys.exit(1)

    # 2. 포트 중복 확인
    #    이미 백엔드가 떠 있으면 (이중 실행 등) UI만 바로 열고 종료
    if _port_in_use(BACKEND_PORT):
        frontend_proc = subprocess.Popen(
            [str(FRONTEND_EXE)],
            cwd=str(FRONTEND_EXE.parent),
        )
        frontend_proc.wait()
        return

    # 3. 백엔드 기동
    #    CREATE_NO_WINDOW: 백엔드 콘솔 창이 사용자에게 보이지 않도록 숨김
    backend_proc = subprocess.Popen(
        [str(BACKEND_EXE)],
        cwd=str(BACKEND_EXE.parent),
        creationflags=subprocess.CREATE_NO_WINDOW,
    )

    # 4. 백엔드 준비 대기 (AI 모델 로드 포함)
    if not _wait_for_backend(STARTUP_TIMEOUT):
        backend_proc.kill()
        _msgbox(
            "EyeProject 오류",
            f"서버가 {STARTUP_TIMEOUT}초 내에 시작되지 않았습니다.\n\n"
            "문제가 지속되면 관리자에게 문의하세요.",
            error=True,
        )
        sys.exit(1)

    # 5. Flutter UI 기동
    frontend_proc = subprocess.Popen(
        [str(FRONTEND_EXE)],
        cwd=str(FRONTEND_EXE.parent),
    )

    # 6. UI가 닫힐 때까지 대기 → 이후 백엔드 정리 종료
    frontend_proc.wait()
    backend_proc.terminate()
    try:
        backend_proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        backend_proc.kill()


if __name__ == "__main__":
    main()
