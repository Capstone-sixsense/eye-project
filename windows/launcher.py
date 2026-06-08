"""
EyeProject Windows 런처

백엔드(FastAPI)를 먼저 기동하고, 준비되면 Flutter UI를 열고,
UI 종료 시 백엔드도 함께 종료합니다.
"""
from __future__ import annotations

import ctypes
import ctypes.wintypes
import socket
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

BASE_DIR = Path(sys.executable).parent

BACKEND_EXE  = BASE_DIR / "eye_backend" / "eye_backend.exe"
FRONTEND_EXE = BASE_DIR / "eye_frontend" / "eye_project.exe"
BACKEND_HOST = "127.0.0.1"
BACKEND_PORT = 8000
HEALTH_URL   = f"http://{BACKEND_HOST}:{BACKEND_PORT}/health"
STARTUP_TIMEOUT = 120

# 로그 파일: 백엔드 기동 실패 시 원인 추적용
LOG_FILE = BASE_DIR / "eye_backend" / "launcher.log"


def _log(msg: str) -> None:
    """런처 동작 기록 — 문제 발생 시 원인 파악용."""
    try:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}\n")
    except Exception:
        pass


def _msgbox(title: str, message: str, *, error: bool = False) -> None:
    icon = 0x10 if error else 0x40
    ctypes.windll.user32.MessageBoxW(0, message, title, icon)


def _port_in_use(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("127.0.0.1", port)) == 0


# ---------------------------------------------------------------
# 로딩 창 (백엔드 시작 대기 중 사용자에게 진행 상황 표시)
# ---------------------------------------------------------------
_LOADING_HWND = None
_WM_CLOSE = 0x0010
_WM_SETTEXT = 0x000C


def _loading_window_thread() -> None:
    """별도 스레드에서 단순 로딩 창을 띄운다."""
    global _LOADING_HWND

    user32 = ctypes.windll.user32
    hinstance = ctypes.windll.kernel32.GetModuleHandleW(None)

    class WNDCLASS(ctypes.Structure):
        _fields_ = [
            ("style", ctypes.c_uint),
            ("lpfnWndProc", ctypes.c_void_p),
            ("cbClsExtra", ctypes.c_int),
            ("cbWndExtra", ctypes.c_int),
            ("hInstance", ctypes.c_void_p),
            ("hIcon", ctypes.c_void_p),
            ("hCursor", ctypes.c_void_p),
            ("hbrBackground", ctypes.c_void_p),
            ("lpszMenuName", ctypes.c_wchar_p),
            ("lpszClassName", ctypes.c_wchar_p),
        ]

    WNDPROC = ctypes.WINFUNCTYPE(
        ctypes.c_long, ctypes.c_void_p, ctypes.c_uint,
        ctypes.c_uint, ctypes.c_long
    )

    def wnd_proc(hwnd, msg, wparam, lparam):
        if msg == _WM_CLOSE:
            user32.DestroyWindow(hwnd)
            return 0
        return user32.DefWindowProcW(hwnd, msg, wparam, lparam)

    proc = WNDPROC(wnd_proc)
    wc = WNDCLASS()
    wc.lpfnWndProc = ctypes.cast(proc, ctypes.c_void_p)
    wc.hInstance = hinstance
    wc.hbrBackground = ctypes.c_void_p(6)  # COLOR_BTNFACE + 1
    wc.lpszClassName = "EyeProjectLoader"

    user32.RegisterClassW(ctypes.byref(wc))

    hwnd = user32.CreateWindowExW(
        0, "EyeProjectLoader", "EyeProject 시작 중",
        0x10CF0000,  # WS_OVERLAPPED | WS_CAPTION | WS_SYSMENU
        0x80000000, 0x80000000,  # CW_USEDEFAULT
        400, 120,
        None, None, hinstance, None
    )

    if not hwnd:
        return

    _LOADING_HWND = hwnd

    # 정적 텍스트 레이블
    user32.CreateWindowExW(
        0, "STATIC",
        "AI 모델을 불러오는 중입니다. 잠시 기다려 주세요...\n(최대 2분 소요)",
        0x50000000,  # WS_CHILD | WS_VISIBLE | SS_CENTER
        20, 20, 360, 50,
        hwnd, None, hinstance, None
    )

    user32.ShowWindow(hwnd, 1)  # SW_SHOWNORMAL
    user32.UpdateWindow(hwnd)

    msg = ctypes.wintypes.MSG()
    while user32.GetMessageW(ctypes.byref(msg), None, 0, 0) > 0:
        user32.TranslateMessage(ctypes.byref(msg))
        user32.DispatchMessageW(ctypes.byref(msg))


def _show_loading() -> None:
    t = threading.Thread(target=_loading_window_thread, daemon=True)
    t.start()
    time.sleep(0.3)  # 창이 뜰 시간 확보


def _close_loading() -> None:
    global _LOADING_HWND
    if _LOADING_HWND:
        ctypes.windll.user32.PostMessageW(_LOADING_HWND, _WM_CLOSE, 0, 0)
        _LOADING_HWND = None
        time.sleep(0.2)


def _health_ok() -> bool:
    try:
        with urllib.request.urlopen(HEALTH_URL, timeout=2) as r:
            return r.status == 200
    except Exception:
        return False


def _wait_for_backend(proc: subprocess.Popen, timeout: int) -> bool:
    """/health 폴링. 백엔드 프로세스가 먼저 종료되면 즉시 False 반환."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        # 백엔드 프로세스가 죽었으면 더 기다려도 소용없음
        if proc.poll() is not None:
            _log(f"eye_backend.exe 조기 종료 (exit code={proc.returncode})")
            return False
        if _health_ok():
            return True
        time.sleep(1)
    return False


def main() -> None:
    _log("=== 런처 시작 ===")

    # 1. 필수 파일 존재 확인
    for exe in (BACKEND_EXE, FRONTEND_EXE):
        if not exe.exists():
            _log(f"필수 파일 없음: {exe}")
            _msgbox(
                "EyeProject 오류",
                f"필수 파일을 찾을 수 없습니다:\n{exe}\n\n프로그램을 다시 설치해 주세요.",
                error=True,
            )
            sys.exit(1)

    # 2. If port 8000 is already occupied, use it only when it is a healthy
    # EyeProject-compatible backend. Fresh offline-installer validation should
    # still be done with Docker/dev servers stopped.
    if _port_in_use(BACKEND_PORT):
        if _health_ok():
            _log("기존 127.0.0.1:8000 백엔드 사용")
            subprocess.Popen(
                [str(FRONTEND_EXE)],
                cwd=str(FRONTEND_EXE.parent),
            ).wait()
            _log("=== 런처 종료 ===")
            return
        _log("포트 8000 사용 중이나 /health 실패")
        _msgbox(
            "EyeProject 오류",
            "127.0.0.1:8000 포트가 이미 사용 중이지만 EyeProject 서버가 아닙니다.\n\n"
            "해당 포트를 사용하는 프로그램을 종료한 뒤 다시 실행해 주세요.",
            error=True,
        )
        sys.exit(1)

    # 3. 로딩 창 표시 (사용자가 진행 중임을 알 수 있도록)
    _show_loading()
    _log("로딩 창 표시")

    # 4. 백엔드 기동 (콘솔 창 숨김)
    _log(f"eye_backend.exe 기동: {BACKEND_EXE}")
    backend_proc = subprocess.Popen(
        [str(BACKEND_EXE)],
        cwd=str(BACKEND_EXE.parent),
        creationflags=subprocess.CREATE_NO_WINDOW,
    )

    # 5. 백엔드 준비 대기
    ready = _wait_for_backend(backend_proc, STARTUP_TIMEOUT)
    _close_loading()

    if not ready:
        backend_proc.kill()
        exit_code = backend_proc.returncode
        _log(f"백엔드 시작 실패 (exit code={exit_code})")
        _msgbox(
            "EyeProject 오류",
            f"서버가 {STARTUP_TIMEOUT}초 내에 시작되지 않았습니다.\n\n"
            f"로그 파일: {LOG_FILE}\n\n"
            "문제가 지속되면 관리자에게 문의하세요.",
            error=True,
        )
        sys.exit(1)

    _log("백엔드 준비 완료 → UI 실행")

    # 6. Flutter UI 실행
    frontend_proc = subprocess.Popen(
        [str(FRONTEND_EXE)],
        cwd=str(FRONTEND_EXE.parent),
    )

    frontend_proc.wait()
    backend_proc.terminate()
    try:
        backend_proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        backend_proc.kill()

    _log("=== 런처 종료 ===")


if __name__ == "__main__":
    main()
