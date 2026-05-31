@echo off
SETLOCAL
chcp 65001 > nul

SET ROOT=%~dp0..
SET WINDOWS_DIR=%ROOT%\windows
SET DIST=%WINDOWS_DIR%\dist
SET VENV=%ROOT%\.venv-win\Scripts\activate.bat

echo.
echo ============================================
echo   Eye Project Windows Build Script
echo ============================================
echo.

REM --- 0. Check prerequisites ---

echo [0/4] Checking prerequisites...

if not exist "%VENV%" (
    echo ERROR: Virtual environment not found.
    echo Run: py -3.11 -m venv .venv-win
    exit /b 1
)

if not exist "%DIST%\eye_frontend\eye_project.exe" (
    echo ERROR: eye_frontend not found.
    echo Download artifact from GitHub Actions and place it in:
    echo   %DIST%\eye_frontend\
    exit /b 1
)

if not exist "C:\Program Files (x86)\Inno Setup 6\ISCC.exe" (
    echo ERROR: Inno Setup 6 not found.
    echo Download from: https://jrsoftware.org/isdl.php
    exit /b 1
)

echo [0/4] OK.
echo.

REM --- 1. Activate virtual environment ---

call "%VENV%"

REM --- 2. Backend build (PyInstaller) ---

echo [1/4] Building backend with PyInstaller...
cd "%ROOT%"
pyinstaller windows\backend.spec --distpath windows\dist --noconfirm
if %ERRORLEVEL% neq 0 (
    echo ERROR: Backend build failed.
    exit /b 1
)
echo [1/4] Backend build complete.
echo.

REM --- 3. Launcher build (PyInstaller) ---

echo [2/4] Building launcher with PyInstaller...
pyinstaller windows\launcher.spec --distpath windows\dist --noconfirm
if %ERRORLEVEL% neq 0 (
    echo ERROR: Launcher build failed.
    exit /b 1
)
echo [2/4] Launcher build complete.
echo.

REM --- 4. Verify dist structure ---

echo [3/4] Verifying dist structure...

if not exist "%DIST%\EyeProject.exe" (
    echo ERROR: EyeProject.exe not found in dist\
    exit /b 1
)
if not exist "%DIST%\eye_backend\eye_backend.exe" (
    echo ERROR: eye_backend\eye_backend.exe not found.
    exit /b 1
)
if not exist "%DIST%\eye_frontend\eye_project.exe" (
    echo ERROR: eye_frontend\eye_project.exe not found.
    exit /b 1
)

echo [3/4] Structure OK.
echo.

REM --- 5. Inno Setup compile ---

echo [4/4] Building installer with Inno Setup...
cd "%WINDOWS_DIR%"
"C:\Program Files (x86)\Inno Setup 6\ISCC.exe" installer.iss
if %ERRORLEVEL% neq 0 (
    echo ERROR: Inno Setup compile failed.
    exit /b 1
)
echo [4/4] Installer build complete.
echo.

echo ============================================
echo   Done.
echo   Output: windows\installer_output\eye_project_setup_v1.0.0.exe
echo ============================================

ENDLOCAL
