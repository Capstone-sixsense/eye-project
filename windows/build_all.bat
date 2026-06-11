@echo off
SETLOCAL
chcp 65001 > nul

SET ROOT=%~dp0..
SET WINDOWS_DIR=%ROOT%\windows
SET DIST=%WINDOWS_DIR%\dist
SET VENV=%ROOT%\.venv-win\Scripts\activate.bat
SET VENV_PYTHON=%ROOT%\.venv-win\Scripts\python.exe
SET DOWNLOADS=%USERPROFILE%\Downloads
SET ARTIFACT_ZIP=%DOWNLOADS%\eye-frontend-windows.zip
SET ARTIFACT_DIR=%DOWNLOADS%\eye-frontend-windows
SET PYTHON_CMD=py -3.11
SET PYTHONUTF8=1

py -3.11 --version > nul 2> nul
if errorlevel 1 (
    where python > nul 2> nul
    if errorlevel 1 (
        echo ERROR: Python 3.11+ is required.
        echo Install Python 3.11 or 3.12, then re-run this script.
        exit /b 1
    )
    SET "PYTHON_CMD=python"
)

REM VERSION 파일에서 버전 읽기
SET /P APP_VERSION=<"%WINDOWS_DIR%\VERSION"
SET APP_VERSION=%APP_VERSION: =%

echo.
echo ============================================
echo   Eye Project Windows Build Script
echo ============================================
echo.

REM ===========================================================
REM [1/7] source snapshot
REM ===========================================================
echo [1/7] Using current local source snapshot...
cd "%ROOT%"
git rev-parse --short HEAD > nul 2> nul
if %ERRORLEVEL% equ 0 (
    FOR /F "tokens=*" %%H IN ('git rev-parse --short HEAD') DO echo   Git HEAD: %%H
)
echo [1/7] Source snapshot ready.
echo.

REM ===========================================================
REM [2/7] deployment artifact preflight
REM ===========================================================
echo [2/7] Checking deployment artifacts...
powershell -NoProfile -ExecutionPolicy Bypass -File "%WINDOWS_DIR%\preflight.ps1" -Root "%ROOT%"
if %ERRORLEVEL% neq 0 (
    echo ERROR: deployment artifact preflight failed.
    exit /b 1
)
echo [2/7] Deployment artifacts ready.
echo.

REM ===========================================================
REM [3/7] Virtual environment setup
REM ===========================================================
echo [3/7] Setting up virtual environment...

if not exist "%VENV_PYTHON%" (
    echo   Creating virtual environment...
    %PYTHON_CMD% -m venv "%ROOT%\.venv-win"
    if %ERRORLEVEL% neq 0 (
        echo ERROR: Failed to create virtual environment.
        echo Make sure Python 3.11 or 3.12 is installed.
        exit /b 1
    )
    echo   Installing dependencies ^(this may take 10-20 minutes^)...
    call "%VENV%"
    python -X utf8 -m pip install --upgrade pip
    if %ERRORLEVEL% neq 0 (
        echo ERROR: pip upgrade failed.
        exit /b 1
    )
    python -X utf8 -m pip install -r "%WINDOWS_DIR%\requirements_win.txt"
    if %ERRORLEVEL% neq 0 (
        echo ERROR: dependency install failed.
        exit /b 1
    )
    python -X utf8 -m pip install pyinstaller
    if %ERRORLEVEL% neq 0 (
        echo ERROR: pyinstaller install failed.
        exit /b 1
    )
    echo   Dependencies installed.
) else (
    echo   Virtual environment already exists. Skipping install.
    call "%VENV%"
)
echo [3/7] Virtual environment ready.
echo.

REM ===========================================================
REM [4/7] Frontend artifact
REM ===========================================================
echo [4/7] Setting up frontend (eye_frontend)...

if not exist "%DIST%\eye_frontend\eye_project.exe" (
    echo   eye_frontend not found in dist. Looking in Downloads...

    if exist "%ARTIFACT_DIR%\eye_project.exe" (
        echo   Found extracted artifact. Copying...
        mkdir "%DIST%\eye_frontend" 2>nul
        xcopy /E /I /Y "%ARTIFACT_DIR%\*" "%DIST%\eye_frontend\" > nul
        echo   Copied from %ARTIFACT_DIR%
    ) else if exist "%ARTIFACT_ZIP%" (
        echo   Found zip. Extracting...
        powershell -Command "Expand-Archive -Path '%ARTIFACT_ZIP%' -DestinationPath '%ARTIFACT_DIR%' -Force"
        mkdir "%DIST%\eye_frontend" 2>nul
        xcopy /E /I /Y "%ARTIFACT_DIR%\*" "%DIST%\eye_frontend\" > nul
        echo   Extracted and copied.
    ) else (
        echo.
        echo ERROR: eye_frontend not found.
        echo.
        echo   1. Go to GitHub ^> Actions ^> Build Flutter Windows
        echo   2. Download 'eye-frontend-windows' artifact
        echo   3. Place the zip file here:
        echo      %ARTIFACT_ZIP%
        echo   4. Re-run this script.
        echo.
        exit /b 1
    )
)
echo [4/7] Frontend ready.
echo.

REM ===========================================================
REM [5/7] PyInstaller - Backend
REM ===========================================================
echo [5/7] Building backend with PyInstaller...
cd "%ROOT%"
pyinstaller windows\backend.spec --distpath windows\dist --noconfirm
if %ERRORLEVEL% neq 0 (
    echo ERROR: Backend build failed.
    exit /b 1
)
echo [5/7] Backend build complete.
echo.

REM ===========================================================
REM [6/7] PyInstaller - Launcher
REM ===========================================================
echo [6/7] Building launcher with PyInstaller...
pyinstaller windows\launcher.spec --distpath windows\dist --noconfirm
if %ERRORLEVEL% neq 0 (
    echo ERROR: Launcher build failed.
    exit /b 1
)
echo [6/7] Launcher build complete.
echo.

REM ===========================================================
REM [7/7] Inno Setup
REM ===========================================================
echo [7/7] Building installer with Inno Setup...

SET ISCC_EXE=C:\Program Files (x86)\Inno Setup 6\ISCC.exe
if not exist "%ISCC_EXE%" (
    SET ISCC_EXE=%LOCALAPPDATA%\Programs\Inno Setup 6\ISCC.exe
)
if not exist "%ISCC_EXE%" (
    echo ERROR: Inno Setup 6 not found.
    echo Download from: https://jrsoftware.org/isdl.php
    exit /b 1
)

cd "%WINDOWS_DIR%"
"%ISCC_EXE%" /DAppVersion=%APP_VERSION% installer.iss
if %ERRORLEVEL% neq 0 (
    echo ERROR: Inno Setup compile failed.
    exit /b 1
)
echo [7/7] Installer build complete.
echo.

echo ============================================
echo   Done.  (version: %APP_VERSION%)
echo   Output: windows\installer_output\eye_project_setup_v%APP_VERSION%.exe
echo ============================================

ENDLOCAL
