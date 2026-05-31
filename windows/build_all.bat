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

echo.
echo ============================================
echo   Eye Project Windows Build Script
echo ============================================
echo.

REM ===========================================================
REM [1/6] git pull
REM ===========================================================
echo [1/6] Pulling latest code from window branch...
cd "%ROOT%"
git pull origin window
if %ERRORLEVEL% neq 0 (
    echo ERROR: git pull failed. Check network or branch state.
    exit /b 1
)
echo [1/6] Code up to date.
echo.

REM ===========================================================
REM [2/6] Virtual environment setup
REM ===========================================================
echo [2/6] Setting up virtual environment...

if not exist "%VENV_PYTHON%" (
    echo   Creating virtual environment with Python 3.11...
    py -3.11 -m venv "%ROOT%\.venv-win"
    if %ERRORLEVEL% neq 0 (
        echo ERROR: Failed to create virtual environment.
        echo Make sure Python 3.11 is installed: py -3.11 --version
        exit /b 1
    )
    echo   Installing dependencies ^(this may take 10-20 minutes^)...
    call "%VENV%"
    pip install -r "%WINDOWS_DIR%\requirements_win.txt"
    pip install pyinstaller
    if %ERRORLEVEL% neq 0 (
        echo ERROR: pip install failed.
        exit /b 1
    )
    echo   Dependencies installed.
) else (
    echo   Virtual environment already exists. Skipping install.
    call "%VENV%"
)
echo [2/6] Virtual environment ready.
echo.

REM ===========================================================
REM [3/6] Frontend artifact
REM ===========================================================
echo [3/6] Setting up frontend (eye_frontend)...

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
echo [3/6] Frontend ready.
echo.

REM ===========================================================
REM [4/6] PyInstaller - Backend
REM ===========================================================
echo [4/6] Building backend with PyInstaller...
cd "%ROOT%"
pyinstaller windows\backend.spec --distpath windows\dist --noconfirm
if %ERRORLEVEL% neq 0 (
    echo ERROR: Backend build failed.
    exit /b 1
)
echo [4/6] Backend build complete.
echo.

REM ===========================================================
REM [5/6] PyInstaller - Launcher
REM ===========================================================
echo [5/6] Building launcher with PyInstaller...
pyinstaller windows\launcher.spec --distpath windows\dist --noconfirm
if %ERRORLEVEL% neq 0 (
    echo ERROR: Launcher build failed.
    exit /b 1
)
echo [5/6] Launcher build complete.
echo.

REM ===========================================================
REM [6/6] Inno Setup
REM ===========================================================
echo [6/6] Building installer with Inno Setup...

if not exist "C:\Program Files (x86)\Inno Setup 6\ISCC.exe" (
    echo ERROR: Inno Setup 6 not found.
    echo Download from: https://jrsoftware.org/isdl.php
    exit /b 1
)

cd "%WINDOWS_DIR%"
"C:\Program Files (x86)\Inno Setup 6\ISCC.exe" installer.iss
if %ERRORLEVEL% neq 0 (
    echo ERROR: Inno Setup compile failed.
    exit /b 1
)
echo [6/6] Installer build complete.
echo.

echo ============================================
echo   Done.
echo   Output: windows\installer_output\eye_project_setup_v1.0.0.exe
echo ============================================

ENDLOCAL
