# Eye Project Windows 배포 가이드

## 빠른 시작

배포용 설치파일을 만들려면 아래 두 단계만 실행합니다.

**사전 조건:** `eye_frontend\`가 준비되어 있어야 합니다.
→ GitHub Actions에서 빌드 완료 후 Artifact를 `windows\dist\eye_frontend\`에 복사

```powershell
# 가상환경 활성화
.venv-win\Scripts\Activate.ps1

# 전체 빌드 실행
.\windows\build_all.bat
```

완료 후 `windows\installer_output\eye_project_setup_v1.0.0.exe` 파일이 생성됩니다.
이 파일 하나를 배포하면 됩니다.

---

## 처음 설정 (최초 1회)

### 필요한 도구

| 도구 | 역할 | 설치 위치 |
|---|---|---|
| Python 3.11 | 백엔드 실행 환경 | python.org |
| Inno Setup 6 | 설치파일(.exe) 생성 | jrsoftware.org/isdl.php |
| Git | 코드 관리 | git-scm.com |

### 가상환경 및 의존성 설치

```powershell
cd D:\eye-project

# Python 3.11 가상환경 생성
py -3.11 -m venv .venv-win

# 활성화
.venv-win\Scripts\Activate.ps1

# 의존성 설치 (PyTorch CPU 포함, 10~20분 소요)
pip install -r windows\requirements_win.txt
pip install pyinstaller
```

---

## 빌드 구성 요소

### 1. Frontend — Flutter Windows 앱 (GitHub Actions 자동 빌드)

Flutter로 작성된 UI 앱입니다.

- **빌드 방식:** WSL2에서 `window` 브랜치에 push하면 GitHub Actions가 자동으로 Windows 빌드를 실행합니다.
- **결과물:** `eye_project.exe` + `flutter_windows.dll` + `data\`
- **배치 위치:** `windows\dist\eye_frontend\`

GitHub Actions 빌드 결과 받는 방법:
```
GitHub 저장소 → Actions 탭 → Build Flutter Windows
→ 완료된 실행 클릭 → Artifacts → eye-frontend-windows 다운로드
→ 압축 해제 후 windows\dist\eye_frontend\ 에 복사
```

---

### 2. Backend — FastAPI 서버 (PyInstaller)

Python으로 작성된 AI 추론 서버입니다. FastAPI 위에서 동작하며 이미지를 받아 분석 결과를 반환합니다.

- **빌드 도구:** PyInstaller
- **설정 파일:** `windows\backend.spec`
- **결과물:** `windows\dist\eye_backend\` 폴더 (exe + DLL + AI 모델)
- **포함 내용:**
  - `eye_backend.exe` — FastAPI 서버 실행파일
  - `configs\` — AI 모델 설정 YAML 파일
  - `models\` — QuickQual SVM 모델 (.pkl)
  - PyTorch, OpenCV 등 DLL 파일들

**PyInstaller란:** Python 스크립트와 모든 의존성(라이브러리, DLL)을 하나의 폴더로 패키징하는 도구입니다. 대상 PC에 Python이 설치되어 있지 않아도 실행 가능합니다.

---

### 3. Launcher — 기동 관리자 (PyInstaller)

백엔드와 프론트엔드를 순서대로 실행하고 종료를 관리하는 작은 프로그램입니다.

- **빌드 도구:** PyInstaller (one-file 모드)
- **설정 파일:** `windows\launcher.spec`
- **결과물:** `windows\dist\EyeProject.exe` (단일 파일)
- **동작 순서:**
  1. `eye_backend\eye_backend.exe` 백그라운드 실행
  2. `http://127.0.0.1:8000/health` 폴링 (최대 120초 대기)
  3. 서버 준비 완료 → `eye_frontend\eye_project.exe` 실행
  4. UI 종료 시 백엔드도 함께 종료

사용자는 `EyeProject.exe` 하나만 실행하면 됩니다.

---

### 4. Installer — 설치파일 생성 (Inno Setup)

위 3개의 결과물을 하나의 설치파일로 묶습니다.

- **빌드 도구:** Inno Setup 6
- **설정 파일:** `windows\installer.iss`
- **결과물:** `windows\installer_output\eye_project_setup_v1.0.0.exe`
- **설치 시 처리 내용:**
  - `C:\Program Files\EyeProject\`에 파일 설치
  - 바탕화면 단축키 생성
  - 시작 메뉴 등록
  - 설정 → 앱에서 제거 가능

---

## 업데이트 반영 방법

### Frontend가 업데이트된 경우

```
WSL2에서
  1. git merge origin/frontend   # 최신 frontend 코드 반영
  2. git push origin window      # push → GitHub Actions 자동 빌드 시작
  3. (빌드 완료 대기, 약 5~10분)

Windows에서
  4. GitHub Actions에서 새 Artifact 다운로드
  5. 기존 eye_frontend\ 교체:
       Remove-Item "windows\dist\eye_frontend\*" -Recurse -Force
       xcopy /E /I /Y "다운로드경로\eye-frontend-windows\*" "windows\dist\eye_frontend\"
  6. .\windows\build_all.bat 실행   # Inno Setup만 재실행해도 무방
```

---

### AI / Backend가 업데이트된 경우

```
WSL2에서
  1. git merge origin/ai      # 또는 git merge origin/backend
  2. git push origin window

Windows에서
  3. git pull origin window
  4. 가상환경 활성화: .venv-win\Scripts\Activate.ps1
  5. .\windows\build_all.bat 실행
     (PyInstaller가 백엔드를 새로 빌드하고 Inno Setup으로 패키징)
```

---

### 둘 다 업데이트된 경우

Frontend와 AI/Backend가 동시에 업데이트된 경우:

```
WSL2에서
  1. git merge origin/frontend
  2. git merge origin/ai
  3. git push origin window

Windows에서
  4. GitHub Actions 빌드 완료 대기 → Artifact 교체
  5. git pull origin window
  6. .venv-win\Scripts\Activate.ps1
  7. .\windows\build_all.bat
```

---

## 폴더 구조 요약

```
D:\eye-project\
  windows\
    dist\                         <- 빌드 결과물 (build_all.bat이 생성)
      EyeProject.exe              <- 런처 (사용자가 실행하는 파일)
      eye_backend\                <- PyInstaller 패키징된 백엔드
      eye_frontend\               <- Flutter 빌드 결과물
    installer_output\
      eye_project_setup_v1.0.0.exe  <- 최종 배포 설치파일
    backend.spec                  <- PyInstaller 백엔드 설정
    launcher.spec                 <- PyInstaller 런처 설정
    installer.iss                 <- Inno Setup 설정
    build_all.bat                 <- 빌드 자동화 스크립트
    requirements_win.txt          <- Python 의존성 목록
  .venv-win\                      <- Python 가상환경 (git 미포함)
```
