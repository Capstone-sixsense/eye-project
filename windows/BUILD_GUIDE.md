# Eye Project Windows Installer Build Guide

이 폴더의 목적은 현재 로컬 repo 상태를 그대로 묶어, Python/Docker/Flutter/Git이 없는 PC에서도 오프라인으로 실행되는 Windows installer를 만드는 것이다.

## 빠른 빌드

```powershell
.\windows\build_all.bat
```

성공하면 아래 파일이 생성된다.

```text
windows\installer_output\eye_project_setup_v1.0.0.exe
```

배포 대상자는 이 installer 하나만 실행하면 된다. 설치 후 바탕화면 또는 시작 메뉴의 `Eye Project`를 실행하면 bundled backend와 frontend가 같이 뜬다.

## 모델만 업데이트할 때

새 모델과 지표가 같은 배포 alias로 갱신되는 경우에는 아래 파일만 교체하고 `build_all.bat`을 다시 실행하면 된다.

```text
ai\artifacts\checkpoints\best.pt
ai\artifacts\evaluations\external_test_<project.version>_best_metrics.json
ai\artifacts\evaluations\xai_<project.version>_lesion_segmentation_test_best_metrics.json
```

`<project.version>`은 `ai\configs\base.yaml`의 `project.version` 값이다. 예를 들어 현재 값이 `v31_v8b_fusion_quickqual_v2`라면 필요한 파일명은 다음과 같다.

```text
ai\artifacts\evaluations\external_test_v31_v8b_fusion_quickqual_v2_best_metrics.json
ai\artifacts\evaluations\xai_v31_v8b_fusion_quickqual_v2_lesion_segmentation_test_best_metrics.json
```

그 다음:

```powershell
.\windows\build_all.bat
```

`build_all.bat`은 실행 초반에 `windows\preflight.ps1`을 호출해서 `base.yaml`, checkpoint, classification metrics, XAI metrics, QuickQual model, HF cache가 모두 있는지 확인한다. 누락되면 installer를 만들지 않고 실패한다.

## 새 버전으로 올릴 때

새 모델 버전명이 바뀌면 다음을 같이 맞춘다.

```text
ai\configs\base.yaml
  project.version
  infer.checkpoint_path
  infer.threshold
  infer.evidence_type
  infer.lesion_threshold
```

기본 배포 checkpoint는 `infer.checkpoint_path: artifacts/checkpoints/best.pt`를 유지하는 것을 권장한다. 그러면 모델 파일은 항상 아래 경로만 교체하면 된다.

```text
ai\artifacts\checkpoints\best.pt
```

classification metrics는 `external_test_<project.version>_best_metrics.json` 이름으로 둔다. 현재 XAI가 lesion segmentation 기반이면 XAI metrics는 `xai_<project.version>_lesion_segmentation_test_best_metrics.json` 이름으로 둔다.

## Frontend 업데이트

frontend가 바뀌지 않았다면 아무것도 하지 않아도 된다. `windows\dist\eye_frontend\eye_project.exe`가 있으면 `build_all.bat`이 그대로 재사용한다.

frontend가 바뀌었다면 GitHub Actions의 `eye-frontend-windows` artifact를 받아서 아래 위치에 둔다.

```text
windows\dist\eye_frontend\
```

또는 zip을 아래 위치에 두면 `build_all.bat`이 자동으로 압축을 풀어 복사한다.

```text
%USERPROFILE%\Downloads\eye-frontend-windows.zip
```

## 빌드 구성

```text
windows\backend.spec       PyInstaller backend bundle 설정
windows\backend_entry.py   frozen exe 전용 backend runtime 설정 진입점
windows\launcher.py        backend health 확인 후 frontend 실행
windows\launcher.spec      launcher exe 설정
windows\installer.iss      Inno Setup installer 설정
windows\build_all.bat      one-click build entrypoint
windows\preflight.ps1      배포 artifact 누락 검증
windows\requirements_win.txt
windows\VERSION
windows\hooks\rthook_torch_home.py
```

`windows\dist\`, `windows\installer_output\`, `.venv-win\`은 빌드 산출물이며 git에 올리지 않는다.

실행 중 생성되는 설정, history DB, 암호화 키, 예측 결과, heatmap, HuggingFace/Torch cache는 `%APPDATA%\EyeProject\` 아래에 둔다. 설치 폴더가 읽기 전용이어도 bundled seed cache를 AppData로 복사해서 오프라인 실행한다.

## 검증 기준

최소 검증은 Docker daemon이 꺼진 상태에서 한다.

1. installer 실행
2. 설치된 `EyeProject.exe` 실행
3. backend `/health`가 `diagnosis_model=ok`, `quickqual=ok`
4. `/deploy-metric`에서 classification/XAI metrics 응답
5. IDRiD 샘플 1회 `/analyze`
6. raw/report image URL이 200으로 응답

이 조건을 통과해야 “아무것도 설치되지 않은 PC에서 오프라인으로 동작하는 installer”로 본다.
