from fastapi import FastAPI, File, UploadFile
import os
import shutil


# Temporary placeholder app so the backend container can start before
# the real API and model integration are implemented.
app = FastAPI(title="eye-project backend")


@app.get("/")
def read_root() -> dict[str, str]:
    return {"message": "temporary backend is running"}


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/analyze")
def analyze_image(file: UploadFile = File(...)):

    raw_result = check_image_quality("storage", "normal_01.png")

    report = PassNonPass(raw_result["issues_found"])

    if not report['is_acceptable']:
        return report # AI 모델을 실행하지 않고 여기서 응답을 끝냅니다
    
    # 합격(True)일 경우에만 실행되는 AI 분석 영역
    # 여기에 실제 AI 모델(예: 안질환 감지, 객체 인식 등) 로직을 넣습니다.
    ai_prediction = run_ai_inference(save_path) # 가정된 AI 실행 함수

    #AI 분석 결과를 프론트로 넘김
    return ai_prediction
