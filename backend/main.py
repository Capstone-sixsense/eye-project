from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse
import os
import shutil
import logging
import datetime
from PIL import Image
from fastapi.staticfiles import StaticFiles
from image_analyzer import check_image_quality, PassNonPass, resize_image_high_quality
from make_result_img import create_medical_report_image


# 로그 설정
logging.basicConfig(
    filename='server_errors.log',
    level=logging.ERROR,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


# Temporary placeholder app so the backend container can start before
# the real API and model integration are implemented.
app = FastAPI(title="Eye-Project Medical AI Backend")

os.makedirs("storage", exist_ok=True)
os.makedirs("results", exist_ok=True)
app.mount("/storage", StaticFiles(directory="storage"), name="storage") # 원본 접근용 추가
app.mount("/results", StaticFiles(directory="results"), name="results") # 분석 결과 접근용

@app.get("/")
def read_root() -> dict[str, str]:
    return {"message": "temporary backend is running"}


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}

#테스트를 위한 가상의 AI 모델 인터페이스
def run_ai_inference(image_path: str):
    """
    AI 모델 분석 시뮬레이션
    Returns: label, probability, heatmap_image_object, metrics
    """
    # 실제 모델이 반환할 데이터 예시
    predicted_label = "Abnormal (DR)" # 망막병증 의심
    abnormal_probability = 0.88
    quality_warning = "None"
    quality_grade = "Good"
    
    # Grad-CAM 결과물이라고 가정하고 원본을 불러와 가공 (테스트용)
    heatmap_img = Image.open(image_path).convert("RGB")
    
    # 가상의 성능 지표 (합성용)
    mock_metrics = {
        'accuracy': 0.95, 'precision': 0.92, 'recall': 0.96, 'specificity': 0.94, 'f1': 0.94
    }
    
    return {
        "label": predicted_label,
        "probability": abnormal_probability,
        "warning": quality_warning,
        "grade": quality_grade,
        "heatmap": heatmap_img,
        "metrics": mock_metrics
    }


@app.post("/analyze")
async def analyze_image(file: UploadFile = File(...)):
    UPLOAD_DIR = "storage"
    
    # 파일 경로 설정 (원본은 raw_ 접두사를 붙여 관리)
    raw_save_path = os.path.join(UPLOAD_DIR, f"raw_{file.filename}")
    processed_path = os.path.join(UPLOAD_DIR, file.filename)

    try:
        #프론트에서 넘어온 이미지 임시 저장 (원본)
        with open(raw_save_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        #고품질 리사이징 수행 (LANCZOS 필터 사용)
        resize_image_high_quality(raw_save_path, processed_path, target_size=(448, 448))

        #품질 분석 및 통과 여부 판단 (image_analyzer.py)
        raw_result = check_image_quality(UPLOAD_DIR, file.filename)
        report = PassNonPass(raw_result)

        if not report['is_acceptable']:
            # 적합하지 않을 경우 보안을 위해 원본과 중간 생성물 모두 삭제
            for path in [raw_save_path, processed_path]:
                if os.path.exists(path): os.remove(path)
            
            return {
                "status": "fail",
                "message": "적합하지 않은 이미지입니다.",
                "details": report['messages']
            }

        #품질 통과 시, AI 모델 분석 수행
        # run_ai_inference는 시뮬레이션 함수임
        ai_result = run_ai_inference(processed_path)
        
        #의료 데이터가 합성된 리포트 이미지 생성 (Heatmap + Metrics)
        #make_result_img.py의 create_medical_report_image 호출
        final_report_path = create_medical_report_image(
            original_filename=file.filename,
            ai_image=ai_result['heatmap'],
            metrics=ai_result['metrics']
        )

        #중간 생성물만 삭제하고 원본은 남깁니다.
        #AI 모델 입력 규격에 맞게 리사이징했던 processed_path만 지웁니다.
        if os.path.exists(processed_path):
            os.remove(processed_path)
        
        #raw_save_path(원본)는 지우지 않고 storage 폴더에 보존합니다.
        #Flutter에서 언제든 꺼내볼 수 있습니다.

        #원본 이미지와 분석 리포트 이미지의 경로를 모두 반환합니다.
        return {
            "status": "success",
            "predicted_label": ai_result["label"],
            "abnormal_probability": ai_result["probability"],
            "quality_grade": ai_result["grade"],
            # Flutter(Dio)에서는 이 경로들을 받아 화면에 표시합니다.
            "original_url": raw_save_path, # 원본 이미지 파일 경로 (storage/raw_파일명)
            "report_url": final_report_path # 분석 결과 리포트 경로 (results/날짜/파일명)
        }

    except Exception as e:
        # 에러 발생 시 데이터 유출 방지를 위해 임시 파일 보안 삭제
        if os.path.exists(raw_save_path): os.remove(raw_save_path)
        if os.path.exists(processed_path): os.remove(processed_path)
        
        logging.error(f"서버 에러 발생: {str(e)}")
        raise HTTPException(status_code=500, detail="서버 분석 중 오류가 발생했습니다.")