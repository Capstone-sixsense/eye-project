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

    UPLOAD_DIR = "storage"
    raw_save_path = os.path.join(UPLOAD_DIR, f"raw_{file.filename}")
    processed_path = os.path.join(UPLOAD_DIR, file.filename) # 리사이징된 파일 이름

    # 원본 파일 임시 저장
    with open(raw_save_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    #고품질 리사이징 수행 (예: AI 모델 입력 규격인 512x512 또는 1024x1024)
    resize_image_high_quality(raw_save_path, processed_path, target_size=(1024, 1024))


    try:
        #이미지 분석 실행
        raw_result = check_image_quality("storage", "normal_01.png")

        #이미지 통과여부 판단
        report = PassNonPass(raw_result["issues_found"])

        #품질 필터 통과 실패 시 처리
        if not report['is_acceptable']:
            #보안을 위해 임시로 생성된 모든 파일 삭제
            #리스트를 돌며 원본(raw_)과 가공본 모두 삭제합니다.
            for path in [raw_save_path, processed_path]:
                if os.path.exists(path):
                    os.remove(path)
                    print(f"🗑️ 보안 삭제 완료: {path}")
            
            return report # AI 모델 실행 없이 결과 반환

        # 합격(True)일 경우에만 실행되는 AI 분석 영역
        # 여기에 실제 AI 모델(예: 안질환 감지, 객체 인식 등) 로직을 넣습니다.
        ai_prediction = run_ai_inference(save_path) # 가정된 AI 실행 함수
        
        #AI 분석 결과를 프론트로 넘김
        return {
            "quality_report": report,
            "ai_result": ai_prediction
        }

    except Exception as e:
        #예상치 못한 에러 발생 시에도 파일 삭제 (보안 가드)
        for path in [raw_save_path, processed_path]:
            if os.path.exists(path):
                os.remove(path)
                
        #서버 콘솔/로그파일에는 상세한 에러 내용을 남깁니다 (개발자용)
        logging.error(f"서버 에러 발생: {str(e)}") 
    
        # 보안을 위해 정제된 메시지만 보냅니다 (사용자용)
        raise HTTPException(
            status_code=500, 
            detail="서버 내부 문제로 분석에 실패했습니다. 잠시 후 다시 시도해주세요."
        )
