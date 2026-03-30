import datetime

#분석 결과 이미지를 저장하는 함수
def save_result_image(user_id, original_filename, ai_image):
    #현재 시간 기반 폴더 생성 (예: results/2026-03-29)
    date_str = datetime.datetime.now().strftime("%Y-%m-%d")
    result_dir = os.path.join("results", date_str)
    os.makedirs(result_dir, exist_ok=True)
    
    #고유한 파일명 생성 (사용자ID + 시간 + 원본파일명)
    # 예: user123_20260329_203015_patient_A.png
    timestamp = datetime.datetime.now().strftime("%H%M%S")
    new_filename = f"{user_id}_{timestamp}_{original_filename}"
    save_path = os.path.join(result_dir, new_filename)
    
    #이미지 저장
    ai_image.save(save_path)
    return save_path