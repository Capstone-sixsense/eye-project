import datetime
import os
from PIL import Image, ImageDraw, ImageFont

def create_medical_report_image(original_filename, ai_image, metrics):
    """
    시간 정보와 원본 파일명으로 분석 보고서를 생성 및 저장합니다.
    """
    #현재 날짜 및 시간 정보 생성
    now = datetime.datetime.now()
    date_str = now.strftime("%Y-%m-%d")    # 폴더 구분용 (예: 2026-03-30)
    timestamp = now.strftime("%H%M%S")      # 파일 구분용 (예: 224930)
    
    #저장 폴더 준비 (results/2026-03-30)
    result_dir = os.path.join("results", date_str)
    os.makedirs(result_dir, exist_ok=True)
    
    #고유한 파일명 생성 (시간_원본파일명)
    # 예: 224930_patient_A.png
    new_filename = f"{timestamp}_{original_filename}"
    save_path = os.path.join(result_dir, new_filename)

    #이미지 합성 로직 (AI 이미지 + 지표 패널)
    img_width, img_height = ai_image.size
    panel_width = 450
    canvas = Image.new("RGB", (img_width + panel_width, img_height), (255, 255, 255))
    canvas.paste(ai_image, (0, 0))
    #이미지 그리기
    draw = ImageDraw.Draw(canvas)
    
    #폰트 로드 (경로는 사용자님의 환경에 맞춰 조정하세요)
    try:
        title_font = ImageFont.truetype("Roboto-Bold.ttf", 32)
        metric_font = ImageFont.truetype("Roboto-Bold.ttf", 24)
        desc_font = ImageFont.truetype("Roboto-Regular.ttf", 16)
    except:
        title_font = metric_font = desc_font = ImageFont.load_default()

    x_offset = img_width + 40
    y_offset = 60

    #텍스트 그리기
    draw.text((x_offset, y_offset), "[ Medical AI Report ]", font=title_font, fill=(0, 0, 0))
    y_offset += 100

    report_items = [
        ("Accuracy", metrics.get('accuracy', 0.0), "전체 성능 판단"),
        ("Precision", metrics.get('precision', 0.0), "불필요 오진 최소화"),
        ("Sensitivity", metrics.get('recall', 0.0), "놓치는 환자 최소화"),
        ("Specificity", metrics.get('specificity', 0.0), "정상 오진 방지"),
        ("F1-score", metrics.get('f1', 0.0), "정밀도와 재현율 조화")
    ]

    for label, value, desc in report_items:
        draw.text((x_offset, y_offset), f"{label}: {value*100:.1f}%", font=metric_font, fill=(0, 51, 102))
        y_offset += 30
        draw.text((x_offset + 10, y_offset), f"({desc})", font=desc_font, fill=(120, 120, 120))
        y_offset += 70

    #최종 이미지 저장
    canvas.save(save_path, "PNG")
    
    return save_path

# ==========================================
# 실행 예시 (테스트용)
if __name__ == "__main__":
    print("🚀 합성 이미지 생성 테스트를 시작합니다...")

    # 가상의 AI 분석 결과 이미지 생성 (실제론 모델의 출력물)
    # 안저 이미지 규격과 유사한 800x800 크기의 배경 생성
    mock_ai_image = Image.new('RGB', (800, 800), color=(30, 30, 30))
    test_draw = ImageDraw.Draw(mock_ai_image)
    test_draw.ellipse((100, 100, 700, 700), outline="red", width=5) # 안저 영역 표시 가정
    test_draw.text((300, 400), "Analyzed Fundus", fill="white")

    # 가상의 AI 분석 지표 데이터
    mock_scores = {
        'accuracy': 0.942,
        'precision': 0.915,
        'recall': 0.958,
        'specificity': 0.920,
        'f1': 0.936
    }

    # 함수 실행
    test_filename = "sample_eye_image.png"
    result_path = create_medical_report_image(test_filename, mock_ai_image, mock_scores)

    print("\n" + "="*40)
    print(f"✅ 테스트 완료!")
    print(f"파일 저장 위치: {result_path}")
    print("="*40)