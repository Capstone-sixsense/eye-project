"""
GradCAM 오버레이 등 분석 결과 이미지를 PNG 바이트로 직렬화한다.
디스크 저장(암호화 포함)은 history.save_report_image가 담당.
"""

from __future__ import annotations

import io 
import os
from PIL import Image


RESULTS_DIR = "results"

def render_report_image_bytes(ai_image: Image.Image) -> bytes:
    """
    PIL 이미지를 PNG 바이트로 직렬화해서 반환.
    - 패널/지표 텍스트 합성은 하지 않는다.
    - RGBA → RGB 변환으로 호환성 확보.
    """
    if ai_image.mode != "RGB":
        ai_image = ai_image.convert("RGB")
    buf = io.BytesIO()
    ai_image.save(buf, format="PNG")
    return buf.getvalue()

# ---------------------------------------------------------------
# create_medical_report_image
# ---------------------------------------------------------------
# 분석 결과 이미지(GradCAM 오버레이 등)를 디스크에 저장하고 그 경로를 돌려준다.
#
# [역할]
#   - AI 추론 결과로 만들어진 PIL 이미지를 받아서 영구 저장한다.
#   - 파일 경로 규약을 history 모듈과 일치시켜, 나중에 record_id 하나로
#     원본/결과/메타데이터를 한 번에 묶어 조회할 수 있게 한다.
#
# [동작 순서]
#   1) record_id의 앞 8자리(YYYYMMDD)를 'YYYY-MM-DD' 폴더명으로 변환
#      예: '20260428_165403_123' -> '2026-04-28'
#   2) results/<날짜폴더>/ 디렉토리를 만든다 (이미 있으면 그대로 사용).
#   3) 저장할 파일명을 'report_<record_id>.png' 로 결정.
#   4) 이미지 모드가 RGB가 아니면(예: RGBA, P) RGB로 변환.
#      - PNG에 알파 채널이 있으면 일부 뷰어에서 깨져 보일 수 있어 안전성 확보.
#   5) PNG 포맷으로 저장하고 상대 경로 문자열을 반환.
#
# [반환값]
#   저장된 파일의 상대 경로 (예: 'results/2026-04-28/report_20260428_165403_123.png')
#   - main.py에서는 이 경로를 응답 JSON의 report_url로,
#     history.save_metadata에는 메타데이터의 report_url로 그대로 사용한다.
#
# [전제]
#   - record_id는 'YYYYMMDD_HHMMSS_mmm' 형식이어야 한다 (history.make_record_id가 보장).
#   - 본 함수는 metrics나 텍스트 패널 합성을 하지 않는다.
#     해석 가능한 점수 데이터는 history.save_metadata가 별도 JSON으로 보관한다.
def create_medical_report_image(
    ai_image: Image.Image,
    record_id: str,
) -> str:
    """
    분석 결과 이미지를 results/<YYYY-MM-DD>/report_<record_id>.png 로 저장.

    Args:
        ai_image: 저장할 이미지 (보통 GradCAM 오버레이가 적용된 PIL 이미지)
        record_id: history.make_record_id()로 발급된 ID
                   예: '20260428_165403_123'

    Returns:
        저장된 파일의 상대 경로
        예: 'results/2026-04-28/report_20260428_165403_123.png'
    """
    # record_id의 앞 8자리(YYYYMMDD)를 폴더 이름(YYYY-MM-DD)으로 변환
    date_str = f"{record_id[0:4]}-{record_id[4:6]}-{record_id[6:8]}"

    result_dir = os.path.join(RESULTS_DIR, date_str)
    os.makedirs(result_dir, exist_ok=True)

    save_path = os.path.join(result_dir, f"report_{record_id}.png")

    # PIL Image가 RGBA 등일 수 있으므로 안전하게 RGB로 변환 후 저장
    if ai_image.mode != "RGB":
        ai_image = ai_image.convert("RGB")
    ai_image.save(save_path, "PNG")

    return save_path


# ==========================================
# 실행 예시 (테스트용)
if __name__ == "__main__":
    import datetime

    print("🚀 record_id 기반 리포트 저장 테스트를 시작합니다...")

    target_filename = "normal_01.png"
    storage_path = os.path.join("storage", target_filename)

    if not os.path.exists(storage_path):
        print(f"❌ 에러: {storage_path} 경로에 파일이 없습니다.")
        print("💡 storage 폴더에 테스트용 이미지를 먼저 넣어주세요!")
    else:
        try:
            real_image = Image.open(storage_path).convert("RGB")

            now = datetime.datetime.now()
            test_record_id = (
                now.strftime("%Y%m%d_%H%M%S_") + f"{now.microsecond // 1000:03d}"
            )

            result_path = create_medical_report_image(
                ai_image=real_image,
                record_id=test_record_id,
            )

            print("\n" + "=" * 40)
            print("✅ 테스트 완료!")
            print(f"원본: {storage_path}")
            print(f"저장: {result_path}")
            print(f"record_id: {test_record_id}")
            print("=" * 40)

        except Exception as e:
            print(f"❌ 이미지 처리 중 오류 발생: {e}")