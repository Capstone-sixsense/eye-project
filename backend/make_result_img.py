"""
GradCAM 오버레이 등 분석 결과 이미지를 PNG 바이트로 직렬화한다.
디스크 저장(암호화 포함)은 history.save_report_image가 담당.
"""

from __future__ import annotations

import io
from PIL import Image


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
