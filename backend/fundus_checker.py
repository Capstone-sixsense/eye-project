"""
안저(Fundus) 이미지 판별 휴리스틱 필터.

안저 이미지의 시각적 특성:
  1. 원형 밝은 영역 + 사방의 어두운 테두리 (카메라 렌즈 구경 특성)
  2. 붉은-주황 계통의 색조 (망막 조직, 혈관)
  3. 밝은 영역이 바운딩 박스를 완전히 채우지 않음 (원형이므로 fill_ratio ≈ π/4 ≈ 0.785)

비안저 이미지(지도, 일반 사진 등)의 특성:
  1. 어두운 테두리 거의 없음 (dark_ratio ≈ 0%)
  2. 다양한 색상 (지도: 파랑, 초록 등)
  3. 밝은 영역이 프레임 전체를 채움 (fill_ratio ≈ 1.0)

임계값 조정:
  아래 상수를 직접 수정해 민감도를 조정할 수 있다.
  - 안저를 너무 많이 거부 → _MIN_DARK_RATIO를 낮추거나 _MAX_FILL_RATIO를 높인다.
  - 비안저가 통과 → _MIN_DARK_RATIO를 높이거나 _MAX_FILL_RATIO를 낮춘다.
"""
from __future__ import annotations

import numpy as np
from PIL import Image

# ---------- 임계값 ----------
_DARK_PIXEL_THRESH = 15    # 채널 평균이 이 값 미만인 픽셀을 "어두움"으로 판정
_MIN_DARK_RATIO    = 0.05  # 어두운 픽셀 최소 비율 — 미만이면 거부 (테두리 없음)
_MIN_ASPECT_RATIO  = 0.60  # 밝은 영역 단축/장축 비율 — 미만이면 거부 (지나치게 세로/가로)
_MAX_FILL_RATIO    = 0.95  # 밝은 픽셀/바운딩박스 비율 — 초과하면 거부 (프레임을 가득 채움)
# ----------------------------


def check_fundus_heuristics(img: Image.Image) -> tuple[bool, str, dict]:
    """
    이미지가 안저 이미지인지 휴리스틱으로 판별한다.
    전처리 전 원본 이미지에 적용하도록 설계되었다.

    Returns:
        (is_fundus, reject_reason, details)
        - is_fundus     : True이면 안저 이미지로 판단
        - reject_reason : 거부 이유 코드 (통과 시 빈 문자열)
        - details       : 각 지표 수치 딕셔너리 (로깅·디버깅용)
    """
    arr = np.array(img.convert("RGB"), dtype=np.float32)
    gray = arr.mean(axis=2)  # (H, W) 채널 평균 밝기

    dark_mask   = gray < _DARK_PIXEL_THRESH
    bright_mask = ~dark_mask
    dark_ratio  = float(dark_mask.mean())

    bright_rows = np.where(bright_mask.any(axis=1))[0]
    bright_cols = np.where(bright_mask.any(axis=0))[0]

    if bright_rows.size == 0 or bright_cols.size == 0:
        return False, "too_dark", {"dark_ratio": dark_ratio}

    row_span = int(bright_rows[-1] - bright_rows[0]) or 1
    col_span = int(bright_cols[-1] - bright_cols[0]) or 1

    # 밝은 영역의 종횡비 (원이면 ≈ 1.0)
    aspect_ratio = float(min(row_span, col_span) / max(row_span, col_span))

    # Fill ratio: 밝은 픽셀 수 / 바운딩박스 면적
    #   원이 정사각형 내에 꽉 찬 경우 ≈ π/4 ≈ 0.785
    #   프레임 전체가 밝은 경우       ≈ 1.0
    bright_area = int(bright_mask.sum())
    fill_ratio  = float(bright_area / (row_span * col_span))

    # 밝은 영역의 채널별 평균
    r_mean = float(arr[:, :, 0][bright_mask].mean()) if bright_area > 0 else 0.0
    g_mean = float(arr[:, :, 1][bright_mask].mean()) if bright_area > 0 else 0.0
    b_mean = float(arr[:, :, 2][bright_mask].mean()) if bright_area > 0 else 0.0

    # 안저 이미지는 R > G > B (붉은-주황 계통)
    is_reddish = r_mean > g_mean and g_mean > b_mean

    details = {
        "dark_ratio":   dark_ratio,
        "aspect_ratio": aspect_ratio,
        "fill_ratio":   fill_ratio,
        "r_mean":       r_mean,
        "g_mean":       g_mean,
        "b_mean":       b_mean,
        "is_reddish":   is_reddish,
    }

    # ---------- 판정 ----------
    # 각 조건이 실패하는 즉시 거부한다.

    # 1. 어두운 테두리가 거의 없음 → 안저 렌즈 특성 없음
    if dark_ratio < _MIN_DARK_RATIO:
        return False, "no_dark_border", details

    # 2. 밝은 영역이 지나치게 가로 또는 세로로 치우침 → 원형이 아님
    if aspect_ratio < _MIN_ASPECT_RATIO:
        return False, "non_circular_shape", details

    # 3. 밝은 영역이 프레임을 거의 가득 채움 → 원형 마스크(테두리) 없음
    if fill_ratio > _MAX_FILL_RATIO:
        return False, "fills_frame", details

    # 4. 색상이 붉은-주황 계통이 아님 → 안저 이미지 색조와 다름
    if not is_reddish:
        return False, "wrong_color_distribution", details

    return True, "", details
