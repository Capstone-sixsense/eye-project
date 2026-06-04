import cv2
import numpy as np
from PIL import Image

from drscreen.data.transforms import FundusPreprocess, is_preprocessed_image_path


def _fundus_like_pil(width: int = 120, height: int = 80) -> Image.Image:
    arr = np.zeros((height, width, 3), dtype=np.uint8)
    cv2.ellipse(arr, (width // 2, height // 2), (50, 30), 0, 0, 360, (40, 80, 120), -1)
    return Image.fromarray(arr)


def test_apply_ben_graham_false_none_mode_is_identity() -> None:
    img = _fundus_like_pil()
    out = FundusPreprocess(preprocess_mode="none", output_size=None, apply_ben_graham=False)(img)
    assert np.array_equal(np.asarray(out), np.asarray(img))


def test_apply_ben_graham_flag_gates_photometry() -> None:
    img = _fundus_like_pil()
    with_bg = np.asarray(FundusPreprocess(preprocess_mode="none", apply_ben_graham=True)(img))
    without_bg = np.asarray(FundusPreprocess(preprocess_mode="none", apply_ben_graham=False)(img))
    # Ben Graham changes pixels; disabling it leaves a none-mode image untouched.
    assert not np.array_equal(with_bg, without_bg)
    assert np.array_equal(without_bg, np.asarray(img))


def test_apply_ben_graham_false_quickqual_changes_geometry_only() -> None:
    img = _fundus_like_pil(width=200, height=120)
    with_bg = np.asarray(FundusPreprocess(preprocess_mode="quickqual", apply_ben_graham=True)(img))
    without_bg = np.asarray(FundusPreprocess(preprocess_mode="quickqual", apply_ben_graham=False)(img))
    # Geometry (square pad) is applied in both -> the flag controls photometry only.
    assert with_bg.shape[0] == with_bg.shape[1]
    assert without_bg.shape == with_bg.shape
    assert not np.array_equal(with_bg, without_bg)


def test_is_preprocessed_image_path_guard() -> None:
    # Already-preprocessed inputs (serve config would double-apply Ben Graham).
    assert is_preprocessed_image_path("data/raw/processed_quickqual/images/ddr/x.png")
    assert is_preprocessed_image_path("processed/images/idrid/IDRiD_001.jpg")
    assert is_preprocessed_image_path("processed_safezoom/images/y.png")
    assert is_preprocessed_image_path(r"C:\Users\x\processed_contentcrop\images\z.png")
    # Raw inputs must not trip the guard.
    assert not is_preprocessed_image_path("data/raw/ddr/DR_grading/007-0004-000.jpg")
    assert not is_preprocessed_image_path("uploads/scan.png")
