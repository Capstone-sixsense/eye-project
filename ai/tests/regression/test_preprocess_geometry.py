import cv2
import numpy as np
import yaml

from drscreen.data.mask_providers import _preprocess_options_for_image_path
from drscreen.data.transforms import FundusPreprocess, preprocess_kwargs_from_config


def _fundus_like_image(width: int = 120, height: int = 80) -> np.ndarray:
    image = np.zeros((height, width, 3), dtype=np.uint8)
    cv2.ellipse(image, (width // 2, height // 2), (50, 30), 0, 0, 360, (40, 80, 120), -1)
    return image


def _old_circular_geometry(image: np.ndarray) -> tuple[int, int, int, int, int, int, int, int] | None:
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    _, mask = cv2.threshold(gray, 7, 255, cv2.THRESH_BINARY)
    coords = cv2.findNonZero(mask)
    if coords is None:
        return None

    moments = cv2.moments(mask)
    if moments["m00"] > 0:
        cx = int(moments["m10"] / moments["m00"])
        cy = int(moments["m01"] / moments["m00"])
    else:
        x, y, w, h = cv2.boundingRect(coords)
        cx, cy = x + w // 2, y + h // 2

    _x, _y, w, h = cv2.boundingRect(coords)
    radius = min(w, h) // 2
    h_img, w_img = image.shape[:2]
    cx = int(np.clip(cx, radius, w_img - radius))
    cy = int(np.clip(cy, radius, h_img - radius))
    x1 = cx - radius
    y1 = cy - radius
    x2 = cx + radius
    y2 = cy + radius
    return x1, y1, x2, y2, 0, 0, 0, 0


def test_legacy_processed_prefix_uses_old_circular_geometry() -> None:
    mode = _preprocess_options_for_image_path("processed/images/idrid/IDRiD_001.jpg")[0]

    image = _fundus_like_image()
    geometry = FundusPreprocess(preprocess_mode=mode)._preprocess_geometry(image)

    assert mode == "circular"
    assert geometry == _old_circular_geometry(image)


def test_new_processed_prefixes_keep_new_geometry_modes() -> None:
    assert _preprocess_options_for_image_path("processed_contentcrop/images/x.png")[0] == "contentcrop"
    assert _preprocess_options_for_image_path("processed_safezoom/images/x.png")[0] == "safezoom"
    assert _preprocess_options_for_image_path("processed_quickqual/images/x.png")[0] == "quickqual"


def test_quickqual_mode_crops_to_square_with_buffer() -> None:
    image = _fundus_like_image(width=200, height=120)
    geometry = FundusPreprocess(preprocess_mode="quickqual")._preprocess_geometry(image)
    assert geometry is not None
    x1, y1, x2, y2, pad_top, pad_bottom, pad_left, pad_right = geometry
    cropped_w, cropped_h = x2 - x1, y2 - y1
    side = max(cropped_w, cropped_h)
    assert pad_top + cropped_h + pad_bottom == side
    assert pad_left + cropped_w + pad_right == side


def test_none_mode_skips_geometry() -> None:
    image = _fundus_like_image()
    assert FundusPreprocess(preprocess_mode="none")._preprocess_geometry(image) is None


def test_circular_mode_uses_circular_ben_graham_mask() -> None:
    image = np.full((32, 32, 3), 100, dtype=np.uint8)
    processed = FundusPreprocess(preprocess_mode="circular")._ben_graham(image)

    assert processed[0, 0].tolist() == [0, 0, 0]
    assert processed[16, 16].sum() > 0


def test_active_deployment_config_requests_quickqual_serve_preprocessing() -> None:
    # Active deployment is v31_v8b_fusion_quickqual_v1: offline training uses quickqual
    # geometry; at serve time backend QuickQual already applied the square crop, so AI
    # inference skips geometry (infer.preprocess_mode: none). v31_v8b_fusion_v2 remains
    # the circular rollback baseline.
    with open("configs/base.yaml", encoding="utf-8") as handle:
        base_config = yaml.safe_load(handle)
    with open("configs/v31_v8b_fusion_v2.yaml", encoding="utf-8") as handle:
        rollback_config = yaml.safe_load(handle)

    assert base_config["data"]["preprocess_mode"] == "quickqual"
    assert base_config["infer"]["preprocess_mode"] == "none"
    # Offline preprocessing reads data-only kwargs -> quickqual geometry.
    assert preprocess_kwargs_from_config(base_config["data"])["preprocess_mode"] == "quickqual"
    # Live serve merges data+infer (infer wins) -> passthrough, since backend already cropped.
    assert preprocess_kwargs_from_config(base_config["data"], base_config["infer"])["preprocess_mode"] == "none"
    assert rollback_config["infer"]["preprocess_mode"] == "circular"


def test_v8b_eval_configs_pin_expected_preprocess_geometry() -> None:
    expected = {
        "configs/seg_evidence_v8b_ddrseg_tjdr_maplesfix.yaml": "circular",
        "configs/seg_evidence_v8b_repro_seed43_geometryfix.yaml": "circular",
        "configs/seg_evidence_v8b_contentcrop_v1.yaml": "contentcrop",
        "configs/seg_evidence_v8b_safezoom_v1.yaml": "safezoom",
    }
    for path, mode in expected.items():
        with open(path, encoding="utf-8") as handle:
            config = yaml.safe_load(handle)
        assert config["data"]["preprocess_mode"] == mode
