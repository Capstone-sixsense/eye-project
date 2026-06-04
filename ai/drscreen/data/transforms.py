"""안저 이미지 전처리 + 학습/평가용 데이터 증강(transform) 정의.

크게 세 부분:
1. FundusPreprocess: 카메라 여백 제거(content crop/square-pad) + Ben Graham 조명
   정규화. preprocess_mode로 geometry를 분기한다
   (contentcrop / safezoom / circular / quickqual / none).
2. 마스크 geometry 동기화: apply_mask_geometry()가 이미지에 적용한 crop/pad/resize와
   '동일한 기하 변환'만 마스크에 적용한다(Ben Graham 같은 광학적 변환은 마스크에 미적용).
3. transform 빌더: build_train/eval_transform(분류), build_segmentation_* (분할).
   분할용은 이미지와 마스크에 같은 공간 증강을 동기 적용한다(_SegmentationTransform).

geometry 규약: 내부 _*_geometry 메서드는 모두 8-튜플
  (x1, y1, x2, y2, pad_top, pad_bottom, pad_left, pad_right)
을 반환한다. 즉 '[y1:y2, x1:x2]로 자른 뒤 상/하/좌/우로 패딩해 정사각형으로'를 뜻한다.
이미지와 마스크가 같은 8-튜플을 공유하므로 픽셀 정합이 보장된다.
이 규약은 tests/regression/test_preprocess_geometry.py가 고정 검증한다.
"""

from __future__ import annotations

import albumentations as A
import cv2
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from PIL import Image as PILImage
from torchvision import transforms

from drscreen.models.profiles import resolve_interpolation_mode


class FundusPreprocess:
    """Fundus-specific adaptive preprocessing pipeline.

    Two-stage pipeline:
    1. Content crop -- removes only low-information border padding introduced
       by fundus cameras. The crop is based on the foreground bounding box and
       then square-padded, so non-circular or partially clipped fundus images
       are not forced into a centered circular disk.
    2. Ben Graham normalization -- subtracts the local mean illumination
       (Gaussian-blurred version of the image) to remove uneven lighting.
       A foreground mask fills non-fundus background with the per-channel
       fundus mean before blurring, preventing border bleed while preserving
       the actual visible field geometry. sigmaX scales with the image's
       longest dimension so the operation is resolution-adaptive.

    Reference: Graham B., "Kaggle Diabetic Retinopathy Detection", 2015
               (1st place, ~0.84 QWK).
    """

    def __init__(
        self,
        crop_tol: int = 7,
        ben_graham_weight: float = 4.0,
        ben_graham_offset: float = 128.0,
        output_size: int | None = None,
        align: bool = False,
        align_decentering_limit: float = 0.35,
        preprocess_mode: str = "contentcrop",
        target_short_fill: float = 0.86,
        max_total_x_trim: float = 0.08,
        max_total_y_trim: float = 0.08,
        saliency_shift: bool = True,
        saliency_weight: float = 1.2,
        saliency_candidates: int = 5,
        safezoom_max_dim: int = 1024,
        apply_ben_graham: bool = True,
    ) -> None:
        mode = str(preprocess_mode or "contentcrop").strip().lower()
        if mode not in {"contentcrop", "safezoom", "circular", "quickqual", "none"}:
            raise ValueError(f"Unsupported preprocess_mode: {preprocess_mode!r}")
        self._crop_tol = crop_tol
        self._weight = ben_graham_weight
        self._offset = ben_graham_offset
        self._output_size = output_size
        self._align = align
        self._decentering_limit = align_decentering_limit
        self._preprocess_mode = mode
        self._target_short_fill = float(target_short_fill)
        self._max_total_x_trim = float(max_total_x_trim)
        self._max_total_y_trim = float(max_total_y_trim)
        self._saliency_shift = bool(saliency_shift)
        self._saliency_weight = float(saliency_weight)
        self._saliency_candidates = max(3, int(saliency_candidates))
        self._safezoom_max_dim = max(0, int(safezoom_max_dim))
        self._apply_ben_graham = bool(apply_ben_graham)

    def __call__(self, img: PILImage.Image) -> PILImage.Image:
        # 처리 순서: (선택)정렬 -> content crop+pad -> (선택)정사각 resize -> (선택)Ben Graham.
        # geometry(crop/pad/resize)를 먼저 끝낸 뒤 마지막에 광학적 정규화를 적용한다.
        arr = np.asarray(img.convert("RGB")).copy()
        if self._align:
            arr = self._correct_alignment(arr)
        arr = self._content_crop(arr)
        result = PILImage.fromarray(arr)
        if self._output_size is not None:
            result = result.resize((self._output_size, self._output_size), PILImage.BICUBIC)
            arr = np.asarray(result).copy()
        if self._apply_ben_graham:
            arr = self._ben_graham(arr)
        result = PILImage.fromarray(arr)
        return result

    def apply_mask_geometry(
        self,
        mask: np.ndarray,
        reference_img: PILImage.Image,
        *,
        output_size: int | None = None,
    ) -> np.ndarray:
        """Apply the image preprocessing geometry to a mask.

        This mirrors the geometric parts of ``__call__``: optional alignment,
        content crop, padding, and final resize. Ben Graham normalization is
        photometric only and is intentionally skipped for masks.
        """
        ref = np.asarray(reference_img.convert("RGB")).copy()
        mask_arr = np.asarray(mask).copy()
        was_singleton_channel = mask_arr.ndim == 3 and mask_arr.shape[-1] == 1
        if mask_arr.shape[:2] != ref.shape[:2]:
            mask_arr = cv2.resize(
                mask_arr,
                (ref.shape[1], ref.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )
            if was_singleton_channel and mask_arr.ndim == 2:
                mask_arr = mask_arr[..., None]

        if self._align:
            matrix = self._alignment_matrix(ref)
            if matrix is not None:
                h, w = ref.shape[:2]
                ref = cv2.warpAffine(
                    ref,
                    matrix,
                    (w, h),
                    flags=cv2.INTER_LANCZOS4,
                    borderMode=cv2.BORDER_CONSTANT,
                    borderValue=(0, 0, 0),
                )
                mask_arr = cv2.warpAffine(
                    mask_arr,
                    matrix,
                    (w, h),
                    flags=cv2.INTER_NEAREST,
                    borderMode=cv2.BORDER_CONSTANT,
                    borderValue=0,
                )
                if was_singleton_channel and mask_arr.ndim == 2:
                    mask_arr = mask_arr[..., None]

        geometry = self._preprocess_geometry(ref)
        if geometry is not None:
            x1, y1, x2, y2, pad_top, pad_bottom, pad_left, pad_right = geometry
            mask_arr = mask_arr[y1:y2, x1:x2]
            mask_arr = cv2.copyMakeBorder(
                mask_arr,
                pad_top,
                pad_bottom,
                pad_left,
                pad_right,
                cv2.BORDER_CONSTANT,
                value=0,
            )
            if was_singleton_channel and mask_arr.ndim == 2:
                mask_arr = mask_arr[..., None]

        target_size = output_size if output_size is not None else self._output_size
        if target_size is not None and mask_arr.shape[:2] != (target_size, target_size):
            mask_arr = cv2.resize(
                mask_arr,
                (target_size, target_size),
                interpolation=cv2.INTER_NEAREST,
            )
            if was_singleton_channel and mask_arr.ndim == 2:
                mask_arr = mask_arr[..., None]
        return mask_arr

    def _alignment_matrix(self, image: np.ndarray) -> np.ndarray | None:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        _, mask = cv2.threshold(gray, self._crop_tol, 255, cv2.THRESH_BINARY)

        ksize = max(5, min(image.shape[:2]) // 30) | 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        largest = max(contours, key=cv2.contourArea)
        h, w = image.shape[:2]
        img_cx, img_cy = w / 2.0, h / 2.0

        moments = cv2.moments(largest)
        if moments["m00"] == 0:
            return None
        disk_cx = moments["m10"] / moments["m00"]
        disk_cy = moments["m01"] / moments["m00"]

        if np.hypot(disk_cx - img_cx, disk_cy - img_cy) > min(h, w) * self._decentering_limit:
            return None

        dx, dy = img_cx - disk_cx, img_cy - disk_cy
        if abs(dx) <= 3 and abs(dy) <= 3:
            return None
        return np.float32([[1, 0, dx], [0, 1, dy]])

    def _correct_alignment(self, image: np.ndarray) -> np.ndarray:
        """Translate the fundus disk centroid to the image center.

        If the disk centroid is more than ``align_decentering_limit`` of the
        shorter image dimension from the frame center, alignment is skipped —
        the image is too severely decentered for reliable geometric correction,
        so preprocessing falls back to content crop + Ben Graham only.
        """
        matrix = self._alignment_matrix(image)
        if matrix is None:
            return image
        h, w = image.shape[:2]
        return cv2.warpAffine(
            image,
            matrix,
            (w, h),
            flags=cv2.INTER_LANCZOS4,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0),
        )

    def _foreground_mask(self, image: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        _, mask = cv2.threshold(gray, self._crop_tol, 255, cv2.THRESH_BINARY)

        ksize = max(3, min(image.shape[:2]) // 80) | 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
        if num_labels <= 1:
            return mask > 0
        largest_label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        return labels == largest_label

    def _get_circle_mask(self, h: int, w: int) -> np.ndarray:
        cy, cx = h / 2.0, w / 2.0
        radius = min(h, w) / 2.0
        Y, X = np.ogrid[:h, :w]
        return ((X - cx) ** 2 + (Y - cy) ** 2) <= radius ** 2

    def _circular_crop_geometry(
        self,
        image: np.ndarray,
    ) -> tuple[int, int, int, int, int, int, int, int] | None:
        # Compatibility alias: mask providers call this private method to keep
        # raw masks aligned with image preprocessing. Dispatch by
        # preprocess_mode instead of imposing circular geometry.
        return self._preprocess_geometry(image)

    def _preprocess_geometry(
        self,
        image: np.ndarray,
    ) -> tuple[int, int, int, int, int, int, int, int] | None:
        # preprocess_mode에 따라 crop/pad 8-튜플을 계산한다. 이 단일 진입점을 이미지
        # 크롭(_content_crop)과 마스크 정합(apply_mask_geometry)이 공유하므로, 어떤 mode든
        # 이미지와 마스크에 같은 geometry가 적용된다. none은 전처리 없음(8-튜플 대신 None).
        if self._preprocess_mode == "circular":
            return self._legacy_circular_crop_geometry(image)
        if self._preprocess_mode == "safezoom":
            return self._safezoom_geometry(image)
        if self._preprocess_mode == "quickqual":
            return self._quickqual_geometry(image)
        if self._preprocess_mode == "none":
            return None
        return self._content_crop_geometry(image)

    def _quickqual_geometry(
        self,
        image: np.ndarray,
    ) -> tuple[int, int, int, int, int, int, int, int] | None:
        # Replicates backend QuickQual geometry (backend/models/quickqual_wrapper.py:24-58):
        # RGB-mean > 15 bbox, +20px buffer, square pad. Final resize handled by __call__.
        mean = image.mean(axis=-1)
        ys = np.where(mean > 15)
        if ys[0].size == 0 or ys[1].size == 0:
            return None
        top, bottom = int(ys[0].min()), int(ys[0].max())
        left, right = int(ys[1].min()), int(ys[1].max())
        h_img, w_img = image.shape[:2]
        left = max(0, left - 20)
        right = min(w_img, right + 20)
        top = max(0, top - 20)
        bottom = min(h_img, bottom + 20)
        cropped_w = right - left
        cropped_h = bottom - top
        side = max(cropped_w, cropped_h)
        pad_top = (side - cropped_h) // 2
        pad_bottom = side - cropped_h - pad_top
        pad_left = (side - cropped_w) // 2
        pad_right = side - cropped_w - pad_left
        return left, top, right, bottom, pad_top, pad_bottom, pad_left, pad_right

    def _legacy_circular_crop_geometry(
        self,
        image: np.ndarray,
    ) -> tuple[int, int, int, int, int, int, int, int] | None:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        _, mask = cv2.threshold(gray, self._crop_tol, 255, cv2.THRESH_BINARY)
        coords = cv2.findNonZero(mask)
        if coords is None:
            return None

        M = cv2.moments(mask)
        if M["m00"] > 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
        else:
            x, y, w, h = cv2.boundingRect(coords)
            cx, cy = x + w // 2, y + h // 2

        x, y, w, h = cv2.boundingRect(coords)
        radius = min(w, h) // 2

        h_img, w_img = image.shape[:2]
        cx = int(np.clip(cx, radius, w_img - radius))
        cy = int(np.clip(cy, radius, h_img - radius))
        x1 = cx - radius
        y1 = cy - radius
        x2 = cx + radius
        y2 = cy + radius
        cropped = image[y1:y2, x1:x2]

        ch, cw = cropped.shape[:2]
        side = max(ch, cw)
        pad_top = (side - ch) // 2
        pad_bottom = side - ch - pad_top
        pad_left = (side - cw) // 2
        pad_right = side - cw - pad_left
        return x1, y1, x2, y2, pad_top, pad_bottom, pad_left, pad_right

    def _content_crop_geometry(
        self,
        image: np.ndarray,
    ) -> tuple[int, int, int, int, int, int, int, int] | None:
        mask = self._foreground_mask(image).astype(np.uint8) * 255
        coords = cv2.findNonZero(mask)
        if coords is None:
            return None

        x, y, w, h = cv2.boundingRect(coords)
        h_img, w_img = image.shape[:2]
        buffer = max(4, int(round(max(w, h) * 0.01)))
        x1 = max(0, x - buffer)
        y1 = max(0, y - buffer)
        x2 = min(w_img, x + w + buffer)
        y2 = min(h_img, y + h + buffer)
        cropped = image[y1:y2, x1:x2]

        ch, cw = cropped.shape[:2]
        side = max(ch, cw)
        pad_top = (side - ch) // 2
        pad_bottom = side - ch - pad_top
        pad_left = (side - cw) // 2
        pad_right = side - cw - pad_left
        return x1, y1, x2, y2, pad_top, pad_bottom, pad_left, pad_right

    def _safezoom_geometry(
        self,
        image: np.ndarray,
    ) -> tuple[int, int, int, int, int, int, int, int] | None:
        h_img, w_img = image.shape[:2]
        scale = 1.0
        work = image
        if self._safezoom_max_dim and max(h_img, w_img) > self._safezoom_max_dim:
            scale = self._safezoom_max_dim / float(max(h_img, w_img))
            new_w = max(1, int(round(w_img * scale)))
            new_h = max(1, int(round(h_img * scale)))
            work = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

        geometry = self._safezoom_geometry_on_work_image(work)
        if geometry is None:
            return geometry
        if scale == 1.0:
            if self._foreground_loss_for_geometry(self._foreground_mask(image), geometry) > max(
                self._max_total_x_trim,
                self._max_total_y_trim,
            ):
                return self._content_crop_geometry(image)
            return geometry

        inv = 1.0 / scale
        x1, y1, x2, y2, pad_top, pad_bottom, pad_left, pad_right = geometry
        scaled_geometry = (
            max(0, min(w_img, int(np.floor(x1 * inv)))),
            max(0, min(h_img, int(np.floor(y1 * inv)))),
            max(0, min(w_img, int(np.ceil(x2 * inv)))),
            max(0, min(h_img, int(np.ceil(y2 * inv)))),
            max(0, int(round(pad_top * inv))),
            max(0, int(round(pad_bottom * inv))),
            max(0, int(round(pad_left * inv))),
            max(0, int(round(pad_right * inv))),
        )
        if self._foreground_loss_for_geometry(self._foreground_mask(image), scaled_geometry) > max(
            self._max_total_x_trim,
            self._max_total_y_trim,
        ):
            return self._content_crop_geometry(image)
        return scaled_geometry

    def _safezoom_geometry_on_work_image(
        self,
        image: np.ndarray,
    ) -> tuple[int, int, int, int, int, int, int, int] | None:
        mask = self._foreground_mask(image)
        bounds = self._robust_foreground_bounds(mask)
        if bounds is None:
            return None

        h_img, w_img = image.shape[:2]
        x1, y1, x2, y2 = bounds
        bbox_w = max(1, x2 - x1)
        bbox_h = max(1, y2 - y1)
        bbox_short = min(bbox_w, bbox_h)
        bbox_long = max(bbox_w, bbox_h)
        fill = float(np.clip(self._target_short_fill, 0.5, 0.98))
        max_trim = self._max_total_x_trim if bbox_w >= bbox_h else self._max_total_y_trim
        max_trim = float(np.clip(max_trim, 0.0, 0.25))

        desired_side = bbox_short / fill
        capped_side = bbox_long * (1.0 - max_trim)
        side = int(np.ceil(max(bbox_short, min(bbox_long, max(desired_side, capped_side)))))
        side = max(1, side)
        min_side = max(1, int(np.ceil(max(bbox_short, desired_side))))
        if min_side < side:
            candidate_side_set = {int(v) for v in np.linspace(min_side, side, 7)}
            candidate_side_set.update(range(max(min_side, side - 6), side + 1))
            candidate_side_set.add(int(np.ceil(bbox_long)))
            candidate_sides = sorted(candidate_side_set)
        else:
            candidate_sides = sorted({side, int(np.ceil(bbox_long))})

        bbox_cx = (x1 + x2) / 2.0
        bbox_cy = (y1 + y2) / 2.0
        saliency = self._safezoom_saliency_map(image, mask) if self._saliency_shift else None
        od_box = self._optic_disc_proxy_box(image, mask) if self._saliency_shift else None
        best_geometry = None
        best_score = float("inf")
        max_foreground_loss = max_trim
        for candidate_side in candidate_sides:
            centers = self._safezoom_candidate_centers(
                bounds=bounds,
                side=candidate_side,
                image_shape=(h_img, w_img),
                vary_x=bbox_w >= bbox_h,
            )
            if not centers:
                centers = [(bbox_cx, bbox_cy)]
            zoom_gap = max(0.0, (candidate_side / max(1.0, desired_side)) - 1.0)
            for cx, cy in centers:
                geometry = self._square_geometry_from_center(
                    cx,
                    cy,
                    candidate_side,
                    width=w_img,
                    height=h_img,
                )
                score = self._safezoom_candidate_score(
                    mask,
                    saliency,
                    geometry,
                    bbox_center=(bbox_cx, bbox_cy),
                    side=candidate_side,
                    od_box=od_box,
                    max_foreground_loss=max_foreground_loss,
                )
                score += 2.0 * zoom_gap
                if score < best_score:
                    best_score = score
                    best_geometry = geometry
        if (
            best_geometry is not None
            and self._foreground_loss_for_geometry(mask, best_geometry) > max_foreground_loss
        ):
            return self._content_crop_geometry(image)
        return best_geometry

    @staticmethod
    def _foreground_loss_for_geometry(
        mask: np.ndarray,
        geometry: tuple[int, int, int, int, int, int, int, int],
    ) -> float:
        x1, y1, x2, y2, *_ = geometry
        inside_rect = np.zeros(mask.shape, dtype=bool)
        inside_rect[y1:y2, x1:x2] = True
        foreground_total = float(mask.sum())
        if foreground_total <= 0.0:
            return 0.0
        return float((mask & ~inside_rect).sum()) / foreground_total

    def _robust_foreground_bounds(self, mask: np.ndarray) -> tuple[int, int, int, int] | None:
        ys, xs = np.where(mask)
        if xs.size == 0:
            return None
        raw_x1, raw_x2 = int(xs.min()), int(xs.max()) + 1
        raw_y1, raw_y2 = int(ys.min()), int(ys.max()) + 1
        if xs.size >= 100:
            x1 = int(np.floor(np.percentile(xs, 1.0)))
            x2 = int(np.ceil(np.percentile(xs, 99.0))) + 1
            y1 = int(np.floor(np.percentile(ys, 1.0)))
            y2 = int(np.ceil(np.percentile(ys, 99.0))) + 1
            raw_w = max(1, raw_x2 - raw_x1)
            raw_h = max(1, raw_y2 - raw_y1)
            if (x2 - x1) < raw_w * 0.85 or (y2 - y1) < raw_h * 0.85:
                x1, x2, y1, y2 = raw_x1, raw_x2, raw_y1, raw_y2
        else:
            x1, x2, y1, y2 = raw_x1, raw_x2, raw_y1, raw_y2

        h_img, w_img = mask.shape[:2]
        buffer = max(4, int(round(max(x2 - x1, y2 - y1) * 0.01)))
        return (
            max(0, x1 - buffer),
            max(0, y1 - buffer),
            min(w_img, x2 + buffer),
            min(h_img, y2 + buffer),
        )

    def _safezoom_candidate_centers(
        self,
        *,
        bounds: tuple[int, int, int, int],
        side: int,
        image_shape: tuple[int, int],
        vary_x: bool,
    ) -> list[tuple[float, float]]:
        h_img, w_img = image_shape
        x1, y1, x2, y2 = bounds
        bbox_cx = (x1 + x2) / 2.0
        bbox_cy = (y1 + y2) / 2.0

        def _clamp_center(value: float, image_len: int) -> float:
            if side >= image_len:
                return image_len / 2.0
            half = side / 2.0
            return float(np.clip(value, half, image_len - half))

        if vary_x:
            left_center = x1 + side / 2.0
            right_center = x2 - side / 2.0
            if right_center <= left_center:
                xs = [bbox_cx]
            else:
                xs = np.linspace(left_center, right_center, self._saliency_candidates).tolist()
            return [(_clamp_center(cx, w_img), _clamp_center(bbox_cy, h_img)) for cx in xs]

        top_center = y1 + side / 2.0
        bottom_center = y2 - side / 2.0
        if bottom_center <= top_center:
            ys = [bbox_cy]
        else:
            ys = np.linspace(top_center, bottom_center, self._saliency_candidates).tolist()
        return [(_clamp_center(bbox_cx, w_img), _clamp_center(cy, h_img)) for cy in ys]

    @staticmethod
    def _square_geometry_from_center(
        cx: float,
        cy: float,
        side: int,
        *,
        width: int,
        height: int,
    ) -> tuple[int, int, int, int, int, int, int, int]:
        x1_full = int(round(cx - side / 2.0))
        y1_full = int(round(cy - side / 2.0))
        x2_full = x1_full + side
        y2_full = y1_full + side

        pad_left = max(0, -x1_full)
        pad_top = max(0, -y1_full)
        pad_right = max(0, x2_full - width)
        pad_bottom = max(0, y2_full - height)

        x1 = max(0, x1_full)
        y1 = max(0, y1_full)
        x2 = min(width, x2_full)
        y2 = min(height, y2_full)
        return x1, y1, x2, y2, pad_top, pad_bottom, pad_left, pad_right

    def _safezoom_saliency_map(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32)
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        sat = hsv[:, :, 1].astype(np.float32)
        val = hsv[:, :, 2].astype(np.float32)
        inside = mask.astype(bool)
        if not inside.any():
            return np.zeros(gray.shape, dtype=np.float32)

        sobel_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        grad = cv2.magnitude(sobel_x, sobel_y)
        grad = self._normalize_inside_mask(grad, inside, high_percentile=95.0)

        sat_high = float(np.percentile(sat[inside], 95.0))
        sat_peak = np.clip((sat - sat_high) / max(1.0, 255.0 - sat_high), 0.0, 1.0)

        dark_low = float(np.percentile(val[inside], 12.0))
        dark = np.clip((dark_low - val) / max(1.0, dark_low), 0.0, 1.0)
        dark_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        dark = cv2.morphologyEx(dark.astype(np.float32), cv2.MORPH_OPEN, dark_kernel)

        ksize = max(9, (min(image.shape[:2]) // 32) | 1)
        bright_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
        bright = cv2.morphologyEx(val.astype(np.uint8), cv2.MORPH_TOPHAT, bright_kernel).astype(np.float32)
        bright = self._normalize_inside_mask(bright, inside, high_percentile=98.0)

        saliency = 0.35 * grad + 0.20 * sat_peak + 0.25 * dark + 0.20 * bright
        saliency[~inside] = 0.0
        return np.clip(saliency, 0.0, 1.0).astype(np.float32)

    @staticmethod
    def _normalize_inside_mask(
        values: np.ndarray,
        mask: np.ndarray,
        *,
        high_percentile: float,
    ) -> np.ndarray:
        inside_values = values[mask]
        if inside_values.size == 0:
            return np.zeros(values.shape, dtype=np.float32)
        high = float(np.percentile(inside_values, high_percentile))
        if high <= 1e-6:
            return np.zeros(values.shape, dtype=np.float32)
        return np.clip(values / high, 0.0, 1.0).astype(np.float32)

    def _optic_disc_proxy_box(
        self,
        image: np.ndarray,
        mask: np.ndarray,
    ) -> tuple[int, int, int, int] | None:
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        sat = hsv[:, :, 1].astype(np.float32)
        val = hsv[:, :, 2].astype(np.float32)
        inside = mask.astype(bool)
        if not inside.any():
            return None
        v_thr = float(np.percentile(val[inside], 92.0))
        s_thr = float(np.percentile(sat[inside], 65.0))
        candidate = (inside & (val >= v_thr) & (sat <= s_thr)).astype(np.uint8) * 255
        ksize = max(7, (min(image.shape[:2]) // 40) | 1)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
        candidate = cv2.morphologyEx(candidate, cv2.MORPH_CLOSE, kernel)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(candidate)
        if num_labels <= 1:
            return None
        areas = stats[1:, cv2.CC_STAT_AREA]
        label = 1 + int(np.argmax(areas))
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < max(20, int(mask.sum() * 0.002)):
            return None
        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        w = int(stats[label, cv2.CC_STAT_WIDTH])
        h = int(stats[label, cv2.CC_STAT_HEIGHT])
        return x, y, x + w, y + h

    def _safezoom_candidate_score(
        self,
        mask: np.ndarray,
        saliency: np.ndarray | None,
        geometry: tuple[int, int, int, int, int, int, int, int],
        *,
        bbox_center: tuple[float, float],
        side: int,
        od_box: tuple[int, int, int, int] | None,
        max_foreground_loss: float,
    ) -> float:
        x1, y1, x2, y2, *_ = geometry
        foreground_loss = self._foreground_loss_for_geometry(mask, geometry)
        inside_rect = np.zeros(mask.shape, dtype=bool)
        inside_rect[y1:y2, x1:x2] = True
        outside = mask & ~inside_rect

        if saliency is not None:
            saliency_total = float(saliency[mask].sum())
            saliency_loss = float(saliency[outside].sum()) / max(1e-6, saliency_total)
        else:
            saliency_loss = foreground_loss

        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        center_shift = (
            abs(cx - bbox_center[0]) + abs(cy - bbox_center[1])
        ) / max(1.0, float(side))

        od_penalty = 0.0
        if od_box is not None:
            ox1, oy1, ox2, oy2 = od_box
            od_area = max(1.0, float((ox2 - ox1) * (oy2 - oy1)))
            ix1, iy1 = max(x1, ox1), max(y1, oy1)
            ix2, iy2 = min(x2, ox2), min(y2, oy2)
            if ix2 <= ix1 or iy2 <= iy1:
                od_penalty = 1.0
            else:
                inside_area = float((ix2 - ix1) * (iy2 - iy1))
                od_penalty = 1.0 - inside_area / od_area

        hard_penalty = 0.0
        if foreground_loss > max_foreground_loss:
            hard_penalty = 10.0 + (foreground_loss - max_foreground_loss) * 100.0

        return (
            foreground_loss
            + self._saliency_weight * saliency_loss
            + 0.7 * od_penalty
            + 0.2 * center_shift
            + hard_penalty
        )

    def _content_crop(self, image: np.ndarray) -> np.ndarray:
        geometry = self._preprocess_geometry(image)
        if geometry is None:
            return image
        x1, y1, x2, y2, pad_top, pad_bottom, pad_left, pad_right = geometry
        cropped = image[y1:y2, x1:x2]
        return cv2.copyMakeBorder(
            cropped, pad_top, pad_bottom, pad_left, pad_right,
            cv2.BORDER_CONSTANT, value=(0, 0, 0),
        )

    def _ben_graham(self, image: np.ndarray) -> np.ndarray:
        h, w = image.shape[:2]
        mask = (
            self._get_circle_mask(h, w)
            if self._preprocess_mode == "circular"
            else self._foreground_mask(image)
        )

        # Fill non-fundus pixels with per-channel mean of the fundus region
        # before blurring. Without this, the black background (value=0)
        # bleeds inward via Gaussian blur, creating a dark halo at the
        # retinal boundary that suppresses peripheral lesion signals.
        work = image.copy()
        for c in range(3):
            channel = image[:, :, c]
            fill_val = int(channel[mask].mean()) if mask.any() else 128
            work[:, :, c][~mask] = fill_val

        sigma_x = max(h, w) / 30.0
        blurred = cv2.GaussianBlur(work, (0, 0), sigma_x)
        result = cv2.addWeighted(work, self._weight, blurred, -self._weight, self._offset)
        result = np.clip(result, 0, 255).astype(np.uint8)

        # Restore the geometry expected by the selected preprocessing mode.
        result[~mask] = 0
        return result


def fda_mix(source: np.ndarray, reference: np.ndarray, alpha: float) -> np.ndarray:
    """Exchange low-frequency Fourier amplitude from reference into source.

    Transfers the global color/illumination style of ``reference`` into
    ``source`` while preserving high-frequency content (lesion edges, vessel
    structures). Operates channel-wise in the spatial frequency domain.

    The swap region is a centered square of half-side b = floor(alpha * min(H, W)).
    With alpha=0.05 and 512px images, b=25 pixels -- enough to capture global
    illumination without touching structural detail.

    Reference: Yang & Soatto, "FDA: Fourier Domain Adaptation for Semantic
    Segmentation", CVPR 2020.  Domain generalization application: DRGen,
    MICCAI 2022.

    Args:
        source: H x W x C uint8 image array.
        reference: H x W x C uint8 image array. Resized to match source if
            shapes differ.
        alpha: Fraction of the spectrum to swap, relative to min(H, W).

    Returns:
        Mixed uint8 array with the same shape as ``source``.
    """
    src = source.astype(np.float32)
    H, W = src.shape[:2]

    if reference.shape[:2] != (H, W):
        reference = cv2.resize(reference, (W, H), interpolation=cv2.INTER_LINEAR)
    ref = reference.astype(np.float32)

    b = max(1, int(np.floor(alpha * min(H, W))))
    cy, cx = H // 2, W // 2

    result = np.empty_like(src)
    for c in range(src.shape[2]):
        fft_src = np.fft.fftshift(np.fft.fft2(src[:, :, c]))
        fft_ref = np.fft.fftshift(np.fft.fft2(ref[:, :, c]))

        amp_src = np.abs(fft_src)
        pha_src = np.angle(fft_src)
        amp_src[cy - b : cy + b, cx - b : cx + b] = np.abs(fft_ref)[cy - b : cy + b, cx - b : cx + b]

        fft_mixed = np.fft.ifftshift(amp_src * np.exp(1j * pha_src))
        result[:, :, c] = np.fft.ifft2(fft_mixed).real

    return np.clip(result, 0, 255).astype(np.uint8)


def ampmix(
    source: np.ndarray,
    reference: np.ndarray,
    *,
    alpha_low: float = 0.0,
    alpha_high: float = 0.5,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """FDA-style amplitude mixing with a sample-wise random spectrum width."""
    generator = rng or np.random.default_rng()
    low = max(0.0, float(alpha_low))
    high = max(low, float(alpha_high))
    alpha = float(generator.uniform(low, high))
    return fda_mix(source, reference, alpha)


class GINAugment:
    """Reference-free random intensity mapping for segmentation DG training."""

    def __init__(
        self,
        num_filters: int = 8,
        activation: str = "leaky_relu",
        strength: float = 0.5,
    ) -> None:
        self._num_filters = max(1, int(num_filters))
        self._activation = str(activation).strip().lower()
        self._strength = float(np.clip(strength, 0.0, 1.0))
        self._rng = np.random.default_rng()

    def _activate(self, values: np.ndarray) -> np.ndarray:
        if self._activation == "relu":
            return np.maximum(values, 0.0)
        if self._activation == "tanh":
            return np.tanh(values)
        return np.where(values >= 0.0, values, 0.1 * values)

    def __call__(self, image: np.ndarray, **_: object) -> np.ndarray:
        if image.dtype != np.uint8:
            image = np.clip(image, 0, 255).astype(np.uint8)
        x = image.astype(np.float32) / 255.0
        flat = x.reshape(-1, 3)

        # Positive 1x1 weights keep the mapping intensity-oriented and reduce
        # the chance of turning lesion colors into arbitrary complements.
        w1 = np.abs(self._rng.normal(0.0, 1.0, size=(3, self._num_filters))).astype(np.float32)
        b1 = self._rng.normal(0.0, 0.15, size=(self._num_filters,)).astype(np.float32)
        hidden = self._activate((flat @ w1) + b1)
        w2 = np.abs(self._rng.normal(0.0, 1.0, size=(self._num_filters, 3))).astype(np.float32)
        b2 = self._rng.normal(0.0, 0.15, size=(3,)).astype(np.float32)
        mapped = (hidden @ w2) + b2

        mapped = mapped.reshape(x.shape)
        mapped_min = mapped.min(axis=(0, 1), keepdims=True)
        mapped_max = mapped.max(axis=(0, 1), keepdims=True)
        mapped = (mapped - mapped_min) / np.maximum(mapped_max - mapped_min, 1e-6)
        out = ((1.0 - self._strength) * x) + (self._strength * mapped)
        return np.clip(out * 255.0, 0, 255).astype(np.uint8)


def _gin_aug_from_config(gin_config: dict | None) -> A.BasicTransform | None:
    cfg = gin_config or {}
    if not bool(cfg.get("enable", False)):
        return None
    augmenter = GINAugment(
        num_filters=int(cfg.get("num_filters", 8)),
        activation=str(cfg.get("activation", "leaky_relu")),
        strength=float(cfg.get("strength", 0.5)),
    )
    return A.Lambda(image=augmenter, p=float(cfg.get("probability", 1.0)))


def _scale_jitter_from_config(scale_jitter_config: dict | None) -> A.BasicTransform | None:
    # Randomizes fundus scale/position so the segmenter becomes invariant to the
    # domain-dependent framing that QuickQual square-pad produces (foreground fill
    # ranges ~0.47-0.76 across domains, vs circular's uniform ~0.78). fill=0 matches
    # the black padding QuickQual uses; masks share the same affine via A.Compose.
    cfg = scale_jitter_config or {}
    if not bool(cfg.get("enable", False)):
        return None
    translate = float(cfg.get("translate", 0.05))
    return A.Affine(
        scale=(float(cfg.get("scale_min", 0.75)), float(cfg.get("scale_max", 1.05))),
        translate_percent={"x": (-translate, translate), "y": (-translate, translate)},
        rotate=0,
        border_mode=cv2.BORDER_CONSTANT,
        fill=0,
        fill_mask=0,
        p=float(cfg.get("probability", 0.7)),
    )


class _TrainTransform:
    """PIL-compatible wrapper for the albumentations-based training pipeline."""

    def __init__(self, pil_steps: list, aug: A.Compose) -> None:
        self._pil_steps = pil_steps
        self._aug = aug

    def __call__(self, img: PILImage.Image) -> torch.Tensor:
        for step in self._pil_steps:
            img = step(img)
        return self._aug(image=np.asarray(img))["image"]


class _SegmentationTransform:
    """PIL + mask transform with synchronized spatial augmentation."""

    def __init__(self, pil_steps: list, aug: A.Compose) -> None:
        self._pil_steps = pil_steps
        self._aug = aug

    def __call__(self, img: PILImage.Image, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        for step in self._pil_steps:
            img = step(img)
        mask_np = mask.detach().cpu().permute(1, 2, 0).numpy().astype(np.float32)
        image_np = np.asarray(img)
        image_h, image_w = image_np.shape[:2]
        if mask_np.shape[:2] != (image_h, image_w):
            mask_np = cv2.resize(mask_np, (image_w, image_h), interpolation=cv2.INTER_NEAREST)
            if mask_np.ndim == 2:
                mask_np = mask_np[..., None]
        out = self._aug(image=image_np, mask=mask_np)
        out_mask = out["mask"]
        if not torch.is_tensor(out_mask):
            out_mask = torch.as_tensor(out_mask)
        if out_mask.ndim == 2:
            out_mask = out_mask.unsqueeze(0)
        elif out_mask.shape[0] != mask.shape[0] and out_mask.shape[-1] == mask.shape[0]:
            out_mask = out_mask.permute(2, 0, 1)
        return out["image"], (out_mask.float() > 0.5).float()


_PREPROCESS_CONFIG_KEYS = (
    "preprocess_mode",
    "target_short_fill",
    "max_total_x_trim",
    "max_total_y_trim",
    "saliency_shift",
    "saliency_weight",
    "saliency_candidates",
    "safezoom_max_dim",
    "apply_ben_graham",
)


def preprocess_kwargs_from_config(*configs: dict | None) -> dict:
    kwargs: dict = {}
    for config in configs:
        if not config:
            continue
        for key in _PREPROCESS_CONFIG_KEYS:
            if key in config:
                kwargs[key] = config[key]
    return kwargs


_PREPROCESSED_PATH_PREFIXES = (
    "processed",
    "processed_quickqual",
    "processed_quickqual_1024",
    "processed_contentcrop",
    "processed_safezoom",
)


def is_preprocessed_image_path(image_path: object) -> bool:
    """True if the path lives under an offline-preprocessed image dir
    (`<prefix>/images/...`), i.e. already geometry+Ben-Graham normalized.

    Feeding such an image through a FundusPreprocess that applies Ben Graham
    again double-applies it (meta AUROC ~0.93 -> ~0.80). Used by the serve-side
    double-preprocess guard.
    """
    parts = str(image_path).replace("\\", "/").split("/")
    return any(
        parts[i] in _PREPROCESSED_PATH_PREFIXES
        and i + 1 < len(parts)
        and parts[i + 1] == "images"
        for i in range(len(parts))
    )


def build_train_transform(
    crop_size: int,
    resize_size: int | None = None,
    interpolation: str = "bilinear",
    mean: tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: tuple[float, float, float] = (0.229, 0.224, 0.225),
    use_preprocessing: bool = False,
    use_random_resized_crop: bool = True,
    preprocess_kwargs: dict | None = None,
) -> _TrainTransform:
    resize = resize_size or crop_size

    pil_steps: list = []
    if use_preprocessing:
        pil_steps.append(FundusPreprocess(**(preprocess_kwargs or {})))

    resize_steps: list = [A.Resize(resize, resize)]
    if use_random_resized_crop:
        # Lower bound 0.8 (not 0.7) to avoid cropping out peripheral lesions.
        resize_steps.append(A.RandomResizedCrop(size=(crop_size, crop_size), scale=(0.8, 1.0)))
    elif resize != crop_size:
        resize_steps.append(A.CenterCrop(crop_size, crop_size))

    aug = A.Compose([
        *resize_steps,

        # Geometric: fundus lesions are rotation-invariant; all orientations valid.
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Rotate(limit=180, border_mode=cv2.BORDER_CONSTANT, fill=0, p=0.8),

        # Photometric: simulate camera/exposure variation across domains.
        # hue is capped at 0.03 -- hemorrhages (red) and exudates (yellow-white)
        # carry diagnostic color information that must not be distorted.
        A.ColorJitter(brightness=0.35, contrast=0.35, saturation=0.25, hue=0.03, p=0.8),
        A.RandomGamma(gamma_limit=(75, 130), p=0.5),

        # Sensor and optics variation: low probability / low magnitude to
        # preserve fine lesion morphology.
        A.GaussianBlur(blur_limit=(3, 5), p=0.2),
        A.GaussNoise(std_range=(0.02, 0.07), p=0.3),

        A.Normalize(mean=mean, std=std),
        ToTensorV2(),
    ])

    return _TrainTransform(pil_steps, aug)


def build_segmentation_train_transform(
    crop_size: int,
    resize_size: int | None = None,
    interpolation: str = "bilinear",
    mean: tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: tuple[float, float, float] = (0.229, 0.224, 0.225),
    use_preprocessing: bool = False,
    use_random_resized_crop: bool = True,
    preprocess_kwargs: dict | None = None,
    gin_config: dict | None = None,
    scale_jitter_config: dict | None = None,
) -> _SegmentationTransform:
    resize = resize_size or crop_size

    pil_steps: list = []
    if use_preprocessing:
        pil_steps.append(FundusPreprocess(**(preprocess_kwargs or {})))

    resize_steps: list = [A.Resize(resize, resize)]
    if use_random_resized_crop:
        resize_steps.append(A.RandomResizedCrop(size=(crop_size, crop_size), scale=(0.8, 1.0)))
    elif resize != crop_size:
        resize_steps.append(A.CenterCrop(crop_size, crop_size))

    gin_aug = _gin_aug_from_config(gin_config)
    scale_jitter_aug = _scale_jitter_from_config(scale_jitter_config)
    photometric_steps: list[A.BasicTransform] = [
        A.ColorJitter(brightness=0.35, contrast=0.35, saturation=0.25, hue=0.03, p=0.8),
        A.RandomGamma(gamma_limit=(75, 130), p=0.5),
    ]
    if gin_aug is not None:
        photometric_steps.append(gin_aug)

    spatial_steps: list[A.BasicTransform] = list(resize_steps)
    if scale_jitter_aug is not None:
        spatial_steps.append(scale_jitter_aug)

    aug = A.Compose([
        *spatial_steps,
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Rotate(limit=180, border_mode=cv2.BORDER_CONSTANT, fill=0, p=0.8),
        *photometric_steps,
        A.GaussianBlur(blur_limit=(3, 5), p=0.2),
        A.GaussNoise(std_range=(0.02, 0.07), p=0.3),
        A.Normalize(mean=mean, std=std),
        ToTensorV2(transpose_mask=True),
    ])

    return _SegmentationTransform(pil_steps, aug)


def build_segmentation_eval_transform(
    crop_size: int,
    resize_size: int | None = None,
    interpolation: str = "bilinear",
    mean: tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: tuple[float, float, float] = (0.229, 0.224, 0.225),
    use_preprocessing: bool = False,
    preprocess_kwargs: dict | None = None,
) -> _SegmentationTransform:
    resize = resize_size or crop_size

    pil_steps: list = []
    if use_preprocessing:
        pil_steps.append(FundusPreprocess(**(preprocess_kwargs or {})))

    resize_steps: list = [A.Resize(resize, resize)]
    if resize != crop_size:
        resize_steps.append(A.CenterCrop(crop_size, crop_size))

    aug = A.Compose([
        *resize_steps,
        A.Normalize(mean=mean, std=std),
        ToTensorV2(transpose_mask=True),
    ])
    return _SegmentationTransform(pil_steps, aug)


def build_eval_transform(
    crop_size: int,
    resize_size: int | None = None,
    interpolation: str = "bilinear",
    mean: tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: tuple[float, float, float] = (0.229, 0.224, 0.225),
    use_preprocessing: bool = False,
    preprocess_kwargs: dict | None = None,
) -> transforms.Compose:
    resize = resize_size or crop_size
    interpolation_mode = resolve_interpolation_mode(interpolation)
    steps = []
    if use_preprocessing:
        steps.append(FundusPreprocess(**(preprocess_kwargs or {})))
    steps.extend(
        [
            transforms.Resize((resize, resize), interpolation=interpolation_mode),
            transforms.CenterCrop(crop_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )
    return transforms.Compose(steps)


