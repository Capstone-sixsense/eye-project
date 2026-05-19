from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path
from typing import Protocol, runtime_checkable

import cv2
import numpy as np
import torch

_SEG_TRAIN_SPLIT = "a. Training Set"
_SEG_TRAIN_IDS = range(1, 55)
_SEG_SPLIT = {_SEG_TRAIN_SPLIT: _SEG_TRAIN_IDS, "b. Testing Set": range(55, 82)}
_SEG_ID_RE = re.compile(r"IDRiD_(\d+)")
_IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".tif", ".tiff")
_PROCESSED_PREFIX = ("processed", "images")


def _infer_raw_root(path: Path) -> Path:
    resolved = path.resolve()
    for candidate in (resolved, *resolved.parents):
        if candidate.name.lower() == "raw":
            return candidate
    return Path("data/raw")


def _is_processed_image_path(image_path: str) -> bool:
    parts = Path(image_path).parts
    return len(parts) >= 2 and tuple(parts[:2]) == _PROCESSED_PREFIX


def _raw_relative_image_path(image_path: str) -> Path:
    path = Path(image_path)
    parts = path.parts
    if len(parts) >= 2 and tuple(parts[:2]) == _PROCESSED_PREFIX:
        return Path(*parts[2:])
    return path


def _resolve_raw_image_path(raw_root: Path, image_path: str) -> Path | None:
    relative = _raw_relative_image_path(image_path)
    direct = raw_root / relative
    if direct.exists():
        return direct
    stem_path = direct.with_suffix("")
    for ext in _IMAGE_EXTENSIONS:
        candidate = stem_path.with_suffix(ext)
        if candidate.exists():
            return candidate
    return None


def _resize_mask_array(mask: np.ndarray, size: int) -> np.ndarray:
    if mask.shape[:2] == (size, size):
        return mask
    resized = cv2.resize(mask, (size, size), interpolation=cv2.INTER_NEAREST)
    if mask.ndim == 3 and mask.shape[-1] == 1 and resized.ndim == 2:
        resized = resized[..., None]
    return resized


@lru_cache(maxsize=4096)
def _cached_preprocess_geometry(
    reference_path: str,
) -> tuple[tuple[int, int], tuple[int, int, int, int, int, int, int, int] | None]:
    from PIL import Image as PILImage

    from drscreen.data.transforms import FundusPreprocess

    with PILImage.open(reference_path) as reference:
        ref = np.asarray(reference.convert("RGB"))
    preprocessor = FundusPreprocess(align=False)
    return ref.shape[:2], preprocessor._circular_crop_geometry(ref)  # noqa: SLF001 - shared geometry


def _apply_cached_geometry(
    mask: np.ndarray,
    *,
    reference_path: Path,
    size: int,
) -> np.ndarray:
    ref_shape, geometry = _cached_preprocess_geometry(str(reference_path))
    was_singleton_channel = mask.ndim == 3 and mask.shape[-1] == 1
    mask_arr = np.asarray(mask)
    if mask_arr.shape[:2] != ref_shape:
        mask_arr = cv2.resize(
            mask_arr,
            (ref_shape[1], ref_shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )
        if was_singleton_channel and mask_arr.ndim == 2:
            mask_arr = mask_arr[..., None]

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

    return _resize_mask_array(mask_arr, size)


def _align_mask_to_image_preprocessing(
    mask: np.ndarray,
    *,
    image_path: str,
    raw_root: Path,
    size: int,
) -> np.ndarray:
    """Map raw mask geometry into the offline-preprocessed image coordinate system."""
    if not _is_processed_image_path(image_path):
        return _resize_mask_array(mask, size)

    reference_path = _resolve_raw_image_path(raw_root, image_path)
    if reference_path is None:
        return _resize_mask_array(mask, size)

    aligned = _apply_cached_geometry(mask, reference_path=reference_path, size=size)
    return (aligned > 0).astype(np.float32)


@runtime_checkable
class LesionMaskProvider(Protocol):
    """Protocol for loading lesion masks for a single dataset row."""

    def load(self, image_path: str, domain: str, size: int) -> tuple[torch.Tensor, bool]:
        """Return (mask [C, size, size] float32, is_valid)."""
        ...


class NullMaskProvider:
    """Always returns a zero mask — use when no pixel-level supervision is needed."""

    def __init__(self, channels: int = 1) -> None:
        self._channels = int(channels)

    def load(self, image_path: str, domain: str, size: int) -> tuple[torch.Tensor, bool]:
        return torch.zeros(self._channels, size, size), False


class _IDRiDBaseMaskProvider:
    """Base loader for IDRiD lesion masks from the segmentation groundtruth directory.

    This training-time provider loads masks only for the IDRiD segmentation
    training IDs (1-54). IDRiD disease-grading training rows also include
    IDs 55-81, but those correspond to the segmentation test set and must not
    be used as training supervision.
    """

    def __init__(self, seg_mask_dir: str | Path, channels: int, raw_root: str | Path | None = None) -> None:
        self._seg_mask_dir = Path(seg_mask_dir)
        self._channels = int(channels)
        self._raw_root = Path(raw_root) if raw_root is not None else _infer_raw_root(self._seg_mask_dir)

    def _load_masks(
        self,
        image_path: str,
        domain: str,
        size: int,
    ) -> tuple[dict[str, np.ndarray], bool]:
        if domain != "IDRiD":
            return {}, False
        if "a. Training Set" not in image_path:
            return {}, False

        m = _SEG_ID_RE.search(image_path)
        if not m:
            return {}, False
        num = int(m.group(1))

        if num not in _SEG_TRAIN_IDS:
            return {}, False

        stem = f"IDRiD_{num:02d}"
        from drscreen.xai.iou import load_lesion_masks

        masks = load_lesion_masks(
            self._seg_mask_dir / _SEG_TRAIN_SPLIT,
            stem,
        )
        return masks, bool(masks)


class IDRiDMaskProvider(_IDRiDBaseMaskProvider):
    """Loads IDRiD union lesion masks (MA+HE+EX+SE)."""

    def __init__(self, seg_mask_dir: str | Path, raw_root: str | Path | None = None) -> None:
        super().__init__(seg_mask_dir, channels=1, raw_root=raw_root)

    def load(self, image_path: str, domain: str, size: int) -> tuple[torch.Tensor, bool]:
        zeros = torch.zeros(1, size, size)

        masks, is_valid = self._load_masks(image_path, domain, size)
        if not is_valid:
            return zeros, False

        from drscreen.xai.iou import union_mask

        gt = union_mask(masks)
        if gt is None:
            return zeros, False
        gt = _align_mask_to_image_preprocessing(
            gt,
            image_path=image_path,
            raw_root=self._raw_root,
            size=size,
        )

        return torch.from_numpy(gt.astype(np.float32)).unsqueeze(0), True


class IDRiDPerLesionMaskProvider(_IDRiDBaseMaskProvider):
    """Loads IDRiD MA/HE/EX/SE masks as four independent channels."""

    def __init__(self, seg_mask_dir: str | Path, raw_root: str | Path | None = None) -> None:
        super().__init__(seg_mask_dir, channels=4, raw_root=raw_root)

    def load(self, image_path: str, domain: str, size: int) -> tuple[torch.Tensor, bool]:
        zeros = torch.zeros(self._channels, size, size)

        masks, is_valid = self._load_masks(image_path, domain, size)
        if not is_valid:
            return zeros, False

        from drscreen.xai.iou import LESION_CODES

        raw_channels = []
        shape = next(iter(masks.values())).shape
        for code in LESION_CODES:
            raw_channels.append(masks.get(code, np.zeros(shape, dtype=np.uint8)))
        stacked = np.stack(raw_channels, axis=-1)
        aligned = _align_mask_to_image_preprocessing(
            stacked,
            image_path=image_path,
            raw_root=self._raw_root,
            size=size,
        )
        return torch.from_numpy(aligned.astype(np.float32)).permute(2, 0, 1), True


class MAPLESTrainMaskProvider:
    """Loads MAPLES-DR lesion masks for training rows with ``domain == 'MAPLES'``.

    MAPLES-DR provides MA / HE / EX / CWS pixel masks for a subset of MESSIDOR
    images. For training-time use, we return the union mask (single channel)
    when ``channels=1`` or a 4-channel per-lesion mask when ``channels=4``.
    Returns a zero tensor for rows whose domain is not ``'MAPLES'``.

    Channel order when 4-channel: MA / HE / EX / SE (== CWS in MAPLES).
    """

    _CHANNEL_DIRS = ("Microaneurysms", "Hemorrhages", "Exudates", "CottonWoolSpots")

    def __init__(
        self,
        annotations_dir: str | Path,
        channels: int = 1,
        raw_root: str | Path | None = None,
    ) -> None:
        self._ann_dir = Path(annotations_dir)
        self._channels = int(channels)
        self._raw_root = Path(raw_root) if raw_root is not None else _infer_raw_root(self._ann_dir)
        if self._channels not in (1, 4):
            raise ValueError("MAPLESTrainMaskProvider channels must be 1 or 4")

    def load(self, image_path: str, domain: str, size: int) -> tuple[torch.Tensor, bool]:
        zeros = torch.zeros(self._channels, size, size)
        if domain != "MAPLES":
            return zeros, False

        stem = Path(image_path).stem
        from PIL import Image as PILImage

        raw_channels: list[np.ndarray | None] = []
        base_shape: tuple[int, int] | None = None
        for lesion_dir in self._CHANNEL_DIRS:
            mask_path = self._ann_dir / lesion_dir / f"{stem}.png"
            if mask_path.exists():
                arr = np.array(PILImage.open(mask_path), dtype=np.uint8)
                if arr.ndim == 3:
                    arr = arr[..., 0]
                base_shape = arr.shape[:2]
                raw_channels.append((arr > 0).astype(np.uint8))
            else:
                raw_channels.append(None)

        if base_shape is None:
            return zeros, False

        stacked = np.stack(
            [
                arr if arr is not None else np.zeros(base_shape, dtype=np.uint8)
                for arr in raw_channels
            ],
            axis=-1,
        )
        aligned = _align_mask_to_image_preprocessing(
            stacked,
            image_path=image_path,
            raw_root=self._raw_root,
            size=size,
        )

        if self._channels == 1:
            union = aligned.max(axis=-1, keepdims=True)
            return torch.from_numpy(union.astype(np.float32)).permute(2, 0, 1), True
        return torch.from_numpy(aligned.astype(np.float32)).permute(2, 0, 1), True


class TJDRMaskProvider:
    """Loads TJDR palette-label masks as union or MA/HE/EX/SE channels.

    TJDR annotation labels are 0=background, 1=EX, 2=HE, 3=MA, 4=SE. The
    project-wide channel order is MA / HE / EX / SE, so the 4-channel mapping is
    3 → MA, 2 → HE, 1 → EX, 4 → SE.
    """

    _LABEL_BY_CODE = {"MA": 3, "HE": 2, "EX": 1, "SE": 4}
    _CHANNEL_CODES = ("MA", "HE", "EX", "SE")

    def __init__(
        self,
        root_dir: str | Path,
        channels: int = 1,
        raw_root: str | Path | None = None,
    ) -> None:
        self._root_dir = Path(root_dir)
        self._channels = int(channels)
        self._raw_root = Path(raw_root) if raw_root is not None else _infer_raw_root(self._root_dir)
        if self._channels not in (1, 4):
            raise ValueError("TJDRMaskProvider channels must be 1 or 4")

    def _annotation_path(self, image_path: str) -> Path | None:
        path = Path(image_path)
        parts = path.parts
        split = "train" if "train" in parts else "test" if "test" in parts else ""
        if not split:
            return None
        return self._root_dir / split / "annotation" / f"{path.stem}.png"

    def load(self, image_path: str, domain: str, size: int) -> tuple[torch.Tensor, bool]:
        zeros = torch.zeros(self._channels, size, size)
        if domain != "TJDR":
            return zeros, False

        mask_path = self._annotation_path(image_path)
        if mask_path is None or not mask_path.exists():
            return zeros, False

        from PIL import Image as PILImage

        arr = np.array(PILImage.open(mask_path), dtype=np.uint8)
        if arr.ndim == 3:
            arr = arr[..., 0]

        if self._channels == 1:
            union = ((arr >= 1) & (arr <= 4)).astype(np.uint8)
            union = _align_mask_to_image_preprocessing(
                union,
                image_path=image_path,
                raw_root=self._raw_root,
                size=size,
            )
            return torch.from_numpy(union).unsqueeze(0), True

        channels = np.stack(
            [(arr == self._LABEL_BY_CODE[code]).astype(np.uint8) for code in self._CHANNEL_CODES],
            axis=-1,
        )
        aligned = _align_mask_to_image_preprocessing(
            channels,
            image_path=image_path,
            raw_root=self._raw_root,
            size=size,
        )
        return torch.from_numpy(aligned.astype(np.float32)).permute(2, 0, 1), True


class CompositeMaskProvider:
    """Dispatches mask loading across multiple per-domain providers.

    Each row is offered to each provider in order; the first one that returns
    ``is_valid=True`` wins. Used to attach IDRiD masks to ``domain='IDRiD'``
    rows and MAPLES masks to ``domain='MAPLES'`` rows from the same training
    manifest.
    """

    def __init__(self, providers: list) -> None:
        if not providers:
            raise ValueError("CompositeMaskProvider needs at least one provider")
        self._providers = list(providers)

    def load(self, image_path: str, domain: str, size: int) -> tuple[torch.Tensor, bool]:
        for p in self._providers:
            mask, valid = p.load(image_path, domain, size)
            if valid:
                return mask, True
        return self._providers[0].load(image_path, domain, size)


class MAPLESMaskProvider:
    """Loads MAPLES-DR MA/HE/EX/CWS masks as four independent channels.

    Channel order matches IDRiDPerLesionMaskProvider: MA / HE / EX / SE(CWS).
    Works for MESSIDOR-domain images included in the MAPLES-DR dataset.
    Returns zeros for any MESSIDOR image not present in MAPLES-DR.

    Args:
        annotations_dir: Path to the MAPLES-DR annotations directory,
            e.g. "data/raw/MAPLES-DR/AdditionalData/annotations".
            Expected subdirs: Microaneurysms, Hemorrhages, Exudates, CottonWoolSpots.
    """

    _CHANNEL_DIRS = ("Microaneurysms", "Hemorrhages", "Exudates", "CottonWoolSpots")

    def __init__(self, annotations_dir: str | Path, raw_root: str | Path | None = None) -> None:
        self._ann_dir = Path(annotations_dir)
        self._raw_root = Path(raw_root) if raw_root is not None else _infer_raw_root(self._ann_dir)

    def load(self, image_path: str, domain: str, size: int) -> tuple[torch.Tensor, bool]:
        zeros = torch.zeros(4, size, size)
        if domain.lower() != "messidor":
            return zeros, False

        stem = Path(image_path).stem
        raw_channels: list[np.ndarray | None] = []
        base_shape: tuple[int, int] | None = None

        for lesion_dir in self._CHANNEL_DIRS:
            mask_path = self._ann_dir / lesion_dir / f"{stem}.png"
            if mask_path.exists():
                from PIL import Image as PILImage
                arr = np.array(PILImage.open(mask_path), dtype=np.uint8)
                if arr.ndim == 3:
                    arr = arr[..., 0]
                base_shape = arr.shape[:2]
                raw_channels.append((arr > 0).astype(np.uint8))
            else:
                raw_channels.append(None)

        if base_shape is None:
            return zeros, False

        stacked = np.stack(
            [
                arr if arr is not None else np.zeros(base_shape, dtype=np.uint8)
                for arr in raw_channels
            ],
            axis=-1,
        )
        aligned = _align_mask_to_image_preprocessing(
            stacked,
            image_path=image_path,
            raw_root=self._raw_root,
            size=size,
        )
        return torch.from_numpy(aligned.astype(np.float32)).permute(2, 0, 1), True
