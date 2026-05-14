from __future__ import annotations

import re
from pathlib import Path
from typing import Protocol, runtime_checkable

import cv2
import numpy as np
import torch

_SEG_SPLIT = {"a. Training Set": range(1, 55), "b. Testing Set": range(55, 82)}
_SEG_ID_RE = re.compile(r"IDRiD_(\d+)")


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

    This training-time provider loads masks for IDRiD disease-grading training
    rows whose numeric ID maps to the segmentation groundtruth naming. It
    returns a zero tensor for all other rows.
    """

    def __init__(self, seg_mask_dir: str | Path, channels: int) -> None:
        self._seg_mask_dir = Path(seg_mask_dir)
        self._channels = int(channels)

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

        seg_split_dir: str | None = None
        for split_name, id_range in _SEG_SPLIT.items():
            if num in id_range:
                seg_split_dir = split_name
                break
        if seg_split_dir is None:
            return {}, False

        stem = f"IDRiD_{num:02d}"
        from drscreen.xai.iou import load_lesion_masks

        masks = load_lesion_masks(
            self._seg_mask_dir / seg_split_dir,
            stem,
            target_size=(size, size),
        )
        return masks, bool(masks)


class IDRiDMaskProvider(_IDRiDBaseMaskProvider):
    """Loads IDRiD union lesion masks (MA+HE+EX+SE)."""

    def __init__(self, seg_mask_dir: str | Path) -> None:
        super().__init__(seg_mask_dir, channels=1)

    def load(self, image_path: str, domain: str, size: int) -> tuple[torch.Tensor, bool]:
        zeros = torch.zeros(1, size, size)

        masks, is_valid = self._load_masks(image_path, domain, size)
        if not is_valid:
            return zeros, False

        from drscreen.xai.iou import union_mask

        gt = union_mask(masks)
        if gt is None:
            return zeros, False

        return torch.from_numpy(gt.astype(np.float32)).unsqueeze(0), True


class IDRiDPerLesionMaskProvider(_IDRiDBaseMaskProvider):
    """Loads IDRiD MA/HE/EX/SE masks as four independent channels."""

    def __init__(self, seg_mask_dir: str | Path) -> None:
        super().__init__(seg_mask_dir, channels=4)

    def load(self, image_path: str, domain: str, size: int) -> tuple[torch.Tensor, bool]:
        zeros = torch.zeros(self._channels, size, size)

        masks, is_valid = self._load_masks(image_path, domain, size)
        if not is_valid:
            return zeros, False

        from drscreen.xai.iou import LESION_CODES

        channels = [
            torch.from_numpy(
                masks.get(code, np.zeros((size, size), dtype=np.uint8)).astype(np.float32)
            )
            for code in LESION_CODES
        ]
        return torch.stack(channels, dim=0), True


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

    def __init__(self, annotations_dir: str | Path) -> None:
        self._ann_dir = Path(annotations_dir)

    def load(self, image_path: str, domain: str, size: int) -> tuple[torch.Tensor, bool]:
        zeros = torch.zeros(4, size, size)
        if domain.lower() != "messidor":
            return zeros, False

        stem = Path(image_path).stem
        channels: list[torch.Tensor] = []
        any_valid = False

        for lesion_dir in self._CHANNEL_DIRS:
            mask_path = self._ann_dir / lesion_dir / f"{stem}.png"
            if mask_path.exists():
                from PIL import Image as PILImage
                arr = np.array(PILImage.open(mask_path), dtype=np.uint8)
                if arr.shape != (size, size):
                    arr = cv2.resize(arr, (size, size), interpolation=cv2.INTER_NEAREST)
                channels.append(torch.from_numpy(arr.astype(np.float32)))
                any_valid = True
            else:
                channels.append(torch.zeros(size, size))

        if not any_valid:
            return zeros, False

        return torch.stack(channels, dim=0), True
