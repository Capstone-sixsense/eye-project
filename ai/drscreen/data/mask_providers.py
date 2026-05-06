from __future__ import annotations

import re
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np
import torch

_SEG_SPLIT = {"a. Training Set": range(1, 55), "b. Testing Set": range(55, 82)}
_SEG_ID_RE = re.compile(r"IDRiD_(\d+)")


@runtime_checkable
class LesionMaskProvider(Protocol):
    """Protocol for loading a union lesion mask for a single dataset row."""

    def load(self, image_path: str, domain: str, size: int) -> tuple[torch.Tensor, bool]:
        """Return (mask [1, size, size] float32, is_valid)."""
        ...


class NullMaskProvider:
    """Always returns a zero mask — use when no pixel-level supervision is needed."""

    def load(self, image_path: str, domain: str, size: int) -> tuple[torch.Tensor, bool]:
        return torch.zeros(1, size, size), False


class IDRiDMaskProvider:
    """Loads IDRiD union lesion masks (MA+HE+EX+SE) from the segmentation groundtruth directory.

    Masks exist only for IDRiD images in the Disease Grading a. Training Set
    with numeric IDs 1–81. Returns a zero tensor for all other images.
    """

    def __init__(self, seg_mask_dir: str | Path) -> None:
        self._seg_mask_dir = Path(seg_mask_dir)

    def load(self, image_path: str, domain: str, size: int) -> tuple[torch.Tensor, bool]:
        zeros = torch.zeros(1, size, size)

        if domain != "IDRiD":
            return zeros, False
        if "a. Training Set" not in image_path:
            return zeros, False

        m = _SEG_ID_RE.search(image_path)
        if not m:
            return zeros, False
        num = int(m.group(1))

        seg_split_dir: str | None = None
        for split_name, id_range in _SEG_SPLIT.items():
            if num in id_range:
                seg_split_dir = split_name
                break
        if seg_split_dir is None:
            return zeros, False

        stem = f"IDRiD_{num:02d}"
        from drscreen.xai.iou import load_lesion_masks, union_mask
        masks = load_lesion_masks(
            self._seg_mask_dir / seg_split_dir,
            stem,
            target_size=(size, size),
        )
        gt = union_mask(masks)
        if gt is None:
            return zeros, False

        return torch.from_numpy(gt.astype(np.float32)).unsqueeze(0), True
