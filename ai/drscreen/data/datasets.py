from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

from drscreen.data.mask_providers import IDRiDMaskProvider, LesionMaskProvider, NullMaskProvider
from drscreen.data.transforms import fda_mix


class ManifestDataset(Dataset):
    def __init__(
        self,
        manifest_path: str | Path,
        image_root: str | Path | None = None,
        split: str | None = None,
        transform: Callable[[Image.Image], Any] | None = None,
        seg_mask_dir: str | Path | None = None,
        seg_mask_size: int = 512,
        mask_provider: LesionMaskProvider | None = None,
    ) -> None:
        self.manifest_path = Path(manifest_path)
        self.image_root = Path(image_root) if image_root else self.manifest_path.parent
        self.transform = transform
        self._seg_mask_size = seg_mask_size

        # mask_provider takes precedence; seg_mask_dir is a convenience shorthand
        if mask_provider is not None:
            self._mask_provider: LesionMaskProvider = mask_provider
        elif seg_mask_dir is not None:
            self._mask_provider = IDRiDMaskProvider(seg_mask_dir)
        else:
            self._mask_provider = NullMaskProvider()

        frame = pd.read_csv(self.manifest_path)
        required_columns = {"image_path", "label", "split"}
        missing = required_columns.difference(frame.columns)
        if missing:
            raise ValueError(f"Manifest is missing columns: {sorted(missing)}")

        if split is not None:
            frame = frame[frame["split"] == split].reset_index(drop=True)

        self.frame = frame

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.frame.iloc[index]
        image_path = self.image_root / str(row["image_path"])
        image = Image.open(image_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        domain = str(row["domain"]) if "domain" in self.frame.columns else ""
        seg_mask, seg_mask_valid = self._mask_provider.load(
            str(row["image_path"]), domain, self._seg_mask_size
        )
        return {
            "image": image,
            "label": int(row["label"]),
            "image_path": str(image_path),
            "split": str(row["split"]),
            "domain": domain,
            "seg_mask": seg_mask,
            "seg_mask_valid": seg_mask_valid,
        }


class FDAManifestDataset(ManifestDataset):
    """ManifestDataset with on-the-fly Fourier Domain Adaptation.

    For each training sample, a reference image is drawn from a different
    source domain (when available) and its low-frequency Fourier amplitude is
    transferred into the source image via fda_mix(). This encourages the model
    to learn features that are invariant to global color and illumination
    differences across acquisition domains.

    FDA is applied on the raw loaded pixel values before the standard training
    transform pipeline (augmentation + normalisation), so it interacts with
    ColorJitter and other photometric augmentations as intended.

    Call rebuild_domain_indices() after any external modification of self.frame
    (e.g. domain exclusion filtering in runner.py) to keep reference sampling
    consistent.

    Args:
        fda_alpha: Fraction of the frequency spectrum to swap, relative to
            min(H, W). Standard value from the FDA paper is 0.05.
        domain_column: Manifest column identifying the source domain. When
            absent or only one domain is present, a random same-dataset image
            is used as the reference instead.
    """

    def __init__(
        self,
        manifest_path: str | Path,
        image_root: str | Path | None = None,
        split: str | None = None,
        transform: Callable[[Image.Image], Any] | None = None,
        fda_alpha: float = 0.05,
        domain_column: str = "domain",
        seg_mask_dir: str | Path | None = None,
        seg_mask_size: int = 512,
        mask_provider: LesionMaskProvider | None = None,
    ) -> None:
        super().__init__(
            manifest_path, image_root, split, transform,
            seg_mask_dir=seg_mask_dir, seg_mask_size=seg_mask_size,
            mask_provider=mask_provider,
        )
        self._alpha = fda_alpha
        self._domain_column = domain_column
        self._rng = np.random.default_rng()
        self._domain_indices: dict[str, list[int]] = {}
        self.rebuild_domain_indices()

    def rebuild_domain_indices(self) -> None:
        """Rebuild the domain -> iloc-position index from the current frame.

        Must be called after self.frame is replaced externally so that
        cross-domain reference sampling reflects the updated dataset.
        """
        self._domain_indices = {}
        if self._domain_column not in self.frame.columns:
            return
        for iloc_pos in range(len(self.frame)):
            domain = str(self.frame.iloc[iloc_pos][self._domain_column])
            self._domain_indices.setdefault(domain, []).append(iloc_pos)

    def _sample_ref_index(self, source_index: int) -> int:
        """Return an iloc position for a reference image.

        Prefers a different source domain. Falls back to any other index
        in the dataset when no cross-domain candidate exists.
        """
        if len(self._domain_indices) > 1:
            src_domain = str(self.frame.iloc[source_index][self._domain_column])
            other_domains = [d for d in self._domain_indices if d != src_domain]
            if other_domains:
                chosen = self._rng.choice(other_domains)
                return int(self._rng.choice(self._domain_indices[chosen]))

        candidates = [i for i in range(len(self.frame)) if i != source_index]
        return int(self._rng.choice(candidates)) if candidates else source_index

    def _load_raw_array(self, iloc_pos: int) -> np.ndarray:
        row = self.frame.iloc[iloc_pos]
        image_path = self.image_root / str(row["image_path"])
        with Image.open(image_path) as img:
            return np.asarray(img.convert("RGB"))

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.frame.iloc[index]
        src_arr = self._load_raw_array(index)
        ref_arr = self._load_raw_array(self._sample_ref_index(index))

        mixed = fda_mix(src_arr, ref_arr, self._alpha)
        image = Image.fromarray(mixed)

        if self.transform is not None:
            image = self.transform(image)

        domain = str(row["domain"]) if "domain" in self.frame.columns else ""
        seg_mask, seg_mask_valid = self._mask_provider.load(
            str(row["image_path"]), domain, self._seg_mask_size
        )
        return {
            "image": image,
            "label": int(row["label"]),
            "image_path": str(self.image_root / str(row["image_path"])),
            "split": str(row["split"]),
            "domain": domain,
            "seg_mask": seg_mask,
            "seg_mask_valid": seg_mask_valid,
        }
