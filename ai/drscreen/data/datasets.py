"""학습/평가용 PyTorch Dataset 정의(manifest CSV 기반).

manifest CSV(image_path/label/split/domain 등)를 읽어 이미지 + 라벨 + 선택적
병변 마스크 + 개념(concept) 라벨을 묶어 반환한다.

클래스 계층:
- ManifestDataset: 기본. 이미지/마스크에 각자 transform 적용.
- SegmentationManifestDataset: 이미지와 마스크에 '동기화된' 공간 증강 적용
  (분할/aux-seg 학습에서 crop/flip/rotate가 둘에 동일하게 들어가야 함).
- FDAManifestDataset / SegmentationFDAManifestDataset: 위에 더해, 다른 도메인의
  이미지를 reference로 뽑아 FDA(주파수영역 스타일 이식)로 도메인 일반화를 노린다.

마스크는 mask_provider가 도메인별로 제공한다(없으면 NullMaskProvider가 0 마스크).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from collections.abc import Callable

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

from drscreen.data.mask_providers import (
    IDRiDMaskProvider,
    LesionMaskProvider,
    NullMaskProvider,
)
from drscreen.data.transforms import ampmix, fda_mix

# 개념(concept) 4종 코드. CBM 등에서 약지도(weak label) 개념 벡터의 채널 순서.
_CONCEPT_CODES = ("MA", "HE", "EX", "SE")


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
        concept_label_path: str | Path | None = None,
    ) -> None:
        self.manifest_path = Path(manifest_path)
        self.image_root = Path(image_root) if image_root else self.manifest_path.parent
        self.transform = transform
        self._seg_mask_size = seg_mask_size

        # 마스크 공급자 우선순위: 명시적 mask_provider > seg_mask_dir(IDRiD 단축) > 없음(0 마스크).
        # mask_provider takes precedence; seg_mask_dir is a convenience shorthand
        if mask_provider is not None:
            self._mask_provider: LesionMaskProvider = mask_provider
        elif seg_mask_dir is not None:
            self._mask_provider = IDRiDMaskProvider(seg_mask_dir)
        else:
            self._mask_provider = NullMaskProvider()
        self._concept_index = self._load_concept_index(concept_label_path)

        frame = pd.read_csv(self.manifest_path)
        required_columns = {"image_path", "label", "split"}
        missing = required_columns.difference(frame.columns)
        if missing:
            raise ValueError(f"Manifest is missing columns: {sorted(missing)}")

        if split is not None:
            frame = frame[frame["split"] == split].reset_index(drop=True)

        self.frame = frame

    @staticmethod
    def _load_concept_index(path: str | Path | None) -> dict[str, dict[str, Any]]:
        if not path:
            return {}
        concept_path = Path(path)
        if not concept_path.exists():
            raise FileNotFoundError(f"Concept label CSV not found: {concept_path}")
        frame = pd.read_csv(concept_path)
        required = {"image_id", *_CONCEPT_CODES, "weak_label_valid", "concept_confidence"}
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"Concept label CSV missing columns: {sorted(missing)}")

        index: dict[str, dict[str, Any]] = {}
        for row in frame.to_dict("records"):
            image_id = str(row.get("image_id", "")).strip()
            if not image_id:
                continue
            keys = {image_id, Path(image_id).stem}
            image_path = str(row.get("image_path", "")).strip()
            if image_path:
                keys.add(Path(image_path).stem)
            for key in keys:
                index.setdefault(key, row)
        return index

    @staticmethod
    def _idrid_segmentation_key(image_id: str, image_path: str, domain: str) -> str | None:
        # IDRiD segmentation 학습셋(ID 1~54)만 픽셀 마스크 키를 부여한다. ID 55~81은
        # segmentation '테스트셋'이라 학습 supervision으로 쓰면 안 된다(7절 오염 참조).
        if domain != "IDRiD" or "a. Training Set" not in image_path:
            return None
        import re

        match = re.search(r"IDRiD_(\d+)", image_id) or re.search(r"IDRiD_(\d+)", image_path)
        if match is None:
            return None
        num = int(match.group(1))
        if 1 <= num <= 54:
            return f"IDRiD_{num:02d}"
        return None

    def _load_concepts(self, row: pd.Series) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, str]:
        zeros = torch.zeros(len(_CONCEPT_CODES), dtype=torch.float32)
        if not self._concept_index:
            return zeros, torch.tensor(False), torch.tensor(0.0), ""

        image_id = str(row.get("image_id", "")).strip()
        image_path = str(row.get("image_path", "")).strip()
        domain = str(row.get("domain", "")).strip()
        keys: list[str] = []
        idrid_key = self._idrid_segmentation_key(image_id, image_path, domain)
        if idrid_key:
            keys.append(idrid_key)
        keys.extend([image_id, Path(image_id).stem, Path(image_path).stem])

        concept_row = next((self._concept_index[key] for key in keys if key in self._concept_index), None)
        if concept_row is None:
            return zeros, torch.tensor(False), torch.tensor(0.0), ""

        valid = bool(int(float(concept_row.get("weak_label_valid", 0) or 0)))
        confidence = float(concept_row.get("concept_confidence", 0.0) or 0.0)
        values = torch.tensor(
            [float(concept_row.get(code, 0.0) or 0.0) for code in _CONCEPT_CODES],
            dtype=torch.float32,
        )
        return (
            values,
            torch.tensor(valid),
            torch.tensor(confidence, dtype=torch.float32),
            str(concept_row.get("concept_source", "")),
        )

    def _base_record(
        self,
        row: pd.Series,
        *,
        image: Any,
        image_path: Path | str,
        domain: str,
        seg_mask: torch.Tensor,
        seg_mask_valid: bool,
    ) -> dict[str, Any]:
        concept_labels, concept_valid, concept_confidence, concept_source = self._load_concepts(row)
        return {
            "image": image,
            "label": int(row["label"]),
            "image_path": str(image_path),
            "split": str(row["split"]),
            "domain": domain,
            "seg_mask": seg_mask,
            "seg_mask_valid": seg_mask_valid,
            "concept_labels": concept_labels,
            "concept_valid": concept_valid,
            "concept_confidence": concept_confidence,
            "concept_source": concept_source,
        }

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
        return self._base_record(
            row,
            image=image,
            image_path=image_path,
            domain=domain,
            seg_mask=seg_mask,
            seg_mask_valid=seg_mask_valid,
        )


class SegmentationManifestDataset(ManifestDataset):
    """Manifest dataset that applies synchronized transforms to image and mask."""

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.frame.iloc[index]
        image_path = self.image_root / str(row["image_path"])
        image = Image.open(image_path).convert("RGB")
        domain = str(row["domain"]) if "domain" in self.frame.columns else ""
        seg_mask, seg_mask_valid = self._mask_provider.load(
            str(row["image_path"]), domain, self._seg_mask_size
        )
        # 핵심 차이: 이미지와 마스크를 한 transform에 함께 넘겨 같은 공간 증강을 받게 한다.
        if self.transform is not None:
            image, seg_mask = self.transform(image, seg_mask)
        return self._base_record(
            row,
            image=image,
            image_path=image_path,
            domain=domain,
            seg_mask=seg_mask,
            seg_mask_valid=seg_mask_valid,
        )


class SegmentationFDAManifestDataset(SegmentationManifestDataset):
    """Segmentation dataset with on-the-fly FDA and synchronized mask transforms.

    FDA is photometric/frequency-domain only, so the mask geometry remains valid
    and can still go through the synchronized segmentation transform.
    """

    def __init__(
        self,
        manifest_path: str | Path,
        image_root: str | Path | None = None,
        split: str | None = None,
        transform: Callable[[Image.Image, torch.Tensor], tuple[Any, torch.Tensor]] | None = None,
        fda_alpha: float = 0.05,
        fda_probability: float = 1.0,
        ampmix_mode: bool = False,
        ampmix_alpha_low: float = 0.0,
        ampmix_alpha_high: float = 0.5,
        fda_target_domain: str | None = None,
        fda_apply_to_target_domain: bool = False,
        domain_column: str = "domain",
        seg_mask_dir: str | Path | None = None,
        seg_mask_size: int = 512,
        mask_provider: LesionMaskProvider | None = None,
        concept_label_path: str | Path | None = None,
    ) -> None:
        super().__init__(
            manifest_path,
            image_root,
            split,
            transform,
            seg_mask_dir=seg_mask_dir,
            seg_mask_size=seg_mask_size,
            mask_provider=mask_provider,
            concept_label_path=concept_label_path,
        )
        self._alpha = fda_alpha
        self._probability = fda_probability
        self._ampmix_mode = ampmix_mode
        self._ampmix_alpha_low = ampmix_alpha_low
        self._ampmix_alpha_high = ampmix_alpha_high
        self._target_domain = fda_target_domain
        self._apply_to_target_domain = fda_apply_to_target_domain
        self._domain_column = domain_column
        self._rng = np.random.default_rng()
        self._domain_indices: dict[str, list[int]] = {}
        self.rebuild_domain_indices()

    def rebuild_domain_indices(self) -> None:
        self._domain_indices = {}
        if self._domain_column not in self.frame.columns:
            return
        for iloc_pos in range(len(self.frame)):
            domain = str(self.frame.iloc[iloc_pos][self._domain_column])
            self._domain_indices.setdefault(domain, []).append(iloc_pos)

    def _should_mix(self, source_domain: str) -> bool:
        if self._probability <= 0:
            return False
        if (
            self._target_domain
            and source_domain == self._target_domain
            and not self._apply_to_target_domain
        ):
            return False
        return bool(self._rng.random() <= self._probability)

    def _sample_ref_index(self, source_index: int, source_domain: str) -> int:
        if self._target_domain and self._target_domain in self._domain_indices:
            candidates = [
                i for i in self._domain_indices[self._target_domain] if i != source_index
            ]
            if candidates:
                return int(self._rng.choice(candidates))

        if len(self._domain_indices) > 1:
            other_domains = [d for d in self._domain_indices if d != source_domain]
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
        image_path = self.image_root / str(row["image_path"])
        domain = str(row["domain"]) if "domain" in self.frame.columns else ""

        with Image.open(image_path) as img:
            image = img.convert("RGB")
            if self._should_mix(domain):
                ref_arr = self._load_raw_array(self._sample_ref_index(index, domain))
                source_arr = np.asarray(image)
                if self._ampmix_mode:
                    mixed = ampmix(
                        source_arr,
                        ref_arr,
                        alpha_low=self._ampmix_alpha_low,
                        alpha_high=self._ampmix_alpha_high,
                        rng=self._rng,
                    )
                else:
                    mixed = fda_mix(source_arr, ref_arr, self._alpha)
                image = Image.fromarray(mixed)

        seg_mask, seg_mask_valid = self._mask_provider.load(
            str(row["image_path"]), domain, self._seg_mask_size
        )
        if self.transform is not None:
            image, seg_mask = self.transform(image, seg_mask)
        return self._base_record(
            row,
            image=image,
            image_path=image_path,
            domain=domain,
            seg_mask=seg_mask,
            seg_mask_valid=seg_mask_valid,
        )


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
        fda_probability: float = 1.0,
        ampmix_mode: bool = False,
        ampmix_alpha_low: float = 0.0,
        ampmix_alpha_high: float = 0.5,
        domain_column: str = "domain",
        seg_mask_dir: str | Path | None = None,
        seg_mask_size: int = 512,
        mask_provider: LesionMaskProvider | None = None,
        concept_label_path: str | Path | None = None,
    ) -> None:
        super().__init__(
            manifest_path, image_root, split, transform,
            seg_mask_dir=seg_mask_dir, seg_mask_size=seg_mask_size,
            mask_provider=mask_provider, concept_label_path=concept_label_path,
        )
        self._alpha = fda_alpha
        self._probability = fda_probability
        self._ampmix_mode = ampmix_mode
        self._ampmix_alpha_low = ampmix_alpha_low
        self._ampmix_alpha_high = ampmix_alpha_high
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
        if self._probability > 0.0 and self._rng.random() <= self._probability:
            ref_arr = self._load_raw_array(self._sample_ref_index(index))
            if self._ampmix_mode:
                mixed = ampmix(
                    src_arr,
                    ref_arr,
                    alpha_low=self._ampmix_alpha_low,
                    alpha_high=self._ampmix_alpha_high,
                    rng=self._rng,
                )
            else:
                mixed = fda_mix(src_arr, ref_arr, self._alpha)
            image = Image.fromarray(mixed)
        else:
            image = Image.fromarray(src_arr)

        if self.transform is not None:
            image = self.transform(image)

        domain = str(row["domain"]) if "domain" in self.frame.columns else ""
        seg_mask, seg_mask_valid = self._mask_provider.load(
            str(row["image_path"]), domain, self._seg_mask_size
        )
        return self._base_record(
            row,
            image=image,
            image_path=self.image_root / str(row["image_path"]),
            domain=domain,
            seg_mask=seg_mask,
            seg_mask_valid=seg_mask_valid,
        )
