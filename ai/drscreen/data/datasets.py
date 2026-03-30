from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


@dataclass(slots=True)
class FundusSample:
    image_path: Path
    label: int
    split: str


class ManifestDataset(Dataset):
    def __init__(
        self,
        manifest_path: str | Path,
        image_root: str | Path | None = None,
        split: str | None = None,
        transform: Callable[[Image.Image], Any] | None = None,
    ) -> None:
        self.manifest_path = Path(manifest_path)
        self.image_root = Path(image_root) if image_root else self.manifest_path.parent
        self.transform = transform

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
        return {
            "image": image,
            "label": int(row["label"]),
            "image_path": str(image_path),
            "split": str(row["split"]),
        }
