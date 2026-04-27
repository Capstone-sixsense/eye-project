from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


class SSLDataset(Dataset):
    """All images in the manifest regardless of split, returning two augmented views each.

    Labels are ignored. This dataset is used for self-supervised pretraining
    where the goal is domain exposure across APTOS, IDRiD, and Messidor.
    """

    def __init__(
        self,
        manifest_path: str | Path,
        image_root: str | Path,
        transform: Callable[[Image.Image], Any],
    ) -> None:
        self.image_root = Path(image_root)
        self.transform = transform
        frame = pd.read_csv(manifest_path)
        self.paths: list[str] = frame["image_path"].tolist()

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> tuple[Any, Any]:
        image_path = self.image_root / self.paths[index]
        with Image.open(image_path) as img:
            img = img.convert("RGB")
        return self.transform(img), self.transform(img)
