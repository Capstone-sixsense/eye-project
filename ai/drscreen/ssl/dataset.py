from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


class SSLManifestDataset(Dataset):
    """Returns two augmented views of each fundus image for contrastive SSL.

    Unlike ManifestDataset, all rows in the manifest are loaded regardless of
    split. SSL pretraining is unsupervised, so Messidor external_test images
    are included alongside APTOS and IDRiD train/val/test images. Labels are
    not used.
    """

    def __init__(
        self,
        manifest_path: str | Path,
        image_root: str | Path | None = None,
        transform: Callable[[Image.Image], tuple[Any, Any]] | None = None,
    ) -> None:
        self.manifest_path = Path(manifest_path)
        self.image_root = Path(image_root) if image_root else self.manifest_path.parent
        self.transform = transform

        frame = pd.read_csv(self.manifest_path)
        if "image_path" not in frame.columns:
            raise ValueError("Manifest is missing required column: image_path")
        self.frame = frame.reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.frame.iloc[index]
        image_path = self.image_root / str(row["image_path"])
        image = Image.open(image_path).convert("RGB")
        if self.transform is not None:
            view1, view2 = self.transform(image)
        else:
            import torchvision.transforms.functional as TF
            view1 = view2 = TF.to_tensor(image)
        return {"view1": view1, "view2": view2}
