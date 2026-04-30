"""
QuickQual Wrapper
- 안저 이미지 표준 전처리 (검은 테두리 제거 + square + resize)
- 이미지 품질 평가 (DenseNet121 features + sklearn SVM → Good/Usable/Bad)

Reference: https://github.com/justinengelmann/QuickQual
"""
from __future__ import annotations

import os
import joblib
import numpy as np
import torch
import timm
from PIL import Image
from torchvision.transforms import functional as TF

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))


# ---------------------------------------------------------------
# 표준 전처리 (QuickQual repo의 image_preprocessing.py 로직과 동일)
# ---------------------------------------------------------------
def preprocess_fundus_image(img: Image.Image, threshold: int = 15) -> Image.Image:
    """
    Step 1: 검은 테두리 제거
    Step 2: 정사각형으로 패딩
    Step 3: 1024x1024 리사이즈
    """
    img = img.convert("RGB")
    arr = np.array(img)
    mean = arr.mean(-1)

    rows = np.where(mean > threshold)[0]
    cols = np.where(mean > threshold)[1]
    if rows.size == 0 or cols.size == 0:
        # 전체가 거의 검은 이미지일 때 fallback
        return img.resize((1024, 1024), Image.LANCZOS)

    top, bottom = int(rows.min()), int(rows.max())
    left, right = int(cols.min()), int(cols.max())

    buffer = 20
    left = max(0, left - buffer)
    right = min(arr.shape[1], right + buffer)
    top = max(0, top - buffer)
    bottom = min(arr.shape[0], bottom + buffer)

    img = img.crop((left, top, right, bottom))

    width, height = img.size
    if width > height:
        pad = width - height
        padding = [0, pad // 2, 0, pad - pad // 2]
    else:
        pad = height - width
        padding = [pad // 2, 0, pad - pad // 2, 0]
    img = TF.pad(img, padding)
    return img.resize((1024, 1024), Image.LANCZOS)


# ---------------------------------------------------------------
# QuickQual: DenseNet121 features + SVM classifier
# ---------------------------------------------------------------
class QuickQualWrapper:
    QUALITY_LABELS = ("good", "usable", "bad")

    def __init__(
        self,
        svm_filename: str = "quickqual_dn121_512.pkl",
        feature_input_size: int = 512,
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.feature_input_size = feature_input_size

        self.svm_path = (
            svm_filename
            if os.path.isabs(svm_filename)
            else os.path.join(CURRENT_DIR, svm_filename)
        )
        if not os.path.exists(self.svm_path):
            raise FileNotFoundError(
                f"QuickQual SVM 가중치를 찾을 수 없습니다: {self.svm_path}\n"
                "https://github.com/justinengelmann/QuickQual/releases 에서 다운로드하세요."
            )

        # DenseNet121 (timm) - num_classes=0 → feature extractor
        self.backbone = timm.create_model(
            "densenet121.tv_in1k", pretrained=True, num_classes=0
        ).to(self.device).eval()

        # sklearn SVM
        self.clf = joblib.load(self.svm_path)

        print(
            f"✅ QuickQual 로드 완료 (device={self.device}, "
            f"svm={os.path.basename(self.svm_path)})",
            flush=True,
        )

    @torch.no_grad()
    def predict_quality(self, pil_image: Image.Image) -> dict:
        """품질 확률 반환: {'good': p, 'usable': p, 'bad': p, 'label': str}"""
        img = pil_image.convert("RGB")
        # QuickQual 표준: 짧은 변 기준 resize + [0.5,0.5,0.5] 정규화
        x = TF.to_tensor(TF.resize(img, self.feature_input_size))
        x = TF.normalize(x, [0.5] * 3, [0.5] * 3).unsqueeze(0).to(self.device)

        feats = self.backbone(x).squeeze().cpu().reshape(1, -1).numpy()
        probs = self.clf.predict_proba(feats)[0]  # [good, usable, bad]

        return {
            "good": float(probs[0]),
            "usable": float(probs[1]),
            "bad": float(probs[2]),
            "label": self.QUALITY_LABELS[int(np.argmax(probs))],
        }

    def preprocess_and_score(
        self, pil_image: Image.Image
    ) -> tuple[Image.Image, dict]:
        """전처리(1024×1024)된 이미지와 품질 점수를 함께 반환."""
        preprocessed = preprocess_fundus_image(pil_image)
        quality = self.predict_quality(preprocessed)
        return preprocessed, quality