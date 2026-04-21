import os
import sys
import torch
import numpy as np
from PIL import Image

# EDSR 소스 코드 경로를 시스템 경로에 추가 (backend/edsr 폴더가 있다고 가정)
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BACKEND_DIR = os.path.dirname(CURRENT_DIR)

EDSR_SOURCE_DIR = os.path.join(BACKEND_DIR, "edsr", "EDSR-PyTorch", "src")
sys.path.append(EDSR_SOURCE_DIR)

try:
    from model import edsr
except ImportError:
    raise ImportError(f"EDSR 소스 코드를 {EDSR_SOURCE_DIR}에서 찾을 수 없습니다.")

class EDSRWrapper:
    def __init__(self, model_name="edsr_baseline_x2-1bc95232.pt", scale=2):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.scale = scale
        
        # 같은 폴더 내의 가중치 파일 경로 설정
        self.weights_path = os.path.join(CURRENT_DIR, model_name)
        
        # 모델 초기화 및 로드
        self.model = self._load_model()
        print(f"✅ EDSR 모델 로드 완료 (Device: {self.device}, Weight: {model_name})")

    def _load_model(self):
        """EDSR-PyTorch의 인자값을 모방하여 모델을 생성하고 가중치를 로드합니다."""
        # EDSR 모델 생성에 필요한 최소 인자값 설정 (기본 EDSR 기준)
        class DummyArgs:
            def __init__(self, scale):
                self.n_resblocks = 16 # cpu 부하를 고려해서 수치 조정
                self.n_feats = 64 # cpu 부하를 고려해서 수치 조정
                self.scale = [scale]
                self.rgb_range = 255
                self.n_colors = 3
                self.res_scale = 1.0 # x4는 0.1
                self.precision = 'single'

        args = DummyArgs(self.scale)
        model = edsr.EDSR(args).to(self.device)
        
        # 가중치 파일 로드
        if not os.path.exists(self.weights_path):
            raise FileNotFoundError(f"가중치 파일이 없습니다: {self.weights_path}")
            
        checkpoint = torch.load(self.weights_path, map_location=self.device, weights_only=True)
        model.load_state_dict(checkpoint, strict=True)
        model.eval()
        return model

    @torch.no_grad()
    def upscale(self, pil_image: Image.Image) -> Image.Image:
        """PIL 이미지를 입력받아 고해상도 PIL 이미지를 반환합니다."""
        # 1. 전처리: PIL -> Numpy -> Tensor
        img_np = np.array(pil_image)
        # RGB 순서 확인 및 채널 위치 변경 (H, W, C -> C, H, W)
        img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).float().to(self.device)
        # 배치 차원 추가 및 범위 조정 (0-255)
        img_tensor = img_tensor.unsqueeze(0)

        # 2. 추론
        sr_tensor = self.model(img_tensor)

        # 3. 후처리: Tensor -> Numpy -> PIL
        sr_img = sr_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
        sr_img = np.clip(sr_img, 0, 255).astype(np.uint8)
        
        return Image.fromarray(sr_img)

# 테스트용 실행 코드
if __name__ == "__main__":
    # wrapper = EDSRWrapper()
    # input_img = Image.open("test.jpg")
    # output_img = wrapper.upscale(input_img)
    # output_img.save("upscaled.jpg")
    pass