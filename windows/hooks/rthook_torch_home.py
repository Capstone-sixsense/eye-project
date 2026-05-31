"""
Runtime hook: PyInstaller 실행 시 TORCH_HOME을 번들 내부로 설정.

QuickQualWrapper가 timm.create_model("densenet121.tv_in1k", pretrained=True)를
호출하면 torchvision/timm이 TORCH_HOME/hub/checkpoints/ 에서 가중치를 찾는다.
TORCH_HOME이 없으면 인터넷에서 다운로드를 시도하므로,
번들에 포함된 가중치 파일을 가리키도록 사전에 설정한다.
"""
import os
import sys
from pathlib import Path

if getattr(sys, "frozen", False):
    bundle_dir = Path(getattr(sys, "_MEIPASS", Path(sys.executable).parent))
    os.environ.setdefault("TORCH_HOME", str(bundle_dir / ".cache" / "torch"))
