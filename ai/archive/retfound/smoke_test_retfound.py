import torch
from drscreen.models.build import build_model, load_retfound_backbone

model = build_model("retfound", pretrained=False, num_outputs=1)
print("model built:", type(model).__name__)
print("head:", model.head)

missing, unexpected = load_retfound_backbone(model, "artifacts/retfound/RETFound_mae_natureCFP.pth")
print("missing keys:", len(missing), "(expected: 2 — head.weight, head.bias)")
print("unexpected keys:", len(unexpected), "(expected: 0)")

x = torch.randn(1, 3, 224, 224)
with torch.no_grad():
    out = model(x)
print("output shape:", out.shape, "(expected: torch.Size([1, 1]))")
