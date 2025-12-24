import torch
from train_model import SimpleCNN

model = SimpleCNN()
model.load_state_dict(torch.load("model.pth"))
model.eval()

dummy_input = torch.randn(1, 1, 28, 28)

torch.onnx.export(
    model,
    dummy_input,
    "model.onnx",
    input_names=["input"],
    output_names=["output"]
)

print("ONNX model exported")
