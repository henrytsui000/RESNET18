import pytest
import torch
from torchsummary import summary
import sys

sys.path.append("./")
from model import load_model

@pytest.mark.parametrize("model_type", ["ResNet18", "ResNet34"])
def test_load_model(model_type):
    model = load_model(model_type)
    assert isinstance(model, torch.nn.Module)

@pytest.mark.parametrize("model_type, modified, nc, expected_shape", [
    ("ResNet18", False, 10, (1, 10)),
    ("ResNet18", True, 10, (1, 10)),
    ("ResNet34", False, 100, (1, 100)),
    ("ResNet34", True, 100, (1, 100)),
])
def test_forward_pass(model_type, modified, nc, expected_shape):
    model = load_model(model_type, modified, nc)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    inputs = torch.randn(1, 3, 32, 32).to(device)
    outputs = model(inputs)
    assert outputs.shape == expected_shape

@pytest.mark.parametrize("model_type", ["ResNet18", "ResNet34"])
def test_summary_model(model_type):
    model = load_model(model_type)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    summary(model, (3, 32, 32))
