import torch
from torchsummary import summary
import sys

sys.path.append("./")
from model.normal18 import ResNet18


def test_forward_pass():
    model = ResNet18()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    inputs = torch.randn(1, 3, 32, 32).to(device)
    outputs = model(inputs)
    assert outputs.shape == (1, 10)  # Assuming 10 classes


def test_summary_model():
    # Create an instance of the model
    model = ResNet18()

    # Move the model to the appropriate device (CPU or GPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Print the summary of the model
    summary(model, (3, 32, 32))  # Input size is (channels, height, width)
