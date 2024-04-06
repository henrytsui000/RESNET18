import torch
import sys

sys.path.append("./")
from dataset.cifar100 import load_data


def test_load_data():
    """
    Test if data loaders are correctly loaded for CIFAR-100 dataset.
    """
    # Load data
    trainloader, testloader = load_data()

    # Check if trainloader and testloader are instances of DataLoader
    assert isinstance(trainloader, torch.utils.data.DataLoader)
    assert isinstance(testloader, torch.utils.data.DataLoader)

    # Check if the sizes of trainset and testset are correct for CIFAR-100 dataset
    assert len(trainloader.dataset) == 50000  # CIFAR-100 train dataset size
    assert len(testloader.dataset) == 10000  # CIFAR-100 test dataset size

    # Check if data transformation is applied
    assert hasattr(trainloader.dataset, "transform") and hasattr(
        testloader.dataset, "transform"
    )
