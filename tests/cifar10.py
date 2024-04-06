import torch
import sys

sys.path.append("./")
from dataset.cifar10 import load_data


def test_load_data():
    """
    Test if data loaders are correctly loaded.
    """
    # Load data
    trainloader, testloader = load_data()

    # Check if trainloader and testloader are instances of DataLoader
    assert isinstance(trainloader, torch.utils.data.DataLoader)
    assert isinstance(testloader, torch.utils.data.DataLoader)


def test_dataset_sizes():
    """
    Test if dataset sizes are as expected.
    """
    # Load data
    trainloader, testloader = load_data()

    # Check if the sizes of trainset and testset are correct
    assert len(trainloader.dataset) == 50000  # CIFAR-10 train dataset size
    assert len(testloader.dataset) == 10000  # CIFAR-10 test dataset size
