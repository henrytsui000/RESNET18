import argparse
import torch
import torch.nn as nn
from loguru import logger
from typing import Any, Tuple, Iterator
from dataset import load_dataset
from model import load_model


def evaluate_model(
    model: nn.Module,
    testloader: Iterator[Tuple[torch.Tensor, torch.Tensor]],
    device: torch.device,
) -> float:
    """
    Evaluate the model on the test data and calculate the accuracy.

    Args:
        model (nn.Module): The trained model.
        testloader (Iterator[Tuple[torch.Tensor, torch.Tensor]]): The data loader for test data.
        device (torch.device): The device to run the evaluation on (e.g., 'cuda' or 'cpu').

    Returns:
        float: The accuracy of the model on the test data.
    """
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    return accuracy


def add_arguments(parser: argparse.ArgumentParser):
    """
    Add command-line arguments to the parser.
    """
    parser.add_argument(
        "model_type", type=str, choices=["normal", "modified"], help="Model type to use"
    )
    parser.add_argument(
        "--dataset", type=str, choices=["cifar10", "cifar100"], default="cifar10"
    )
    parser.add_argument(
        "--model_path", type=str
    )
    parser.add_argument("--device", type=str, default="cuda")

    return parser


def main(args: argparse.Namespace):
    """
    Main function to load data, define model, train, and evaluate.
    """
    (_, testloader), nc = load_dataset(args.dataset)
    if args.model_path:
        model = torch.load(args.model_path)
    else:
        model = load_model(args.model_type, nc)

    # Train the model
    device = torch.device(args.device)
    model = model.to(device)
    accuracy = evaluate_model(model, testloader, device)
    logger.info(f"Accuracy on test set: {accuracy:.2f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Choose the model type: 'normal' or 'modified'."
    )
    add_arguments(parser)
    args = parser.parse_args()

    logger.add("output.log", format="{time} {level} {message}", level="INFO")

    main(args)
