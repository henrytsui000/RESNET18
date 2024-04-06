import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from loguru import logger
from utils.trainer import Trainer
from typing import Any, Tuple, Iterator
from dataset import load_dataset
from model import load_model


def train_model(
    model, trainloader, testloader, criterion, optimizer, device, epochs=10
):
    """
    Train the model.
    """
    trainer = Trainer(model, criterion, optimizer, device)
    trainer.epochs = epochs

    for epoch in range(epochs):  # Train for specified number of epochs
        trainer.train_one_epoch(trainloader, epoch + 1)
        accuracy = evaluate_model(model, testloader, device)
        logger.info(f"Epoch {epoch+1} Accuracy: {accuracy:.2f}%")

    logger.info("Finished Training")


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


def add_arguments(parser):
    """
    Add command-line arguments to the parser.
    """
    parser.add_argument(
        "model_type", type=str, choices=["normal", "modified"], help="Model type to use"
    )
    parser.add_argument(
        "--dataset", type=str, choices=["cifar10", "cifar100"], default="cifar10"
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("-e", "--epochs", type=int, default=10)

def main(args):
    """
    Main function to load data, define model, train, and evaluate.
    """
    (trainloader, testloader), nc = load_dataset(args.dataset)
    model = load_model(args.model_type, nc)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)

    # Train the model
    device = torch.device(args.device)
    model = model.to(device)
    train_model(
        model, trainloader, testloader, criterion, optimizer, device, epochs=args.epochs
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Choose the model type: 'normal' or 'modified'."
    )
    add_arguments(parser)
    args = parser.parse_args()

    logger.add("output.log", format="{time} {level} {message}", level="INFO")

    main(args)
