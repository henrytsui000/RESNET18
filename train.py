import argparse
import torch
from torch.nn import CrossEntropyLoss
import torch.optim as optim
from loguru import logger

from utils.trainer import Trainer
from dataset import load_dataset
from model import load_model
from test import evaluate_model


def train_model(
    model, trainloader, testloader, criterion, optimizer, device, epochs=10, patience=3
):
    """
    Train the model.
    """
    trainer = Trainer(model, criterion, optimizer, device, patience=patience)

    best_val_acc = float('-inf')

        
    for epoch in range(epochs):  # Train for specified number of epochs
        train_loss = trainer.train_one_epoch(trainloader, epoch + 1)
        
        val_acc = evaluate_model(model, testloader, device)
        logger.info(f"Epoch {epoch+1} Train Loss: {train_loss:.4f} | Val Acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            logger.info("Saving model...")
            trainer.save_model("best_model.pt")
        if trainer.early_stop(val_acc):
            logger.info(f"Validation loss hasn't improved for {patience} epochs. Stopping training.")
            break

    logger.info("Finished Training")


def add_arguments(parser: argparse.ArgumentParser):
    """
    Add command-line arguments to the parser.
    """
    parser.add_argument(
        "model_type", type=str, choices=["ResNet18", "ResNet34"], help="Model type to use"
    )
    parser.add_argument(
        "--dataset", type=str, choices=["cifar10", "cifar100"], default="cifar10"
    )
    parser.add_argument(
        "-m", "--modified", action='store_true', default=False
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("-e", "--epochs", type=int, default=10)
    parser.add_argument("--patience", type=int, default=5)

def main(args):
    """
    Main function to load data, define model, train, and evaluate.
    """
    (trainloader, testloader), nc = load_dataset(args.dataset)
    model = load_model(args.model_type, modified=args.modified, nc = nc)

    # Define loss function and optimizer
    criterion = CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)

    # Train the model
    device = torch.device(args.device)
    model = model.to(device)
    train_model(
        model, trainloader, testloader, criterion, optimizer, device, epochs=args.epochs
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Choose the model type: 'normal' or 'modified'.")
    add_arguments(parser)
    args = parser.parse_args()

    logger.add("output.log", format="{time} {level} {message}", level="INFO")

    main(args)
