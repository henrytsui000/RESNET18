from tqdm import tqdm
from torch import nn
import torch
from typing import Any, Tuple, Iterator


class Trainer:
    def __init__(
        self, model: nn.Module, criterion: Any, optimizer: Any, device: torch.device
    ) -> None:
        """
        Initialize the Trainer class.

        Args:
            model (nn.Module): The neural network model to train.
            criterion: The loss function criterion.
            optimizer: The optimizer used for training.
            device (torch.device): The device to run the training on (e.g., 'cuda' or 'cpu').
        """
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device

    def train_one_iter(self, inputs: torch.Tensor, labels: torch.Tensor) -> float:
        """
        Perform one iteration of training.

        Args:
            inputs (torch.Tensor): The input data tensor.
            labels (torch.Tensor): The target labels tensor.

        Returns:
            float: The loss value for this iteration.
        """
        self.model.train()
        inputs, labels = inputs.to(self.device), labels.to(self.device)

        self.optimizer.zero_grad()
        outputs = self.model(inputs)
        loss = self.criterion(outputs, labels)
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def train_one_epoch(
        self, trainloader: Iterator[Tuple[torch.Tensor, torch.Tensor]], epoch: int
    ) -> None:
        """
        Train the model for one epoch.

        Args:
            trainloader (Iterator[Tuple[torch.Tensor, torch.Tensor]]): The data loader for training.
            epoch (int): The current epoch number.
        """
        running_loss = 0.0
        with tqdm(
            total=len(trainloader), desc=f"Epoch {epoch}", unit="batch", leave=True
        ) as pbar:
            for inputs, labels in trainloader:
                loss = self.train_one_iter(inputs, labels)
                running_loss += loss
                pbar.update(1)
            pbar.set_postfix_str(f"Loss: {running_loss / len(trainloader):.3f}")
