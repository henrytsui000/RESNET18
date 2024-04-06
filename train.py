import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
import argparse


def load_data():
    """
    Load CIFAR-10 dataset and define data loaders.
    """
    # Define transforms for data preprocessing
    transform_train = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    transform_test = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    # Load CIFAR-10 dataset
    trainset = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform_train
    )
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=128, shuffle=True, num_workers=16
    )

    testset = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transform_test
    )
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=128, shuffle=False, num_workers=16
    )

    return trainloader, testloader


def train_model(model, trainloader, criterion, optimizer, device, epochs=10):
    """
    Train the model.
    """
    model.to(device)
    model.train()

    for epoch in range(epochs):  # Train for specified number of epochs
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data[0].to(device), data[1].to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 100 == 99:  # Print every 100 mini-batches
                print("[%d, %5d] loss: %.3f" % (epoch + 1, i + 1, running_loss / 100))
                running_loss = 0.0

    print("Finished Training")


def evaluate_model(model, testloader, device):
    """
    Evaluate the model.
    """
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print("Accuracy of the network on the 10000 test images: %d %%" % accuracy)


def main(model_type):
    """
    Main function to load data, define model, train, and evaluate.
    """
    # Load data
    trainloader, testloader = load_data()

    # Choose model
    if model_type == "normal":
        from normal18 import ResNet18
    elif model_type == "modified":
        from modified18 import ResNet18
    else:
        raise ValueError("Invalid model_type. Choose 'normal' or 'modified'.")

    # Define model
    resnet18 = ResNet18()
    num_ftrs = resnet18.fc.in_features
    resnet18.fc = nn.Linear(num_ftrs, 10)  # 10 classes in CIFAR-10

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        resnet18.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4
    )

    # Train the model
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    train_model(resnet18, trainloader, criterion, optimizer, device)

    # Evaluate the model
    evaluate_model(resnet18, testloader, device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Choose the model type: 'normal' or 'modified'."
    )
    parser.add_argument(
        "model_type", type=str, choices=["normal", "modified"], help="Model type to use"
    )
    args = parser.parse_args()

    main(args.model_type)
