import argparse
import sys
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import torch.optim as optim
import torch.nn as nn

import os
import torch
from tqdm import tqdm


def conv_block(in_channels, out_channels, kernel_size=3, stride=1, padding=1):
    layers = [
        nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2)
    ]
    return nn.Sequential(*layers)


class CNNClassifier(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()

        # First CONV layer: filter size = 5x5, stride = 1, 32 filters
        self.conv1 = conv_block(in_channels, 32, kernel_size=5, stride=1, padding=2)

        # The rest of the layers use the default kernel size of 3x3, stride 1
        self.conv2 = conv_block(32, 64)

        # First Residual Block
        self.res1_conv1 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.res1_conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)

        self.conv3 = conv_block(64, 128)
        self.conv4 = conv_block(128, 256)

        # Second Residual Block
        self.res2_conv1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.res2_conv2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)

        # Global average pooling and classifier
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(p=0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)

        # First Residual Block
        res1 = out
        res1_out = self.res1_conv1(out)
        res1_out = self.res1_conv2(res1_out)
        out = nn.ReLU(inplace=True)(res1_out + res1)  # Skip connection + ReLU after addition

        out = self.conv3(out)
        out = self.conv4(out)

        # Second Residual Block
        res2 = out
        res2_out = self.res2_conv1(out)
        res2_out = self.res2_conv2(res2_out)
        out = nn.ReLU(inplace=True)(res2_out + res2)  # Skip connection + ReLU after addition

        # Classifier
        out = self.classifier(out)
        return out


def train_model(model, train_loader, test_loader, criterion, optimizer, scheduler, device, epochs=500):
    best_acc = 0.0  # Track the best test accuracy

    for epoch in range(1, epochs + 1):
        print(f"Epoch {epoch}/{epochs}")

        # Training Phase
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        train_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc="Training", leave=False)

        for batch_idx, (inputs, labels) in train_bar:
            # Move inputs and labels to the specified device
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # Log training loss and accuracy
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            # Update the tqdm progress bar
            train_bar.set_postfix({
                'Loss': running_loss / (batch_idx + 1),
                'Accuracy': 100. * correct / total
            })

        train_loss = running_loss / len(train_loader)
        train_acc = 100. * correct / total

        # Testing Phase
        model.eval()
        test_loss = 0.0
        correct = 0
        total = 0

        test_bar = tqdm(enumerate(test_loader), total=len(test_loader), desc="Testing", leave=False)

        with torch.no_grad():
            for batch_idx, (inputs, labels) in test_bar:
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)
                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

                test_bar.set_postfix({
                    'Loss': test_loss / (batch_idx + 1),
                    'Accuracy': 100. * correct / total
                })

        test_loss /= len(test_loader)
        test_acc = 100. * correct / total

        print(f"{epoch}/{epochs:<5} | Train Loss: {train_loss:<10.4f} | Train Acc: {train_acc:<12.4f} | "
              f"Test Loss: {test_loss:<10.4f} | Test Acc: {test_acc:<10.4f}")

        scheduler.step()

        # Save the best model based on test accuracy
        if test_acc > best_acc:
            best_acc = test_acc
            # Ensure the "model" directory exists before saving
            model_dir = "model"
            if not os.path.exists(model_dir):
                os.makedirs(model_dir)

            # Save the model in the "model" folder
            model_path = os.path.join(model_dir, f"best_model_epoch_{epoch}.pth")
            torch.save(model.state_dict(), model_path)
            print(f"Best model saved with accuracy {best_acc:.2f}% at epoch {epoch}.")


def load_dataset(dataset_name, batch_size):
    # Define the transformations for the datasets
    if dataset_name == 'mnist':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

        # Download and load MNIST dataset
        train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
        input_channels = 1  # Grayscale

    elif dataset_name == 'cifar':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        # Download and load CIFAR-10 dataset
        train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
        input_channels = 3  # RGB

    else:
        raise ValueError(f"Dataset {dataset_name} is not supported. Please choose 'mnist' or 'cifar'.")

    # Create DataLoader for train and test datasets
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Number of classes in the dataset
    num_classes = len(train_dataset.classes)

    return train_loader, test_loader, input_channels, num_classes


def main():
    import argparse
    import sys

    description = """
          ______ .__   __. .__   __.      ______  __          ___           _______.     _______. __   _______  __   _______ .______         
         /      ||  \ |  | |  \ |  |     /      ||  |        /   \         /       |    /       ||  | |   ____||  | |   ____||   _  \        
        |  ,----'|   \|  | |   \|  |    |  ,----'|  |       /  ^  \       |   (----`   |   (----`|  | |  |__   |  | |  |__   |  |_)  |       
        |  |     |  . `  | |  . `  |    |  |     |  |      /  /_\  \       \   \        \   \    |  | |   __|  |  | |   __|  |      /        
        |  `----.|  |\   | |  |\   |    |  `----.|  `----./  _____  \  .----)   |   .----)   |   |  | |  |     |  | |  |____ |  |\  \----.   
         \______||__| \__| |__| \__|     \______||_______/__/     \__\ |_______/    |_______/    |__| |__|     |__| |_______|| _| `._____|   
    """

    epilog = """Usage examples:
      python3 CNNclassify.py train --mnist       Train the model using the MNIST dataset
      python3 CNNclassify.py train --cifar       Train the model using the CIFAR-10 dataset
      python3 CNNclassify.py test car.png        Test the model using 'car.png'
    """

    parser = argparse.ArgumentParser(
        usage='python3 main.py [-h] {train,test} ...',
        formatter_class=argparse.RawTextHelpFormatter,
        add_help=False,
        epilog=epilog
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to execute: train or test")
    train_parser = subparsers.add_parser('train', help="Train the model")
    train_parser.add_argument('--mnist', action='store_true', help="Use the MNIST dataset for training")
    train_parser.add_argument('--cifar', action='store_true', help="Use the CIFAR-10 dataset for training")
    test_parser = subparsers.add_parser('test', help="Test the model")
    test_parser.add_argument('image_file', nargs='*', help="Image file(s) for testing (e.g., car.png)")

    args = parser.parse_args()

    if len(sys.argv) == 1 or args.command is None:
        print(description)
        parser.print_help()
        sys.exit(1)

    if args.command == 'train':
        if not args.mnist and not args.cifar:
            print("Error: 'train' command requires either --mnist or --cifar argument.", file=sys.stderr)
            print("Use --help for more information.")
            sys.exit(1)

        # Choose dataset
        dataset = 'mnist' if args.mnist else 'cifar'
        print(f"Training the model on {dataset} dataset")
        # Set batch size dynamically
        batch_size = 64  # You can change this as needed or pass it as an argument
        train_loader, test_loader, input_channels, num_classes = load_dataset(dataset, batch_size)
        # Initialize model
        model = CNNClassifier(in_channels=input_channels, num_classes=num_classes)
        # Device setup: use MPS if available, otherwise fall back to CUDA or CPU
        device = torch.device("mps" if torch.backends.mps.is_available() else
                              "cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        # Loss function and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        # Learning rate scheduler
        scheduler = StepLR(optimizer, step_size=15, gamma=0.5)
        # Call the training function
        train_model(
            model=model,
            train_loader=train_loader,
            test_loader=test_loader,
            criterion=nn.CrossEntropyLoss(),
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            epochs=50,
        )
    elif args.command == 'test':
        if not args.image_file:
            print("Error: 'test' command requires at least one image file.", file=sys.stderr)
            print("Use --help for more information.")
            sys.exit(1)

        for img_file in args.image_file:
            print(f"Predicting the class for image: {img_file}")
            # Add your prediction logic here


if __name__ == "__main__":
    main()
