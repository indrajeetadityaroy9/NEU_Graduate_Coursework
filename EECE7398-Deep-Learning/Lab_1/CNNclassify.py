import argparse
import torch.optim as optim
import torch.nn as nn
from models.model import CNNClassifier
from utility.data_loader import  load_dataset
from utility.model_trainer import train_model
from torch.optim.lr_scheduler import StepLR

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train CNN on different datasets")
    parser.add_argument('command', choices=['train'], help="Command to execute")
    parser.add_argument('--dataset', choices=['mnist', 'cifar'], required=True, help="Dataset to use: mnist or cifar")
    parser.add_argument('--epochs', type=int, default=500, help="Number of training epochs")
    parser.add_argument('--batch_size', type=int, default=64, help="Batch size")
    parser.add_argument('--log_interval', type=int, default=50, help="Interval to log training/testing status")

    args = parser.parse_args()
    # Load dataset
    train_loader, test_loader, input_channels, num_classes = load_dataset(args.dataset, args.batch_size)

    # Initialize model
    model = CNNClassifier(in_channels=input_channels, num_classes=num_classes)

    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # Define the learning rate scheduler
    scheduler = StepLR(optimizer, step_size=15, gamma=0.5)

    # Call the training function
    if args.command == 'train':
        train_model(model, train_loader, test_loader, criterion, optimizer, scheduler, epochs=args.epochs)