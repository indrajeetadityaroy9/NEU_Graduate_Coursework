import torch
import torch.nn as nn
import torchmetrics
import pytorch_lightning as pl
import argparse
import sys
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import torch.optim as optim
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping
from torch.optim.lr_scheduler import LambdaLR


def load_dataset(dataset_name, batch_size):
    if dataset_name == 'mnist':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
        input_channels = 1  # Grayscale

    elif dataset_name == 'cifar':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
        input_channels = 3  # RGB

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)
    num_classes = len(train_dataset.classes)
    return train_loader, test_loader, input_channels, num_classes


def conv_block(in_channels, out_channels, kernel_size=3, stride=1, padding=1):
    """Helper function to create a convolutional block with Conv2D, BatchNorm, ReLU, and MaxPooling."""
    layers = [
        nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2)  # Reduces spatial dimension by half
    ]
    return nn.Sequential(*layers)


class CNNClassifier(pl.LightningModule):
    def __init__(self, in_channels, num_classes, warmup_epochs=5):
        super(CNNClassifier, self).__init__()
        self.warmup_epochs = warmup_epochs

        # Model definition (same as before)
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )
        self.conv2 = conv_block(32, 64)
        self.res1_conv1 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.res1_bn1 = nn.BatchNorm2d(64)
        self.res1_conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.res1_bn2 = nn.BatchNorm2d(64)
        self.conv3 = conv_block(64, 128)
        self.conv4 = conv_block(128, 256)
        self.res2_conv1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.res2_bn1 = nn.BatchNorm2d(256)
        self.res2_conv2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.res2_bn2 = nn.BatchNorm2d(256)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(p=0.8),
            nn.Linear(256, num_classes)
        )
        self.accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        res1 = out
        out = self.res1_conv1(out)
        out = self.res1_bn1(out)
        out = nn.ReLU(inplace=True)(out)
        out = self.res1_conv2(out)
        out = self.res1_bn2(out)
        out += res1
        out = nn.ReLU(inplace=True)(out)
        out = self.conv3(out)
        out = self.conv4(out)
        res2 = out
        out = self.res2_conv1(out)
        out = self.res2_bn1(out)
        out = nn.ReLU(inplace=True)(out)
        out = self.res2_conv2(out)
        out = self.res2_bn2(out)
        out += res2
        out = nn.ReLU(inplace=True)(out)
        out = self.classifier(out)
        return out

    def training_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(inputs)
        loss = nn.CrossEntropyLoss(label_smoothing=0.1)(outputs, labels)
        acc = self.accuracy(outputs, labels)
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_acc', acc, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(inputs)
        loss = nn.CrossEntropyLoss(label_smoothing=0.1)(outputs, labels)
        acc = self.accuracy(outputs, labels)
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-3, weight_decay=5e-4)

        # Warmup for 5 epochs followed by Cosine Annealing
        warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=lambda epoch: (epoch + 1) / 5 if epoch < 5 else 1.0
        )
        cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)

        return [optimizer], [
            {"scheduler": warmup_scheduler, "interval": "epoch", "frequency": 1},
            {"scheduler": cosine_scheduler, "interval": "epoch", "frequency": 1}
        ]


def main():

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

        dataset = 'mnist' if args.mnist else 'cifar'
        print(f"Training the model on {dataset} dataset")

        batch_size = 256
        train_loader, val_loader, input_channels, num_classes = load_dataset(dataset, batch_size)

        # Early Stopping Callback
        early_stopping_callback = pl.callbacks.EarlyStopping(monitor='val_loss', patience=10, verbose=True)

        # Initialize the Trainer with EarlyStopping
        trainer = pl.Trainer(max_epochs=50, callbacks=[early_stopping_callback], accumulate_grad_batches=1)

        # Initialize the model with correct input channels and number of classes
        model = CNNClassifier(in_channels=input_channels, num_classes=num_classes)

        # Train the model
        trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)


if __name__ == "__main__":
    main()
