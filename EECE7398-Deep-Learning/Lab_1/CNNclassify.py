import os
import sys
import argparse

import numpy as np
# PyTorch
import torch
import torch.nn as nn
import torch.optim as optim  # Optimizers
import torch.nn.functional as F  # Neural network functions
from torch.utils.data import DataLoader  # Data Loading utilities
from torch.optim.lr_scheduler import ReduceLROnPlateau  # Learning rate schedulers
# torchvision
from torchvision import datasets, transforms  # Datasets and data utilities
# Image processing
from PIL import Image, ImageOps  # Image processing
# Visualization
import matplotlib.pyplot as plt
from tqdm import tqdm


# -------------------- Model Architecture --------------------
class CNNClassifier(nn.Module):
    def __init__(self, in_channels, num_classes, dataset_name):
        super(CNNClassifier, self).__init__()

        if dataset_name == 'mnist':
            self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=5, stride=1, padding=2)
            self.bn1 = nn.BatchNorm2d(32)
            self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
            self.bn2 = nn.BatchNorm2d(64)
            self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))

            self.fc1 = nn.Linear(64 * 4 * 4, 128)
            self.fc2 = nn.Linear(128, num_classes)

        elif dataset_name == 'cifar':
            self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=5, stride=1, padding=2)
            self.bn1 = nn.BatchNorm2d(32)
            self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
            self.bn2 = nn.BatchNorm2d(64)
            self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
            self.bn3 = nn.BatchNorm2d(128)
            self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
            self.bn4 = nn.BatchNorm2d(256)
            self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))
            self.fc1 = nn.Linear(256 * 4 * 4, 512)
            self.fc2 = nn.Linear(512, 128)
            self.fc3 = nn.Linear(128, num_classes)

        self.dropout = nn.Dropout(0.5)

    def forward(self, x, dataset_name):
        if dataset_name == 'mnist':
            x = F.relu(self.bn1(self.conv1(x)))
            x = F.max_pool2d(x, 2)
            x = F.relu(self.bn2(self.conv2(x)))
            x = F.max_pool2d(x, 2)
            x = self.adaptive_pool(x)
            x = x.view(x.size(0), -1)
            x = F.relu(self.fc1(x))
            x = self.dropout(x)
            x = self.fc2(x)

        elif dataset_name == 'cifar':
            x = F.relu(self.bn1(self.conv1(x)))
            x = F.max_pool2d(x, 2)
            x = F.relu(self.bn2(self.conv2(x)))
            x = F.max_pool2d(x, 2)
            x = F.relu(self.bn3(self.conv3(x)))
            x = F.max_pool2d(x, 2)
            x = F.relu(self.bn4(self.conv4(x)))
            x = self.adaptive_pool(x)
            x = x.view(x.size(0), -1)
            x = F.relu(self.fc1(x))
            x = self.dropout(x)
            x = F.relu(self.fc2(x))
            x = self.dropout(x)
            x = self.fc3(x)

        return x


# -------------------- Training and Evaluation --------------------
def train_model(model, train_loader, test_loader, criterion, optimizer, scheduler, device, epochs, dataset_name):
    best_acc = 0.0
    model_dir = "model"

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    model_filename = f"{dataset_name}_trained_model.pth"
    best_model_path = os.path.join(model_dir, model_filename)

    print(f"{'Loop':>6}{'Train Loss':>15}{'Train Acc %':>20}{'Test Loss':>20}{'Test Acc %':>20}")

    for epoch in range(epochs):
        model.train()
        total_correct = 0
        total_samples = 0
        running_loss = 0.0

        with tqdm(train_loader, unit="batch", leave=False) as train_epoch:
            train_epoch.set_description(f"Epoch [{epoch + 1}/{epochs}] - Training")
            for batch_idx, (inputs, labels) in enumerate(train_epoch):
                inputs, labels = inputs.to(device), labels.to(device)
                total_samples += labels.size(0)

                outputs = model(inputs, dataset_name)
                loss = criterion(outputs, labels)

                optimizer.zero_grad()
                loss.backward()

                nn.utils.clip_grad_value_(model.parameters(), 0.1)
                optimizer.step()

                running_loss += loss.item()
                total_correct += (torch.max(outputs, 1)[1] == labels).sum().item()

                train_epoch.set_postfix({
                    'Loss': running_loss / (batch_idx + 1),
                    'Accuracy': 100. * total_correct / total_samples
                })

        train_loss = running_loss / len(train_loader)
        train_acc = 100. * total_correct / total_samples

        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with tqdm(test_loader, unit="batch", leave=False) as test_bar:
            test_bar.set_description(f"Epoch [{epoch + 1}/{epochs}] - Testing")
            with torch.no_grad():
                for batch_idx, (inputs, labels) in enumerate(test_bar):
                    inputs, labels = inputs.to(device), labels.to(device)

                    outputs = model(inputs, dataset_name)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()

                    _, predicted = torch.max(outputs, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

                    test_bar.set_postfix({
                        'Loss': val_loss / (batch_idx + 1),
                        'Accuracy': 100. * correct / total
                    })

        val_loss /= len(test_loader)
        val_acc = 100. * correct / total

        print(f"{epoch:>2}/{epochs:<1}{train_loss:>15f}{train_acc:>20f}{val_loss:>20f}{val_acc:>20f}") if \
            (epochs == 10 or (epochs == 50 and (epoch % 10 == 0 or epoch == epochs - 1))) else None

        scheduler.step(val_loss)

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), best_model_path)

    print(f"Model trained on the {dataset_name} dataset with best test accuracy: {best_acc:.2f}% saved in file: {best_model_path}.")


# -------------------- Dataset Utilities --------------------
def get_dataset_info(dataset_name):
    if dataset_name == 'cifar':
        dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transforms.ToTensor())
        mnist_labels = dataset.classes
        input_channels = dataset[0][0].shape[0]
        num_classes = len(dataset.classes)
        means = (dataset.data / 255.0).mean(axis=(0, 1, 2))
        stds = (dataset.data / 255.0).std(axis=(0, 1, 2))
        return input_channels, num_classes, means, stds, mnist_labels
    elif dataset_name == 'mnist':
        dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())
        cifar_10_labels = dataset.classes
        input_channels = 1
        num_classes = len(dataset.classes)
        means = (dataset.data / 255.0).mean()
        stds = (dataset.data / 255.0).std()
        return input_channels, num_classes, means, stds, cifar_10_labels


def load_dataset(dataset_name, batch_size, means, stds):
    if dataset_name == 'cifar':
        transform_train = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=means, std=stds)
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=means, std=stds)
        ])
        train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
        test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    elif dataset_name == 'mnist':
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(means,), std=(stds,))
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(means,), std=(stds,))
        ])

        train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform_train)
        test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=4)
    return train_loader, test_loader


# -------------------- Image Utilities --------------------
def preprocess_image(image_path, dataset_name, mean=None, std=None):

    if dataset_name == 'mnist':
        image = Image.open(image_path).convert("L")
        image_np = np.array(image)
        mean_brightness = image_np.mean()

        if mean_brightness > 127:
            image = ImageOps.invert(image)

        transform = transforms.Compose([
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
            transforms.Normalize((mean,), (std,))
        ])
    elif dataset_name == 'cifar':

        image = Image.open(image_path)

        if image.mode != 'RGB':
            image = image.convert("RGB")

        transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

    image = transform(image).unsqueeze(0)
    return image


# -------------------- Inference --------------------
def test_single_image(model, image_path, device, dataset_name, mean, std, dataset_labels):
    image = preprocess_image(image_path, dataset_name, mean, std).to(device)

    model.eval()
    with torch.no_grad():
        conv1_output = model.conv1(image)

        output = model(image, dataset_name)
        _, predicted = torch.max(output, 1)
        predicted_class_idx = predicted.item()

    visualize_first_conv_layer(conv1_output, dataset_name)

    return dataset_labels[predicted_class_idx]


# -------------------- Visualization --------------------
def visualize_first_conv_layer(conv1_output, dataset_name):
    conv1_output = conv1_output.cpu()
    num_filters = conv1_output.shape[1]

    fig, axes = plt.subplots(4, 8, figsize=(14, 8), dpi=300)

    for i in range(num_filters):
        ax = axes[i // 8, i % 8]
        feature_map = conv1_output[0, i].numpy()
        feature_map = (feature_map - feature_map.min()) / (feature_map.max() - feature_map.min())
        ax.imshow(feature_map, cmap='gray', interpolation='bicubic')
        ax.axis('off')

    output_filename = f"CONV_rslt_{dataset_name}.png"
    plt.tight_layout()
    plt.savefig(output_filename, dpi=200)
    plt.close()


# -------------------- Model utilities --------------------

def load_saved_model(model_class, num_classes, in_channels, model_dir, dataset_name, device):
    model_filename = f"{dataset_name}_trained_model.pth"
    model_path = os.path.join(model_dir, model_filename)
    model = model_class(in_channels=in_channels, num_classes=num_classes, dataset_name=dataset_name)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model


# -------------------- CLI --------------------

def determine_dataset(image_path):
    image = Image.open(image_path)

    if image.mode == 'RGB':
        return 'cifar'
    elif image.mode == 'L':  # Grayscale image
        return 'mnist'

def main():
    epilog = """Usage examples:
      python3 CNNclassify.py train --mnist       Train the model using the MNIST dataset
      python3 CNNclassify.py train --cifar       Train the model using the CIFAR-10 dataset
      python3 CNNclassify.py test car.png        Test the model using 'car.png'
    """

    parser = argparse.ArgumentParser(
        usage='python3 CNNclassify.py [-h] {train,test,save} ...',
        formatter_class=argparse.RawTextHelpFormatter,
        add_help=False,
        epilog=epilog
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to execute: train, test, or save")

    # Training parser
    train_parser = subparsers.add_parser('train', help="Train the model")
    train_parser.add_argument('--mnist', action='store_true', help="Train the CNN using the MNIST dataset.")
    train_parser.add_argument('--cifar', action='store_true', help="Train the CNN using the CIFAR-10 dataset.")

    # Testing parser
    test_parser = subparsers.add_parser('test', help="Test the model")
    test_parser.add_argument('image_file', nargs='*', help="Image file(s) for testing (e.g., car.png)")

    args = parser.parse_args()

    if len(sys.argv) == 1 or args.command is None:
        parser.print_help()
        sys.exit(1)

    # Train command
    if args.command == 'train':
        if not args.mnist and not args.cifar:
            print("Error: 'train' command requires either --mnist or --cifar argument.", file=sys.stderr)
            print("Use --help for more information.")
            sys.exit(1)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        dataset = 'mnist' if args.mnist else 'cifar'

        batch_size = 64
        input_channels, num_classes, means, stds, dataset_labels = get_dataset_info(dataset)
        train_loader, test_loader = load_dataset(dataset, batch_size, means, stds)

        model = CNNClassifier(in_channels=input_channels, num_classes=num_classes, dataset_name=dataset).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0005)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2, verbose=False)
        num_epochs = 10 if dataset == 'mnist' else 50

        train_model(model, train_loader, test_loader, criterion, optimizer, scheduler, device, num_epochs, dataset)

    # Test command
    elif args.command == 'test':
        if not args.image_file:
            print("Error: 'test' command requires at least one image file.", file=sys.stderr)
            print("Use --help for more information.")
            sys.exit(1)

        model_dir = "model"
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        dataset = 'cifar'
        input_channels, num_classes, means, stds, dataset_labels = get_dataset_info(dataset)
        model = load_saved_model(CNNClassifier, num_classes, input_channels, model_dir, dataset, device)

        for img_file in args.image_file:
            image_path = img_file
            predicted_label = test_single_image(model, image_path, device, dataset, means, stds, dataset_labels)
            print(f"Prediction result: {predicted_label}")


if __name__ == '__main__':
    main()
