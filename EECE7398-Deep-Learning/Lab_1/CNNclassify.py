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
    def __init__(self, in_channels, num_classes):
        super(CNNClassifier, self).__init__()

        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=5, stride=1, padding=2)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        # Adaptive pooling to ensure a consistent output size regardless of input dimensions
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))  # Fixed output size of 4x4 for any input

        # Fully connected layers (after adaptive pooling, size will be 128 * 4 * 4 = 2048)
        self.fc1 = nn.Linear(128 * 4 * 4, 256)  # Fixed size after adaptive pooling
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, num_classes)

        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # Convolutional layers with ReLU and BatchNorm
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.max_pool2d(x, 2)

        # Adaptive pooling to ensure the output feature map is always 4x4
        x = self.adaptive_pool(x)

        # Flatten the output for fully connected layers
        x = x.view(x.size(0), -1)

        # Fully connected layers with dropout
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

                # Forward pass
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                # Backward pass and optimization
                optimizer.zero_grad()
                loss.backward()

                # Optional: Gradient clipping
                nn.utils.clip_grad_value_(model.parameters(), 0.1)
                optimizer.step()

                # Track running loss and correct predictions
                running_loss += loss.item()
                total_correct += (torch.max(outputs, 1)[1] == labels).sum().item()

                # Update progress bar
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

                    outputs = model(inputs)
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
            (epochs == 10 or (epochs == 100 and (epoch % 10 == 0 or epoch == epochs - 1))) else None

        scheduler.step(val_loss)

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), best_model_path)

    print(f"Model trained on {dataset_name} with best test accuracy: {best_acc:.2f}% saved in file: {best_model_path}.")


# -------------------- Dataset Utilities --------------------
def get_dataset_info(dataset_name):
    if dataset_name == 'cifar':
        dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transforms.ToTensor())
        input_channels = dataset[0][0].shape[0]
        num_classes = len(dataset.classes)
        means = (dataset.data / 255.0).mean(axis=(0, 1, 2))
        stds = (dataset.data / 255.0).std(axis=(0, 1, 2))
        return input_channels, num_classes, means, stds
    elif dataset_name == 'mnist':
        dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())
        input_channels = 1
        num_classes = len(dataset.classes)
        means = (dataset.data / 255.0).mean()
        stds = (dataset.data / 255.0).std()
        return input_channels, num_classes, means, stds


def load_dataset(dataset_name, batch_size, means, stds):
    if dataset_name == 'cifar':
        transform_train = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
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

    # Data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=4)
    return train_loader, test_loader


# -------------------- Image Utilities --------------------
def preprocess_image(image_path, dataset_name, mean, std):
    # Open the image
    image = Image.open(image_path)

    if dataset_name == 'mnist':
        # Convert to grayscale (for MNIST images)
        image = image.convert("L")

        # Check the background brightness (assume most of the background is the pixel majority)
        image_np = np.array(image)
        mean_brightness = image_np.mean()

        # Invert the image if the background is white (brightness > 127)
        if mean_brightness > 127:
            image = ImageOps.invert(image)

        # Define MNIST-specific transforms
        transform = transforms.Compose([
            transforms.Resize((28, 28)),  # Ensure it's 28x28
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))  # MNIST normalization
        ])
    elif dataset_name == 'cifar':
        # Define CIFAR-10-specific transforms
        transform = transforms.Compose([
            transforms.Resize((32, 32)),  # Ensure it's 32x32
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)  # CIFAR-10 normalization
        ])

    # Apply the transformations
    image_tensor = transform(image).unsqueeze(0)  # Add batch dimension (1, C, H, W)

    return image_tensor


# -------------------- Inference --------------------
def test_single_image(model, image_path, device, dataset_name, mean=None, std=None):
    # Check if the dataset is MNIST or CIFAR and apply respective transformations
    if dataset_name == 'mnist':
        # Load the image in grayscale
        image = Image.open(image_path).convert("L")  # Convert to grayscale

        # Check the background brightness (assume most of the background should be the pixel majority)
        image_np = np.array(image)
        mean_brightness = image_np.mean()

        # If the background is bright (closer to white), we assume it's inverted and needs to be flipped
        if mean_brightness > 127:  # Background is bright, invert the image
            image = ImageOps.invert(image)

        # Define MNIST-specific transforms
        transform = transforms.Compose([
            transforms.Resize((28, 28)),  # Ensure it's 28x28
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))  # MNIST normalization
        ])
    elif dataset_name == 'cifar':
        # Load the image and apply CIFAR-10 transformations
        image = Image.open(image_path)

        # Define CIFAR-specific transforms
        transform = transforms.Compose([
            transforms.Resize((32, 32)),  # Ensure it's 32x32
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)  # CIFAR-10 normalization
        ])

    # Apply the transformations
    image = transform(image).unsqueeze(0)  # Add batch dimension (1, C, H, W)
    image = image.to(device)

    # Set the model to evaluation mode and make predictions
    model.eval()
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)
        predicted_label = predicted.item()

    return predicted_label


# -------------------- Visualization --------------------
def visualize_first_conv_layer(model, image_path, device, output_file="CONV_rslt_mnist.png"):
    model.eval()  # Set the model to evaluation mode

    # Load and preprocess the input image
    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0)  # Add batch dimension
    image = image.to(device)

    # Extract the feature maps from the first convolutional layer
    with torch.no_grad():
        conv1_output = model.conv1(image)

    # Convert the output to CPU for visualization
    conv1_output = conv1_output.cpu()

    # Plot all the feature maps (32 filters) from the first CONV layer
    num_filters = conv1_output.shape[1]  # Should be 32
    fig, axes = plt.subplots(4, 8, figsize=(12, 6))  # Create subplots (4 rows, 8 columns)

    for i in range(num_filters):
        ax = axes[i // 8, i % 8]  # Determine the position in the 4x8 grid
        ax.imshow(conv1_output[0, i].numpy(), cmap='gray')  # Plot the feature map for each filter
        ax.axis('off')  # Turn off the axes for a cleaner look

    # Adjust the spacing between the subplots
    plt.subplots_adjust(wspace=0.5, hspace=0.5)  # Reduce the width and height space between subplots

    plt.tight_layout()
    plt.savefig(output_file)  # Save the resulting figure as a PNG file
    print(f"Feature maps saved as {output_file}")


# -------------------- Model utilities --------------------

def load_saved_model(model_class, num_classes, in_channels, model_dir, dataset_name, device):
    model_filename = f"{dataset_name}_trained_model.pth"
    model_path = os.path.join(model_dir, model_filename)

    # Load the saved model's state_dict
    state_dict = torch.load(model_path, map_location=device)

    # Initialize the model and load the state_dict
    model = model_class(in_channels=in_channels, num_classes=num_classes)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


# -------------------- CLI --------------------

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
    train_parser.add_argument('--mnist', action='store_true', help="Use the MNIST dataset for training")
    train_parser.add_argument('--cifar', action='store_true', help="Use the CIFAR-10 dataset for training")

    # Testing parser
    test_parser = subparsers.add_parser('test', help="Test the model")
    test_parser.add_argument('image_file', nargs='*', help="Image file(s) for testing (e.g., car.png)")

    # Saving images parser
    save_parser = subparsers.add_parser('save', help="Save images from the dataset")
    save_parser.add_argument('--mnist', action='store_true', help="Use the MNIST dataset to save images")
    save_parser.add_argument('--cifar', action='store_true', help="Use the CIFAR-10 dataset to save images")
    save_parser.add_argument('--output_dir', type=str, default='./output_images', help="Directory to save images")
    save_parser.add_argument('--num_images', type=int, default=10, help="Number of images to save (default: 10)")

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
        # Determine which dataset to use
        dataset = 'mnist' if args.mnist else 'cifar'
        print(f"Training the model on {dataset} dataset")

        # Set batch size and get dataset info
        batch_size = 64
        input_channels, num_classes, means, stds = get_dataset_info(dataset)
        train_loader, test_loader = load_dataset(dataset, batch_size, means, stds)

        # Initialize the model, loss function, optimizer, and learning rate scheduler
        model = CNNClassifier(in_channels=input_channels, num_classes=num_classes).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0005)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2, verbose=True)

        # Set the number of epochs depending on the dataset
        if dataset == 'mnist':
            num_epochs = 10
        else:
            num_epochs = 100

        # Train the model with the specified number of epochs
        train_model(model, train_loader, test_loader, criterion, optimizer, scheduler, device, num_epochs, dataset)

    # Test command
    elif args.command == 'test':
        if not args.image_file:
            print("Error: 'test' command requires at least one image file.", file=sys.stderr)
            print("Use --help for more information.")
            sys.exit(1)

        model_dir = "model"
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        dataset = 'mnist'
        input_channels, num_classes, means, stds = get_dataset_info(dataset)
        model = load_saved_model(CNNClassifier, num_classes, input_channels, model_dir, dataset, device)

        for img_file in args.image_file:
            print(f"Predicting the class for image: {img_file}")
            image_path = img_file

            predicted_label = test_single_image(model, image_path, device, dataset, means, stds)
            print(f"Image: {img_file}, Predicted Label: {predicted_label}")
            visualize_first_conv_layer(model, img_file, device, output_file="CONV_rslt_mnist.png")


if __name__ == '__main__':
    main()
