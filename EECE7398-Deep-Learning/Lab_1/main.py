import argparse
import sys
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import torch.optim as optim
import torch.nn as nn
import pytorch_lightning as pl
from PIL import Image
import os
import torch
import matplotlib.pyplot as plt
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
        super(CNNClassifier, self).__init__()

        # First CONV layer: filter size = 5x5, stride = 1, 32 filters
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
            nn.Dropout(p=0.75),
            nn.Linear(256, num_classes)
        )

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


def calculate_accuracy(preds, labels):
    _, predicted = preds.max(1)
    correct = predicted.eq(labels).sum().item()
    return correct


def train_model(model, train_loader, test_loader, criterion, optimizer, scheduler, device, epochs, dataset_name):
    best_acc = 0.0
    model_dir = "model"

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    model_filename = f"{dataset_name}_best_acc_model.pth"
    best_model_path = os.path.join(model_dir, model_filename)

    print(f"{'Loop':>5}{'Train Loss':>15}{'Train Acc %':>20}{'Test Loss':>20}{'Test Acc %':>20}")

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        total_correct = 0
        total_samples = 0

        train_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc="Training", leave=False)

        for batch_idx, (inputs, labels) in train_bar:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            nn.utils.clip_grad_value_(model.parameters(), 0.1)
            optimizer.step()

            running_loss += loss.item()
            total_correct += calculate_accuracy(outputs, labels)
            total_samples += labels.size(0)

            train_bar.set_postfix({
                'Loss': running_loss / (batch_idx + 1),
                'Accuracy': 100. * total_correct / total_samples
            })

        train_loss = running_loss / len(train_loader)
        train_acc = 100. * total_correct / total_samples

        model.eval()
        test_loss = 0.0
        total_correct = 0
        total_samples = 0

        test_bar = tqdm(enumerate(test_loader), total=len(test_loader), desc="Testing", leave=False)

        with torch.no_grad():
            for batch_idx, (inputs, labels) in test_bar:
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)
                test_loss += loss.item()

                total_correct += calculate_accuracy(outputs, labels)
                total_samples += labels.size(0)

                test_bar.set_postfix({
                    'Loss': test_loss / (batch_idx + 1),
                    'Accuracy': 100. * total_correct / total_samples
                })

        test_loss /= len(test_loader)
        test_acc = 100. * total_correct / total_samples

        if epoch == 1 or epoch % 5 == 0 or epoch == epochs:
            print(f"{epoch:>2}/{epochs:<1}{train_loss:>15.4f}{train_acc:>20.4f}{test_loss:>20.4f}{test_acc:>20.4f}")

        scheduler.step(test_loss)

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), best_model_path)

    print(f"Model trained on {dataset_name} with test accuracy {best_acc:.2f}% saved in file: {best_model_path}.")


def load_dataset(dataset_name, batch_size):
    if dataset_name == 'mnist':
        # Define transformations for MNIST
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))  # Normalize for grayscale
        ])
        # Load MNIST dataset
        train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
        input_channels = 1  # Grayscale

    elif dataset_name == 'cifar':
        # Define transformations for CIFAR-10
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize for RGB
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize for RGB
        ])
        # Load CIFAR-10 dataset
        train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
        test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
        input_channels = 3  # RGB

    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=4)

    num_classes = len(train_dataset.classes)
    return train_loader, test_loader, input_channels, num_classes


def load_saved_model(model_class, num_classes, in_channels, model_dir, dataset_name, device):
    model_filename = f"{dataset_name}_best_acc_model.pth"
    model_path = os.path.join(model_dir, model_filename)
    model = model_class(in_channels=in_channels, num_classes=num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model


def preprocess_image(image_path, dataset_name):
    if dataset_name == "mnist":
        image_size = (28, 28)  # For MNIST (grayscale images)
        transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))  # Normalize for grayscale
        ])
        image = Image.open(image_path).convert('L')  # Convert to grayscale
    elif dataset_name == "cifar":
        image_size = (32, 32)  # For CIFAR-10 (RGB images)
        transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize for RGB
        ])
        image = Image.open(image_path).convert('RGB')  # Convert to RGB

    image = transform(image).unsqueeze(0)  # Add batch dimension
    return image


# Hook to capture the output of the first CONV layer
def hook_fn(module, input, output):
    global conv_output
    conv_output = output  # Save the output for visualization


# Function to visualize and save the output of the first CONV layer for test image
def visualize_first_conv_layer(model, image_tensor, device, dataset_name, output_filename):
    global conv_output
    conv_output = None

    # Register a hook on the first CONV layer
    first_conv_layer = model.conv1[0]  # Assuming conv1 is the first layer
    hook = first_conv_layer.register_forward_hook(hook_fn)

    # Forward pass through the model with the test image
    image_tensor = image_tensor.to(device)
    with torch.no_grad():
        model(image_tensor)

    # Remove the hook after forward pass
    hook.remove()

    # Get the activations of the first CONV layer
    if conv_output is None:
        raise RuntimeError("Conv output is not captured, make sure the hook is correctly registered.")

    # Squeeze batch dimension and convert to CPU
    activations = conv_output.squeeze(0).cpu()

    # Plot the activations for each filter
    num_filters = activations.shape[0]
    fig, axes = plt.subplots(4, 8, figsize=(12, 6))  # Create a 4x8 grid for 32 filters
    for i, ax in enumerate(axes.flat):
        if i < num_filters:
            ax.imshow(activations[i].detach().numpy(), cmap='viridis')
            ax.set_title(f'Filter {i + 1}')
        ax.axis('off')

    # Save the visualization
    plt.suptitle(f'Activations of First CONV Layer ({dataset_name})')
    plt.tight_layout()
    plt.savefig(output_filename)
    plt.close()
    print(f"Visualization saved as {output_filename}")


def classify_image(model, image_tensor, device):
    image_tensor = image_tensor.to(device)  # Move the image tensor to the correct device
    with torch.no_grad():
        output = model(image_tensor)  # Forward pass
        _, predicted_class = torch.max(output, 1)  # Get the class with the highest score
    return predicted_class.item()


# Function to get MNIST class labels
def get_mnist_label(class_idx):
    mnist_labels = {0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9'}
    return mnist_labels.get(class_idx, "Unknown")


# Function to get CIFAR-10 class labels
def get_cifar_label(class_idx):
    cifar_labels = {0: 'airplane', 1: 'automobile', 2: 'bird', 3: 'cat', 4: 'deer', 5: 'dog', 6: 'frog', 7: 'horse',
                    8: 'ship', 9: 'truck'}
    return cifar_labels.get(class_idx, "Unknown")


def dynamic_test_classification(model_class, num_classes_mnist, num_classes_cifar, model_dir, image_path, device):
    # Load the MNIST model (grayscale images)
    mnist_model = load_saved_model(model_class, num_classes=10, in_channels=1, model_dir=model_dir,
                                   dataset_name="mnist", device=device)

    # Load the CIFAR-10 model (RGB images)
    cifar_model = load_saved_model(model_class, num_classes=10, in_channels=3, model_dir=model_dir,
                                   dataset_name="cifar", device=device)

    # Preprocess the image for MNIST
    mnist_image_tensor = preprocess_image(image_path, "mnist")

    # Preprocess the image for CIFAR-10
    cifar_image_tensor = preprocess_image(image_path, "cifar")

    # Classify the image using MNIST model
    mnist_pred_class = classify_image(mnist_model, mnist_image_tensor, device)

    # Classify the image using CIFAR-10 model
    cifar_pred_class = classify_image(cifar_model, cifar_image_tensor, device)

    # Get class labels
    mnist_label = get_mnist_label(mnist_pred_class)
    cifar_label = get_cifar_label(cifar_pred_class)

    # Visualize the first CONV layer for both models
    visualize_first_conv_layer(mnist_model, mnist_image_tensor, device, "mnist", "CONV_rslt_mnist.png")
    visualize_first_conv_layer(cifar_model, cifar_image_tensor, device, "cifar", "CONV_rslt_cifar.png")

    return mnist_pred_class, cifar_pred_class, mnist_label, cifar_label


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
        train_loader, test_loader, input_channels, num_classes = load_dataset(dataset, batch_size)

        model = CNNClassifier(in_channels=input_channels, num_classes=num_classes)
        device = torch.device(
            "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5,
                                                               verbose=False)
        train_model(model, train_loader, test_loader, criterion, optimizer, scheduler, device, 50, dataset)

    elif args.command == 'test':
        if not args.image_file:
            print("Error: 'test' command requires at least one image file.", file=sys.stderr)
            print("Use --help for more information.")
            sys.exit(1)

        for img_file in args.image_file:
            print(f"Predicting the class for image: {img_file}")
            model_dir = "model"  # Directory where your models are saved
            image_path = img_file  # Path to the input image
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

            # Load and visualize for MNIST
            mnist_model = load_saved_model(CNNClassifier, num_classes=10, in_channels=1, model_dir=model_dir,
                                           dataset_name="mnist", device=device)
            # Replace CNNClassifier with the class name of your model architecture
            mnist_pred_class, cifar_pred_class, mnist_label, cifar_label = dynamic_test_classification(CNNClassifier,
                                                                                                       num_classes_mnist=10,
                                                                                                       num_classes_cifar=10,
                                                                                                       model_dir=model_dir,
                                                                                                       image_path=image_path,
                                                                                                       device=device)

            print(f"The MNIST model predicted class: {mnist_pred_class}")
            print(f"The CIFAR-10 model predicted class: {cifar_pred_class}")
            print(f"MNIST model predicted class: {mnist_label}")
            print(f"CIFAR-10 model predicted class: {cifar_label}")


if __name__ == "__main__":
    main()
