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
import torch
import torch.nn as nn
import torch.nn.functional as F
import random


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

    # Set task based on dataset
    task = 'mnist' if dataset_name == 'mnist' else 'cifar'

    annealer = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 0.95 ** epoch)

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        total_correct = 0
        total_samples = 0

        train_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Training {dataset_name}", leave=False)

        for batch_idx, (inputs, labels) in train_bar:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs, task=task)  # Pass task to the model
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

        test_bar = tqdm(enumerate(test_loader), total=len(test_loader), desc=f"Testing {dataset_name}", leave=False)

        with torch.no_grad():
            for batch_idx, (inputs, labels) in test_bar:
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = model(inputs, task=task)  # Pass task to the model
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

        #scheduler.step(test_loss)
        annealer.step()

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), best_model_path)

    print(f"Model trained on {dataset_name} with test accuracy {best_acc:.2f}% saved in file: {best_model_path}.")


class MultiTaskCNN(nn.Module):
    def __init__(self, num_classes_mnist, num_classes_cifar):
        super(MultiTaskCNN, self).__init__()

        # MNIST-specific components (simplified, no residuals)
        self.conv1_mnist = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=5, stride=2, padding=2),  # Strided conv to reduce spatial dimensions
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)  # Increased Dropout for regularization
        )

        # CIFAR-specific components (with residuals)
        self.conv1_cifar = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)  # Max pooling for downsampling
        )

        # Shared convolutional layers
        self.conv2 = self.conv_block(32, 64)
        self.conv3 = self.conv_block(64, 128)
        self.conv4 = self.conv_block(128, 256)

        # Residual Block 1 (for CIFAR)
        self.res1_conv1 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.res1_bn1 = nn.BatchNorm2d(64)
        self.res1_conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.res1_bn2 = nn.BatchNorm2d(64)

        # Residual Block 2 (for CIFAR)
        self.res2_conv1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.res2_bn1 = nn.BatchNorm2d(256)
        self.res2_conv2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.res2_bn2 = nn.BatchNorm2d(256)

        # Classification heads for MNIST and CIFAR-10
        self.classifier_mnist = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(p=0.5),  # Increased Dropout for MNIST
            nn.Linear(256, num_classes_mnist)  # Fully connected layer for MNIST
        )

        self.classifier_cifar = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(p=0.8),  # Dropout for CIFAR-10
            nn.Linear(256, num_classes_cifar)
        )

    def conv_block(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        ]
        return nn.Sequential(*layers)

    def forward(self, x, task):
        if task == 'mnist':
            x = self.conv1_mnist(x)  # MNIST-specific conv block (no residuals)
        elif task == 'cifar':
            x = self.conv1_cifar(x)  # CIFAR-specific conv block

        x = self.conv2(x)

        if task == 'cifar':
            # Residual Block 1 for CIFAR
            res1 = x
            x = self.res1_conv1(x)
            x = self.res1_bn1(x)
            x = F.relu(x)
            x = self.res1_conv2(x)
            x = self.res1_bn2(x)
            x += res1  # Skip connection
            x = F.relu(x)

        x = self.conv3(x)
        x = self.conv4(x)

        if task == 'cifar':
            # Residual Block 2 for CIFAR
            res2 = x
            x = self.res2_conv1(x)
            x = self.res2_bn1(x)
            x = F.relu(x)
            x = self.res2_conv2(x)
            x = self.res2_bn2(x)
            x += res2  # Skip connection
            x = F.relu(x)

        if task == 'mnist':
            x = self.classifier_mnist(x)
        elif task == 'cifar':
            x = self.classifier_cifar(x)

        return x


def load_dataset(dataset_name, batch_size):
    if dataset_name == 'mnist':
        train_transform = transforms.Compose([
            transforms.RandomRotation(15),  # Random rotations to make the task slightly harder
            transforms.RandomAffine(0, translate=(0.1, 0.1)),  # Random translations to mimic real-world distortions
            transforms.RandomPerspective(distortion_scale=0.2, p=0.5),  # Perspective transformations
            transforms.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 2.0)),  # Slight blur
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))  # Normalization for grayscale images
        ])

        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))  # Normalization for grayscale images
        ])

        train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=train_transform)
        test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=test_transform)
        input_channels = 1  # Grayscale for MNIST

    elif dataset_name == 'cifar':
        transform_train = transforms.Compose([
            transforms.RandomHorizontalFlip(),  # Horizontal flips for CIFAR
            transforms.RandomCrop(32, padding=4),  # Random cropping
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalization for RGB images
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
        test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
        input_channels = 3  # RGB for CIFAR-10

    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=0)

    num_classes = len(train_dataset.classes)
    return train_loader, test_loader, input_channels, num_classes


def load_saved_model(model_class, num_classes_mnist, num_classes_cifar, model_dir, device):
    model = model_class(num_classes_mnist=num_classes_mnist, num_classes_cifar=num_classes_cifar)

    # Load state_dict for MNIST and CIFAR separately
    mnist_state_dict = torch.load(f"{model_dir}/mnist_best_acc_model.pth", map_location=device)
    cifar_state_dict = torch.load(f"{model_dir}/cifar_best_acc_model.pth", map_location=device)

    model.load_state_dict(mnist_state_dict, strict=False)  # Load MNIST weights
    model.load_state_dict(cifar_state_dict, strict=False)  # Load CIFAR weights

    model.to(device)
    model.eval()
    return model


def preprocess_image(image_path, dataset_name):
    image = Image.open(image_path).convert("RGB")
    if dataset_name == 'mnist':
        transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
    elif dataset_name == 'cifar':
        transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    else:
        raise ValueError(f"Unknown dataset name: {dataset_name}")

    image_tensor = transform(image).unsqueeze(0)
    print(f"{dataset_name.upper()} image tensor shape: {image_tensor.shape}")  # Debugging step: print shape
    return image_tensor


# Hook function for visualizing first convolution layer
# Hook function to capture the output of the first convolutional layer
def hook_fn(module, input, output):
    global conv_output
    conv_output = output  # Save output for visualization

# Visualize the first convolutional layer's output
def visualize_first_conv_layer(model, image_tensor, device, dataset_name, output_filename, task):
    global conv_output
    conv_output = None

    try:
        first_conv_layer = model.conv1_mnist[0] if task == 'mnist' else model.conv1_cifar[0]
    except Exception as e:
        raise RuntimeError(f"Error accessing the first convolutional layer: {e}")

    hook = first_conv_layer.register_forward_hook(hook_fn)
    image_tensor = image_tensor.to(device)
    with torch.no_grad():
        model(image_tensor, task=task)

    hook.remove()

    if conv_output is None:
        raise RuntimeError("Conv output not captured correctly.")

    activations = conv_output.squeeze(0).cpu()
    num_filters = activations.shape[0]
    grid_size = int(num_filters ** 0.5)
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(12, 12))

    for i, ax in enumerate(axes.flat):
        if i < num_filters:
            ax.imshow(activations[i].detach().numpy(), cmap='viridis')
            ax.set_title(f'Filter {i + 1}')
        ax.axis('off')

    plt.suptitle(f'Activations of First CONV Layer ({dataset_name})')
    plt.tight_layout()
    plt.savefig(output_filename)
    plt.close()
    print(f"Visualization saved as {output_filename}")


# Classify an image
def classify_image(model, image_tensor, device, task):
    image_tensor = image_tensor.to(device)
    with torch.no_grad():
        output = model(image_tensor, task=task)
        probabilities = F.softmax(output, dim=1)  # Get softmax probabilities
        _, predicted_class = torch.max(output, 1)

    print(f"Predicted class probabilities: {probabilities.cpu().numpy()}")  # Log the prediction probabilities
    return predicted_class.item()


# Dynamic Test Classification
def dynamic_test_classification(model_class, num_classes_mnist, num_classes_cifar, model_dir, image_path, device):
    model = load_saved_model(model_class, num_classes_mnist, num_classes_cifar, model_dir, device)

    print("Loaded model layers:", model.state_dict().keys())  # Debugging step: print model layers

    mnist_image_tensor = preprocess_image(image_path, 'mnist').to(device)
    visualize_first_conv_layer(model, mnist_image_tensor, device, 'mnist', 'mnist_conv_output.png', task='mnist')
    mnist_pred_class = classify_image(model, mnist_image_tensor, device, task='mnist')
    mnist_label = get_mnist_label(mnist_pred_class)
    print(f"The MNIST model predicted class: {mnist_pred_class} (Label: {mnist_label})")

    cifar_image_tensor = preprocess_image(image_path, 'cifar').to(device)
    visualize_first_conv_layer(model, cifar_image_tensor, device, 'cifar', 'cifar_conv_output.png', task='cifar')
    cifar_pred_class = classify_image(model, cifar_image_tensor, device, task='cifar')
    cifar_label = get_cifar_label(cifar_pred_class)
    print(f"The CIFAR-10 model predicted class: {cifar_pred_class} (Label: {cifar_label})")

    return mnist_pred_class, cifar_pred_class, mnist_label, cifar_label


# Label mappings
def get_mnist_label(pred_class):
    mnist_labels = [str(i) for i in range(10)]
    return mnist_labels[pred_class]


def get_cifar_label(pred_class):
    cifar_labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    return cifar_labels[pred_class]


def test_on_mnist_images(model, device):
    # Load the MNIST test dataset
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transforms.ToTensor())

    # Select a random image from the MNIST test set
    random_idx = random.randint(0, len(test_dataset) - 1)
    test_image, test_label = test_dataset[random_idx]

    # Convert to batch format
    test_image = test_image.unsqueeze(0).to(device)

    # Classify using the model
    mnist_pred_class = classify_image(model, test_image, device, task='mnist')

    print(f"Testing MNIST image (Label: {test_label})")
    print(f"MNIST model predicted class: {mnist_pred_class}")


def test_on_external_image(image_path, model, device):
    mnist_pred_class, cifar_pred_class, mnist_label, cifar_label = dynamic_test_classification(
        MultiTaskCNN, num_classes_mnist=10, num_classes_cifar=10, model_dir='model', image_path=image_path,
        device=device
    )
    print(f"The MNIST model predicted class: {mnist_pred_class} (Label: {mnist_label})")
    print(f"The CIFAR-10 model predicted class: {cifar_pred_class} (Label: {cifar_label})")

def main():
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
    test_parser.add_argument("--mnist", action="store_true", help="Test on a random MNIST dataset image")

    args = parser.parse_args()

    if len(sys.argv) == 1 or args.command is None:
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
        model = MultiTaskCNN(num_classes_mnist=10, num_classes_cifar=10)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)
        train_model(model, train_loader, test_loader, criterion, optimizer, scheduler, device, 50, dataset)

    elif args.command == 'test':
        if not args.image_file:
            print("Error: 'test' command requires at least one image file.", file=sys.stderr)
            print("Use --help for more information.")
            sys.exit(1)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = MultiTaskCNN(num_classes_mnist=10, num_classes_cifar=10)
        model_dir = "model"
        model = load_saved_model(MultiTaskCNN, num_classes_mnist=10, num_classes_cifar=10, model_dir=model_dir,
                                 device=device)

        if args.mnist:
            print("Testing on random MNIST dataset image:")
            test_on_mnist_images(model, device)

        if args.image_file:
            for img_file in args.image_file:
                print(f"\nPredicting the class for image: {img_file}")
                test_on_external_image(img_file, model, device)


if __name__ == "__main__":
    main()
