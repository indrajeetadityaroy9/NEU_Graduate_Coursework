# -------------------- Imports --------------------
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


# -------------------- Model Architecture --------------------

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

        # Residual block 1
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

        # Residual block 2
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


# -------------------- Training and Evaluation --------------------

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


# -------------------- Data Handling --------------------

def load_dataset(dataset_name, batch_size):
    if dataset_name == 'mnist':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
        input_channels = 1
    elif dataset_name == 'cifar':
        transform_train = transforms.Compose([
            transforms.RandomHorizontalFlip(),  # Horizontal flips for CIFAR
            transforms.RandomCrop(32, padding=4),  # Random cropping
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalization for RGB images
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalization for RGB images
        ])
        train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
        test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
        input_channels = 3
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=4)

    num_classes = len(train_dataset.classes)
    return train_loader, test_loader, input_channels, num_classes


def preprocess_image(image_path, dataset_name):
    if dataset_name == "mnist":
        image_size = (28, 28)
        transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        image = Image.open(image_path).convert('L')

    elif dataset_name == "cifar":
        image_size = (32, 32)
        transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        image = Image.open(image_path).convert('RGB')

    image = transform(image).unsqueeze(0)
    return image


def classify_image(model, image_tensor, device):
    image_tensor = image_tensor.to(device)  # Move the image tensor to the correct device
    with torch.no_grad():
        output = model(image_tensor)  # Forward pass
        _, predicted_class = torch.max(output, 1)  # Get the class with the highest score
    return predicted_class.item()


def get_mnist_label(class_idx):
    mnist_labels = {0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9'}
    return mnist_labels.get(class_idx, "Unknown")


def get_cifar_label(class_idx):
    cifar_labels = {0: 'airplane', 1: 'automobile', 2: 'bird', 3: 'cat', 4: 'deer', 5: 'dog', 6: 'frog', 7: 'horse',
                    8: 'ship', 9: 'truck'}
    return cifar_labels.get(class_idx, "Unknown")


def load_saved_model(model_class, num_classes, in_channels, model_dir, dataset_name, device):
    model_filename = f"{dataset_name}_best_acc_model.pth"
    model_path = os.path.join(model_dir, model_filename)

    # Load the saved model's state_dict
    state_dict = torch.load(model_path, map_location=device)

    # Remap the keys based on the dataset name (mnist or cifar)
    remapped_state_dict = {}
    if dataset_name == "mnist":
        for key in state_dict:
            new_key = key.replace("conv1_mnist", "conv1").replace("classifier_mnist", "classifier")
            remapped_state_dict[new_key] = state_dict[key]
    elif dataset_name == "cifar":
        for key in state_dict:
            new_key = key.replace("conv1_cifar", "conv1").replace("classifier_cifar", "classifier")
            remapped_state_dict[new_key] = state_dict[key]

    # Initialize the model and load the remapped state_dict
    model = model_class(in_channels=in_channels, num_classes=num_classes)
    model.load_state_dict(remapped_state_dict)
    model.to(device)
    model.eval()
    return model


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

    return mnist_pred_class, cifar_pred_class, mnist_label, cifar_label


# -------------------- Visualization --------------------

def hook_fn(module, input, output):
    global conv_output
    conv_output = output


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

    # Ensure 6 rows and 8 columns grid (even if some are empty)
    fig, axes = plt.subplots(6, 8, figsize=(12, 9))  # Create a 6x8 grid for up to 48 filters
    axes = axes.flatten()  # Flatten axes for easier indexing

    for i in range(48):  # Loop over 48 positions (even though we might have fewer filters)
        if i < num_filters:
            # Display filter activation using grayscale colormap
            axes[i].imshow(activations[i].detach().numpy(), cmap='gray')
            axes[i].set_title(f'Filter {i + 1}')
        else:
            # Hide axes if there are no more filters to display
            axes[i].axis('off')

        # Turn off axis for each subplot
        axes[i].axis('off')

    # Save the visualization
    plt.suptitle(f'Activations of First CONV Layer ({dataset_name})')
    plt.tight_layout()
    plt.savefig(output_filename)
    plt.close()
    print(f"Visualization saved as {output_filename}")


# -------------------- Save Images from Dataset --------------------

def save_images_from_dataset(dataset_name, output_dir, num_images=10):
    """
    Save images from a given dataset to a specified directory.

    Parameters:
    - dataset_name: Name of the dataset ('mnist' or 'cifar').
    - output_dir: Directory where the images will be saved.
    - num_images: Number of images to save (default 10).
    """

    # Create the directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if dataset_name == 'mnist':
        # Load MNIST dataset
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

        for i in range(min(num_images, len(dataset))):
            img, label = dataset[i]
            # Convert the tensor image back to a PIL Image
            img = transforms.ToPILImage()(img)
            img = img.convert('L')  # Convert to grayscale for MNIST
            # Save as PNG
            img.save(os.path.join(output_dir, f'mnist_{i}_label_{label}.png'))
        print(f"Saved {num_images} MNIST images to {output_dir}")

    elif dataset_name == 'cifar':
        # Load CIFAR-10 dataset
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

        for i in range(min(num_images, len(dataset))):
            img, label = dataset[i]
            # Convert the tensor image back to a PIL Image
            img = transforms.ToPILImage()(img)
            # Save as PNG
            img.save(os.path.join(output_dir, f'cifar_{i}_label_{label}.png'))
        print(f"Saved {num_images} CIFAR-10 images to {output_dir}")

    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")


# -------------------- Command Line Interface and Main --------------------

def main():
    epilog = """Usage examples:
      python3 CNNclassify.py train --mnist       Train the model using the MNIST dataset
      python3 CNNclassify.py train --cifar       Train the model using the CIFAR-10 dataset
      python3 CNNclassify.py test car.png        Test the model using 'car.png'
      python3 CNNclassify.py save --mnist --output_dir ./mnist_pngs --num_images 10   Save 10 MNIST images as PNG files
      python3 CNNclassify.py save --cifar --output_dir ./cifar_pngs --num_images 10   Save 10 CIFAR-10 images as PNG files
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

    # Test command
    elif args.command == 'test':
        if not args.image_file:
            print("Error: 'test' command requires at least one image file.", file=sys.stderr)
            print("Use --help for more information.")
            sys.exit(1)

        for img_file in args.image_file:
            print(f"Predicting the class for image: {img_file}")
            model_dir = "model"
            image_path = img_file
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

            mnist_pred_class, cifar_pred_class, mnist_label, cifar_label = dynamic_test_classification(
                CNNClassifier, num_classes_mnist=10, num_classes_cifar=10, model_dir=model_dir, image_path=image_path,
                device=device)

            print(f"The MNIST model predicted class: {mnist_pred_class}")
            print(f"The CIFAR-10 model predicted class: {cifar_pred_class}")
            print(f"MNIST model predicted class: {mnist_label}")
            print(f"CIFAR-10 model predicted class: {cifar_label}")

    # Save images command
    elif args.command == 'save':
        if not args.mnist and not args.cifar:
            print("Error: 'save' command requires either --mnist or --cifar argument.", file=sys.stderr)
            print("Use --help for more information.")
            sys.exit(1)

        dataset = 'mnist' if args.mnist else 'cifar'
        output_dir = args.output_dir
        num_images = args.num_images
        print(f"Saving {num_images} images from {dataset} dataset to {output_dir}")
        save_images_from_dataset(dataset, output_dir, num_images=num_images)


if __name__ == '__main__':
    main()
