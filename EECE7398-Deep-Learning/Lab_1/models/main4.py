import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
from PIL import Image
import os
import matplotlib.pyplot as plt

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# MNIST dataset and DataLoader
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)

batch_size = 64
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)


# Define CNN model
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2)
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
        self.fc3 = nn.Linear(128, 10)

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


# Initialize model, criterion, optimizer, and scheduler
model = CNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0005)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2, verbose=True)


# Function to evaluate on the validation set
def validate(model, test_loader, criterion, device):
    model.eval()  # Set model to evaluation mode
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    val_loss /= len(test_loader)
    accuracy = correct / total
    return val_loss, accuracy


# Modified training function to output all metrics (train loss, train acc, val loss, val acc) on one line
def train_with_plateau_scheduler(model, train_loader, test_loader, criterion, optimizer, scheduler, device):
    epochs = 10
    # Print header for the results
    print(f"{'Loop':>6}{'Train Loss':>15}{'Train Acc %':>20}{'Test Loss':>20}{'Test Acc %':>20}")

    for epoch in range(epochs):
        model.train()
        total_correct = 0
        total_samples = 0
        running_loss = 0.0

        # Training loop with tqdm progress bar
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

        # Validation phase
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with tqdm(test_loader, unit="batch", leave=False) as test_bar:
            test_bar.set_description(f"Epoch [{epoch + 1}/{epochs}] - Testing")
            with torch.no_grad():
                for batch_idx, (inputs, labels) in enumerate(test_bar):
                    inputs, labels = inputs.to(device), labels.to(device)

                    # Forward pass
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()

                    # Track correct predictions
                    _, predicted = torch.max(outputs, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

                    # Update progress bar
                    test_bar.set_postfix({
                        'Loss': val_loss / (batch_idx + 1),
                        'Accuracy': 100. * correct / total
                    })

        val_loss /= len(test_loader)
        val_acc = 100. * correct / total

        # Print epoch summary
        print(f"{epoch:>2}/{epochs:<1}{train_loss:>15f}{train_acc:>20f}{val_loss:>20f}{val_acc:>20f}")

        # Scheduler step (depends on the type of scheduler)
        scheduler.step(val_loss)


# Function to predict for a batch of images in a directory
def main(model, image_directory, device):
    # Ensure the model is in evaluation mode
    model.eval()

    # Process each image in the directory
    for filename in os.listdir(image_directory):
        if filename.endswith(".png"):  # Only process PNG files
            image_path = os.path.join(image_directory, filename)
            predicted_label = test_single_image(model, image_path, device)
            print(f"Image: {filename}, Predicted Label: {predicted_label}")


# Test single image function
def test_single_image(model, image_path, device):
    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    image = Image.open(image_path)
    image = transform(image).unsqueeze(0)  # Add batch dimension (1, 1, 28, 28)
    image = image.to(device)

    model.eval()  # Set model to evaluation mode
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)
        predicted_label = predicted.item()

    return predicted_label


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

    plt.tight_layout()
    plt.savefig(output_file)  # Save the resulting figure as a PNG file
    print(f"Feature maps saved as {output_file}")


# Entry point of the script
if __name__ == '__main__':
    # Train the model
    train_with_plateau_scheduler(model, train_loader, test_loader, criterion, optimizer, scheduler, device)

    # Specify the directory containing the test images
    image_directory = '/Users/indrajeetadityaroy/Desktop/EECE7398/Lab_1/mnist_pngs'  # Replace with your actual directory

    # Call the main function to predict labels for all images in the directory
    main(model, image_directory, device)

    example_image = os.path.join(image_directory,
                                 '/mnist_pngs/mnist_7_label_9.png')  # Replace with an actual image file
    visualize_first_conv_layer(model, example_image, device, output_file="../CONV_rslt_mnist.png")
