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
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.fc1 = nn.Linear(128 * 3 * 3, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.max_pool2d(x, 2)
        x = x.view(-1, 128 * 3 * 3)
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
    for epoch in range(10):
        model.train()  # Set model to training mode
        running_loss = 0.0
        running_corrects = 0
        total_samples = 0

        # Wrap the training loader in tqdm to create a progress bar for each epoch
        with tqdm(train_loader, unit="batch") as tepoch:
            tepoch.set_description(f"Epoch [{epoch + 1}/10]")  # Set progress bar description
            for images, labels in tepoch:
                images, labels = images.to(device), labels.to(device)
                total_samples += labels.size(0)

                # Forward pass
                outputs = model(images)
                loss = criterion(outputs, labels)

                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                running_corrects += (torch.max(outputs, 1)[1] == labels).sum().item()

                # Update progress bar with running loss and accuracy
                tepoch.set_postfix(loss=running_loss / total_samples, acc=running_corrects / total_samples)

        # Calculate training metrics
        train_loss = running_loss / len(train_loader)
        train_acc = running_corrects / total_samples

        # Validate on the test set and step the scheduler based on validation loss
        val_loss, val_acc = validate(model, test_loader, criterion, device)
        scheduler.step(val_loss)

        print(f"Epoch [{epoch + 1}/10], Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")


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
                                 '/Users/indrajeetadityaroy/Desktop/EECE7398/Lab_1/mnist_pngs/mnist_7_label_9.png')  # Replace with an actual image file
    visualize_first_conv_layer(model, example_image, device, output_file="CONV_rslt_mnist.png")
