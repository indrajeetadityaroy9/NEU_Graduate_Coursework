import argparse
import torch
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from models.model import CNNClassifier
from utility.data_loader import load_dataset
from utility.model_trainer import train_model
from torch.optim.lr_scheduler import StepLR
from PIL import Image
from torchvision import transforms


def normalize_feature_map(feature_map):
    """Normalize the feature map for better visualization."""
    feature_map_min = feature_map.min()
    feature_map_max = feature_map.max()
    feature_map = (feature_map - feature_map_min) / (feature_map_max - feature_map_min + 1e-8)  # Avoid division by zero
    return feature_map


def visualize_conv_layer(image, model, device, save_path="CONV_rslt_cifar.png"):
    """Visualize the output of the first CONV layer."""
    model.eval()

    # Hook to capture the output of the first conv layer
    activation = {}

    def hook_fn(module, input, output):
        activation['conv1'] = output

    # Register the hook to the first conv layer (Conv2d inside conv_block)
    hook = model.conv1[0].register_forward_hook(hook_fn)

    # Forward pass through the model
    with torch.no_grad():
        _ = model(image.to(device))  # Perform a forward pass to trigger the hook

    # Remove the hook after obtaining the activation
    hook.remove()

    # Get the conv1 output (feature maps), expected shape (1, 32, H, W)
    conv1_output = activation['conv1'].squeeze(
        0).cpu().numpy()  # Remove batch dimension, convert to numpy array (32, H, W)

    # Normalize each feature map for better visualization
    conv1_output = np.array([normalize_feature_map(fm) for fm in conv1_output])

    # Plot the filters' output
    num_filters = conv1_output.shape[0]  # Should be 32 filters
    fig, axes = plt.subplots(4, 8, figsize=(16, 8))  # Create a grid of 4x8 for 32 filters
    for i, ax in enumerate(axes.flat):
        if i < num_filters:
            ax.imshow(conv1_output[i], cmap='gray')  # Visualize each normalized filter output in grayscale
        ax.axis('off')  # Turn off the axis

    # Save the visualization
    plt.subplots_adjust(wspace=0.1, hspace=0.1)  # Reduce spacing between images for a cleaner look
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    print(f"Visualization saved as {save_path}")


test_transforms = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=means, std=stds)
])

# When visualizing the feature maps, make sure you are using test_transforms
def predict_image(image_path, model, device, transform):
    """Predict the class of an input image using a trained model."""
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0).to(device)

    # Perform the prediction
    model.eval()
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)

    return predicted.item(), image


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train or Test CNN on different datasets")
    parser.add_argument('command', choices=['train', 'test'], help="Command to execute: train or test")
    parser.add_argument('--dataset', choices=['mnist', 'cifar'], required=True, help="Dataset to use: mnist or cifar")
    parser.add_argument('--epochs', type=int, default=500, help="Number of training epochs")
    parser.add_argument('--batch_size', type=int, default=64, help="Batch size")
    parser.add_argument('--log_interval', type=int, default=50, help="Interval to log training/testing status")
    parser.add_argument('--checkpoint', type=str, help="Path to the model checkpoint for testing")
    parser.add_argument('--image', type=str, help="Path to the image to be predicted (required for test command)")
    parser.add_argument('--visualize', action='store_true', help="Visualize the output of the first CONV layer")

    args = parser.parse_args()

    # Load dataset and model configurations
    train_loader, test_loader, input_channels, num_classes = load_dataset(args.dataset, args.batch_size)
    model = CNNClassifier(in_channels=input_channels, num_classes=num_classes)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Training mode
    if args.command == 'train':
        # Loss function and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=1e-3)

        # Define the learning rate scheduler
        scheduler = StepLR(optimizer, step_size=15, gamma=0.5)

        # Call the training function
        train_model(model, train_loader, test_loader, criterion, optimizer, scheduler, epochs=args.epochs)

    # Testing mode with optional visualization
    elif args.command == 'test':
        if not args.checkpoint or not args.image:
            print("Error: --checkpoint and --image arguments are required for testing.")
            exit(1)

        # Load the model checkpoint
        model.load_state_dict(torch.load(args.checkpoint))
        model.eval()

        # Define image transformation based on dataset
        transform = transforms.Compose([
            transforms.Resize((32, 32)) if args.dataset == 'cifar' else transforms.Resize((28, 28)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)) if args.dataset == 'mnist' else transforms.Normalize((0.5, 0.5, 0.5),
                                                                                                      (0.5, 0.5, 0.5))
        ])

        # Predict the class of the input image
        predicted_class, image_tensor = predict_image(args.image, model, device, transform)
        print(f"Predicted class for {args.image}: {predicted_class}")

        # Visualize the output of the first CONV layer if requested
        if args.visualize:
            visualize_conv_layer(image_tensor, model, device, save_path="CONV_rslt_cifar.png")
