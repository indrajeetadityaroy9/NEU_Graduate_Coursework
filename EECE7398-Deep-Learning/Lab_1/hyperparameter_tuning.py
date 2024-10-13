import optuna
import torch
import torch.nn as nn
import torch.optim as optim
from models.model import CNNClassifier
from utility.data_loader import  load_dataset
from torchvision import datasets, transforms as T


def objective(trial):
    # Expanded hyperparameter space
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
    batch_size = trial.suggest_int('batch_size', 32, 128)
    conv1_out_channels = trial.suggest_int('conv1_out_channels', 32, 64)
    conv2_out_channels = trial.suggest_int('conv2_out_channels', 64, 128)
    conv3_out_channels = trial.suggest_int('conv3_out_channels', 128, 256)
    dropout_rate = trial.suggest_float('dropout_rate', 0.2, 0.5)
    weight_decay = trial.suggest_float('weight_decay', 1e-5, 1e-2, log=True)

    # Load dataset
    dataset_name = 'cifar'
    train_loader, test_loader, input_channels, num_classes = load_dataset(dataset_name, batch_size)

    # Define model
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = CNNClassifier(in_channels=input_channels, num_classes=num_classes).to(device)

    # Loss, optimizer, and scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)  # Learning rate decay

    # Training loop (fewer epochs for faster tuning)
    num_epochs = 10
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        scheduler.step()  # Adjust the learning rate

    # Test loop
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    return accuracy


# Run Optuna study with more trials
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50)

print(f"Best hyperparameters: {study.best_trial.params}")

