from torchvision import datasets, transforms as T
from torch.utils.data import DataLoader


def load_dataset(dataset_name, batch_size):
    if dataset_name == 'mnist':

        train_dataset = datasets.MNIST(root='./data', train=True, download=True)
        test_dataset = datasets.MNIST(root='./data', train=False, download=True)
        input_channels = 1  # Grayscale

    elif dataset_name == 'cifar':
        # Loading CIFAR10 dataset with transformations
        train_dataset = datasets.CIFAR10(root='./data', train=True, download=True,)
        test_dataset = datasets.CIFAR10(root='./data', train=False, download=True,)
        input_channels = 3  # RGB

    num_classes = len(train_dataset.classes)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader, input_channels, num_classes
