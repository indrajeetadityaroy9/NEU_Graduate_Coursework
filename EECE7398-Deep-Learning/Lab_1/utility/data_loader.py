from torchvision import datasets, transforms as T
from torch.utils.data import DataLoader


def load_dataset(dataset_name, batch_size):
    if dataset_name == 'mnist':
        means = (0.1307,)
        stds = (0.3081,)

        train_transforms = T.Compose([
            T.RandomRotation(10),
            T.ToTensor(),
            T.Normalize(mean=means, std=stds)
        ])

        test_transforms = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=means, std=stds)
        ])

        train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=train_transforms)
        test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=test_transforms)
        input_channels = 1  # Grayscale

    elif dataset_name == 'cifar':
        train_dataset_temp = datasets.CIFAR10(root='./data', train=True, download=True)
        # Calculate mean and std of the CIFAR dataset for normalization
        means = train_dataset_temp.data.mean(axis=(0, 1, 2)) / 255
        stds = train_dataset_temp.data.std(axis=(0, 1, 2)) / 255
        # CIFAR10 transformations with AutoAugment for training
        train_transforms = T.Compose([
            T.AutoAugment(T.AutoAugmentPolicy.CIFAR10),  # AutoAugment for training
            T.ToTensor(),
            T.Normalize(mean=means, std=stds)
        ])
        # CIFAR10 transformations for test (no augmentation, only normalization)
        test_transforms = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=means, std=stds)
        ])
        # Loading CIFAR10 dataset with transformations
        train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transforms)
        test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transforms)
        input_channels = 3  # RGB

    num_classes = len(train_dataset.classes)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader, input_channels, num_classes
