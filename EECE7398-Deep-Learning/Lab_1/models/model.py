import torch
import torch.nn as nn


def conv_block(in_channels, out_channels, kernel_size=3, stride=1, padding=1, pool=False):
    layers = [
        nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    ]
    if pool:
        layers.append(nn.MaxPool2d(2))
    return nn.Sequential(*layers)


class CNNClassifier(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()

        # First CONV layer: filter size = 5x5, stride = 1, 32 filters
        self.conv1 = conv_block(in_channels, 32, kernel_size=5, stride=1, padding=2)

        # The rest of the layers use the default kernel size of 3x3, stride 1
        self.conv2 = conv_block(32, 64, pool=True)

        # First Residual Block
        self.res1_conv1 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.res1_conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.res1_relu = nn.ReLU()

        self.conv3 = conv_block(64, 128, pool=True)
        self.conv4 = conv_block(128, 256, pool=True)

        # Second Residual Block
        self.res2_conv1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.res2_conv2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.res2_relu = nn.ReLU()

        # Global average pooling and classifier
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)

        # First Residual Block
        res1 = out
        res1_out = self.res1_conv1(out)
        res1_out = self.res1_relu(res1_out)
        res1_out = self.res1_conv2(res1_out)
        out = self.res1_relu(res1_out + res1)  # Skip connection + ReLU

        out = self.conv3(out)
        out = self.conv4(out)

        # Second Residual Block
        res2 = out
        res2_out = self.res2_conv1(out)
        res2_out = self.res2_relu(res2_out)
        res2_out = self.res2_conv2(res2_out)
        out = self.res2_relu(res2_out + res2)  # Skip connection + ReLU

        # Classifier
        out = self.classifier(out)
        return out