import torch.nn as nn

from boltzmann_experimentation.config.settings import (
    general_settings as g,
)


class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)  # Batch norm for conv1 output
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)  # Batch norm for conv2 output
        self.fc1 = nn.Linear(64 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, 10)  # CIFAR10 has 10 classes
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        x = self.relu(self.pool(self.conv1(x)))
        if g.model_kwargs.get("batch_norm"):
            x = self.bn1(x)
        x = self.relu(self.pool(self.conv2(x)))
        if g.model_kwargs.get("batch_norm"):
            x = self.bn2(x)
        x = x.view(-1, 64 * 8 * 8)  # Flatten
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
