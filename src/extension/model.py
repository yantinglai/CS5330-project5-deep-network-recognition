import torch
import torch.nn as nn


class MyNetwork(nn.Module):
    def __init__(self):
        """Initialize the network layers."""
        super(MyNetwork, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        """Forward pass for the network."""
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.dropout(self.pool(torch.relu(self.conv2(x))))
        x = x.view(-1, 320)  # Flatten
        x = torch.relu(self.fc1(x))
        x = torch.log_softmax(self.fc2(x), dim=1)
        return x
