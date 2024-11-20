
# Yanting Lai
# CS5330
# Date: November-19-2024
# Description: GaborMNISTNetwork

import torch
import torch.nn as nn
import numpy as np
import math


class GaborFilter(nn.Module):
    def __init__(self, num_filters=10, kernel_size=5):
        """
        Initialize Gabor filter bank
        Args:
            num_filters: Number of Gabor filters to use
            kernel_size: Size of each filter kernel
        """
        super(GaborFilter, self).__init__()

        # Generate Gabor filter bank
        self.weight = nn.Parameter(
            torch.zeros(num_filters, 1, kernel_size, kernel_size),
            requires_grad=False  # Freeze the parameters
        )

        # Generate different Gabor filters
        for i in range(num_filters):
            # Calculate parameters for each filter
            theta = i * math.pi / num_filters  # Different orientations
            sigma = 3.0  # Standard deviation of Gaussian envelope
            lambda_val = 4.0  # Wavelength of sinusoidal factor
            gamma = 0.5  # Spatial aspect ratio
            psi = 0  # Phase offset

            # Generate kernel
            kernel = self._create_gabor_kernel(
                kernel_size, sigma, theta, lambda_val, gamma, psi
            )
            self.weight.data[i, 0] = torch.from_numpy(kernel)

    def _create_gabor_kernel(self, kernel_size, sigma, theta, lambda_val, gamma, psi):
        """Create a single Gabor filter kernel"""
        # Generate grid coordinates
        x_range = y_range = (kernel_size-1)//2
        (y, x) = np.meshgrid(np.arange(-x_range, x_range + 1),
                             np.arange(-y_range, y_range + 1))

        # Rotation
        x_theta = x * np.cos(theta) + y * np.sin(theta)
        y_theta = -x * np.sin(theta) + y * np.cos(theta)

        # Calculate Gabor response
        gb = np.exp(-.5 * (x_theta**2 + (gamma*y_theta)**2) / sigma**2)
        gb *= np.cos(2*np.pi*x_theta/lambda_val + psi)

        return gb.astype(np.float32)

    def forward(self, x):
        return nn.functional.conv2d(x, self.weight)


class GaborMNISTNetwork(nn.Module):
    def __init__(self):
        """Initialize the network with Gabor filter first layer"""
        super(GaborMNISTNetwork, self).__init__()

        # First layer: Fixed Gabor filter bank
        self.gabor = GaborFilter(num_filters=10, kernel_size=5)

        # Rest of the network (similar to original)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        # Apply Gabor filters
        x = self.pool(torch.relu(self.gabor(x)))
        # Rest of the network
        x = self.dropout(self.pool(torch.relu(self.conv2(x))))
        x = x.view(-1, 320)
        x = torch.relu(self.fc1(x))
        x = torch.log_softmax(self.fc2(x), dim=1)
        return x

    def visualize_filters(self):
        """Visualize the Gabor filters"""
        return self.gabor.weight.data.cpu().numpy()
