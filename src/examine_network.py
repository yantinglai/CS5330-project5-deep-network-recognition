# Yanting Lai
# CS5330
# Date: November-19-2024
# Description: examine_network.py

import os
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from model import MyNetwork


def analyze_first_layer(model):
    """
    Analyze and visualize the first layer filters of the model.

    Args:
        model: Trained PyTorch model.
    """
    # Access the first layer filters
    first_layer_weights = model.conv1.weight

    # Print the weights and their shape
    print("\nFirst Layer Weights:\n")
    print(first_layer_weights)
    print("\nShape of First Layer Weights:")
    print(first_layer_weights.shape)

    # Visualize the filters
    num_filters = first_layer_weights.shape[0]
    fig, axes = plt.subplots(3, 4, figsize=(10, 8))  # 3x4 grid
    fig.suptitle("Visualization of First Layer Filters (conv1)")

    for i in range(num_filters):
        row, col = divmod(i, 4)
        ax = axes[row, col]
        # Extract ith filter
        filter_weights = first_layer_weights[i, 0].detach().numpy()
        ax.imshow(filter_weights, cmap='viridis')  # Use viridis colormap
        ax.set_title(f"Filter {i}")
        ax.set_xticks([])
        ax.set_yticks([])

    # Remove unused subplots
    for i in range(num_filters, 12):
        row, col = divmod(i, 4)
        fig.delaxes(axes[row, col])

    plt.tight_layout()
    plt.show()


def apply_filters_to_image(image, filters):
    """
    Apply the filters to the input image using OpenCV's filter2D.

    Args:
        image (numpy.ndarray): The input image.
        filters (torch.Tensor): Filters from the first layer (shape: [num_filters, 1, kernel_h, kernel_w]).

    Returns:
        List[numpy.ndarray]: List of images filtered by each filter.
    """
    filtered_images = []
    for i in range(filters.shape[0]):
        kernel = filters[i, 0].detach().numpy()  # Extract the ith filter
        filtered_image = cv2.filter2D(image, -1, kernel)  # Apply the filter
        filtered_images.append(filtered_image)
    return filtered_images


def visualize_filters_and_effects_four_columns(filters, filtered_images):
    """
    Visualize the filters and their corresponding effects in a 4-column layout:
    Filter | Effect | Filter | Effect. No titles to keep the layout compact.

    Args:
        filters (torch.Tensor): Filters from the first layer.
        filtered_images (List[numpy.ndarray]): Filtered images after applying filters.
    """
    num_filters = filters.shape[0]
    cols = 4  # Fixed 4 columns (Filter | Effect | Filter | Effect)
    rows = (num_filters + 1) // 2  # Two filters and their effects per row

    fig, axes = plt.subplots(rows, cols, figsize=(12, 3 * rows))

    for i in range(num_filters):
        row = i // 2
        col = (i % 2) * 2  # 0 or 2 for Filter | Effect alignment

        # Plot the filter
        axes[row, col].imshow(filters[i, 0].detach().numpy(), cmap='gray')
        axes[row, col].axis('off')

        # Plot the filtered image
        axes[row, col + 1].imshow(filtered_images[i], cmap='gray')
        axes[row, col + 1].axis('off')

    # Hide unused subplots if the total rows * cols exceed the number of filters
    for i in range(rows * cols):
        if i >= num_filters * 2:  # Each filter and effect pair occupies 2 spots
            row, col = divmod(i, cols)
            axes[row, col].axis('off')

    plt.tight_layout()
    plt.show()


def main():
    """
    Main function to analyze the trained network and apply filters to a sample image.
    """
    # Dynamically construct the relative path to the model
    script_dir = os.path.dirname(__file__)
    model_path = os.path.join(script_dir, "model", "mnist_model.pth")

    # Load the trained model
    model = MyNetwork()
    # model.load_state_dict(torch.load(model_path))
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()  # Set the model to evaluation mode

    # Load the MNIST dataset and get the first training example
    transform = transforms.Compose([transforms.ToTensor()])
    mnist_train = datasets.MNIST(
        root='./data', train=True, download=True, transform=transform)
    first_image, _ = mnist_train[0]  # Get the first training example
    first_image = first_image.squeeze(0).numpy()  # Convert to numpy array

    # Apply filters to the image
    with torch.no_grad():
        filters = model.conv1.weight
        filtered_images = apply_filters_to_image(first_image, filters)

    # Visualize filters and their effects in a 4-column layout without titles
    visualize_filters_and_effects_four_columns(filters, filtered_images)


if __name__ == "__main__":
    main()
