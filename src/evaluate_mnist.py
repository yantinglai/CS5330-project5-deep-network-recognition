# Yanting Lai
# CS5330
# Date: November-19-2024
# Description: evaluate_mnist.py

import os
import torch
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from model import MyNetwork


def load_test_data():
    """
    Load the MNIST test dataset.

    Returns:
        DataLoader: DataLoader for the test set.
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    test_dataset = datasets.MNIST(
        root='./data', train=False, download=True, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=10, shuffle=False)
    return test_loader


def evaluate_model(model_path, test_loader):
    """
    Evaluate the model on the first 10 examples of the test set.

    Args:
        model_path (str): Path to the saved model file.
        test_loader (DataLoader): DataLoader for the test set.
    """
    # Load the model
    model = MyNetwork()
    model.load_state_dict(torch.load(model_path))
    model.eval()  # Set the model to evaluation mode

    # Get the first batch of 10 images
    test_iter = iter(test_loader)
    images, labels = next(test_iter)

    # Pass the images through the network
    outputs = model(images)
    probabilities = torch.nn.functional.softmax(outputs, dim=1)

    # Process and display results
    predictions = torch.argmax(probabilities, dim=1)

    print("Results for the first 10 test examples:\n")
    for i in range(10):
        output_values = probabilities[i].detach().numpy()
        rounded_values = [f"{val:.2f}" for val in output_values]
        predicted_label = predictions[i].item()
        true_label = labels[i].item()

        print(f"Image {i + 1}:")
        print(f"  Network Outputs: {rounded_values}")
        print(
            f"  Predicted Label: {predicted_label}, True Label: {true_label}")
        print()

    # Plot the first 9 images in a 3x3 grid with their predictions
    plt.figure(figsize=(10, 10))
    for i in range(9):  # First 9 images
        plt.subplot(3, 3, i + 1)
        plt.imshow(images[i][0], cmap="gray")
        plt.title(f"Predicted: {predictions[i].item()}")
        plt.axis('off')

    plt.suptitle("First 9 Test Examples with Predictions")
    plt.savefig("predictions_grid.png")  # Save the plot
    plt.show()


def main():
    """
    Main function to evaluate the trained MNIST model.
    """
    # Dynamically construct the relative path to the model
    script_dir = os.path.dirname(__file__)  # Directory of the current script
    # Relative path to the model
    model_path = os.path.join(script_dir, "model", "mnist_model.pth")

    # Load test data and evaluate
    test_loader = load_test_data()
    evaluate_model(model_path, test_loader)


if __name__ == "__main__":
    main()
