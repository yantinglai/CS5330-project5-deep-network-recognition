# Yanting Lai
# CS5330
# Date: November-19-2024
# Description: train_mnist.py with the GaborMNISTNetwork model

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import time
from model import GaborMNISTNetwork


def load_data(batch_size):
    """
    Load the MNIST dataset and return data loaders for training and testing.
    Args:
        batch_size (int): The size of each batch during training/testing.
    Returns:
        tuple: Data loaders for training and testing.
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    train_dataset = datasets.MNIST(
        root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(
        root='./data', train=False, download=True, transform=transform)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


def train_network(model, train_loader, test_loader, epochs, lr):
    """
    Train the network and track performance metrics.
    Args:
        model: The neural network model
        train_loader: DataLoader for training data
        test_loader: DataLoader for test data
        epochs: Number of training epochs
        lr: Learning rate
    Returns:
        tuple: Lists of training and testing metrics
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    print(f"Using device: {device}")

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    train_losses = []
    test_losses = []
    train_accuracies = []
    test_accuracies = []
    total_train_samples = len(train_loader.dataset)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        epoch_start_time = time.time()

        # Training phase
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

            if batch_idx % 100 == 0:
                print(f'Epoch: {epoch+1}/{epochs} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                      f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')

        # Calculate training metrics
        epoch_loss = running_loss / len(train_loader)
        epoch_accuracy = 100. * correct / total
        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_accuracy)

        # Testing phase
        model.eval()
        test_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                test_loss += criterion(output, target).item()
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()

        test_loss /= len(test_loader)
        test_accuracy = 100. * correct / total
        test_losses.append(test_loss)
        test_accuracies.append(test_accuracy)

        epoch_time = time.time() - epoch_start_time
        print(f'\nEpoch {epoch+1} completed in {epoch_time:.2f} seconds')
        print(
            f'Training Loss: {epoch_loss:.4f}, Training Accuracy: {epoch_accuracy:.2f}%')
        print(
            f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%\n')

    # Plot training results
    plt.figure(figsize=(12, 5))

    # Loss plot
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Test Loss')
    plt.legend()
    plt.grid(True)

    # Accuracy plot
    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label='Training Accuracy')
    plt.plot(test_accuracies, label='Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Training and Test Accuracy')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('training_results.png')
    plt.show()

    return train_losses, test_losses, train_accuracies, test_accuracies


def visualize_gabor_filters(model):
    """
    Visualize the Gabor filters used in the first layer.
    Args:
        model: The neural network model
    """
    filters = model.visualize_filters()

    plt.figure(figsize=(15, 3))
    for i in range(10):
        plt.subplot(2, 5, i + 1)
        plt.imshow(filters[i, 0], cmap='gray')
        plt.title(f'Filter {i+1}')
        plt.axis('off')

    plt.suptitle('Gabor Filters')
    plt.tight_layout()
    plt.savefig('gabor_filters.png')
    plt.show()


def save_model(model, file_path):
    """
    Save the trained model.
    Args:
        model: The trained model
        file_path: Path to save the model
    """
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)
    torch.save(model.state_dict(), file_path)
    print(f"Model saved to {file_path}")


def main():
    """
    Main function to run the training process.
    """
    # Set hyperparameters
    batch_size = 64
    epochs = 10
    learning_rate = 0.001
    model_path = './model/mnist_gabor_model.pth'

    # Load data
    print("Loading data...")
    train_loader, test_loader = load_data(batch_size)

    # Initialize model
    print("Initializing model with Gabor filters...")
    model = GaborMNISTNetwork()

    # Visualize Gabor filters
    print("Visualizing Gabor filters...")
    visualize_gabor_filters(model)

    # Train model
    print("Starting training...")
    train_losses, test_losses, train_accuracies, test_accuracies = train_network(
        model, train_loader, test_loader, epochs, learning_rate)

    # Save model
    save_model(model, model_path)


if __name__ == "__main__":
    main()
