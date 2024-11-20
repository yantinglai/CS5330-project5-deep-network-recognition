# train_mnist.py
# Description: Train and save a PyTorch model for MNIST digit recognition with loss plots and visualization.

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from model import MyNetwork  # Import the MyNetwork class from model.py
import time  # Import for logging elapsed time


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


def display_first_six_digits(test_loader):
    """
    Display the first six digits from the test set as images.

    Args:
        test_loader (DataLoader): DataLoader for the test set.
    """
    examples = iter(test_loader)
    images, labels = next(examples)
    images, labels = images[:6], labels[:6]

    # Plot the first six digits in a 2x3 grid
    plt.figure(figsize=(10, 5))
    for i in range(6):
        plt.subplot(2, 3, i + 1)  # 2 rows, 3 columns
        plt.imshow(images[i][0], cmap="gray")  # Show the digit in grayscale
        plt.title(f"Label: {labels[i].item()}")
        plt.axis('off')  # Turn off axis labels for better visualization

    plt.suptitle("First Six Digits from the Test Set")  # Add an overall title
    plt.savefig("first_six_digits.png")  # Save the plot as an image
    plt.show(block=False)  # Non Blocking plot


def calculate_accuracy(model, data_loader):
    """
    Calculate the accuracy of the model on a given dataset.

    Args:
        model (torch.nn.Module): The trained PyTorch model.
        data_loader (DataLoader): DataLoader for the dataset to evaluate.

    Returns:
        float: The accuracy percentage of the model.
    """
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in data_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total


def train_network(model, train_loader, test_loader, epochs, lr):
    """
    Train the PyTorch model and generate a plot for negative log likelihood loss and accuracy.

    Args:
        model (torch.nn.Module): The PyTorch model to be trained.
        train_loader (DataLoader): DataLoader for the training set.
        test_loader (DataLoader): DataLoader for the test set.
        epochs (int): Number of training epochs.
        lr (float): Learning rate for the optimizer.

    Returns:
        tuple: Train losses, test losses, train accuracies, and test accuracies recorded during training.
    """
    print("Training the model...")
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    # Tracking variables
    train_losses = []
    test_losses = []
    train_accuracies = []  # Track train accuracy for each epoch
    test_accuracies = []   # Track test accuracy for each epoch
    total_train_samples = len(train_loader.dataset)

    for epoch in range(epochs):
        epoch_start_time = time.time()  # Record start time for the epoch
        model.train()
        running_loss = 0.0
        total_examples = 0

        print(f"\nEpoch {epoch + 1}/{epochs} - Training started")

        for batch_idx, (images, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # Accumulate loss
            running_loss += loss.item()

            # Update total examples seen
            total_examples += images.size(0)

            # Record train loss for this batch
            train_losses.append(
                (total_examples + epoch * total_train_samples, loss.item()))

            # Print progress for every 100 batches
            if batch_idx % 100 == 0 or batch_idx == len(train_loader) - 1:
                print(f"  Batch {batch_idx + 1}/{len(train_loader)} - "
                      f"Examples Seen: {total_examples} - "
                      f"Current Loss: {loss.item():.4f}")

        # Compute accuracy for the training set
        train_correct = 0
        train_total = 0
        with torch.no_grad():
            for train_images, train_labels in train_loader:
                train_outputs = model(train_images)
                _, predicted = torch.max(train_outputs, 1)
                train_total += train_labels.size(0)
                train_correct += (predicted == train_labels).sum().item()
        train_accuracy = 100 * train_correct / train_total
        train_accuracies.append(train_accuracy)

        # Compute test loss and accuracy after each epoch
        model.eval()
        test_loss = 0.0
        test_correct = 0
        test_total = 0
        with torch.no_grad():
            for test_images, test_labels in test_loader:
                test_outputs = model(test_images)
                loss = criterion(test_outputs, test_labels)
                test_loss += loss.item()

                _, predicted = torch.max(test_outputs, 1)
                test_total += test_labels.size(0)
                test_correct += (predicted == test_labels).sum().item()
        avg_test_loss = test_loss / len(test_loader)  # Average test loss
        test_losses.append(
            (total_examples + epoch * total_train_samples, avg_test_loss))
        test_accuracy = 100 * test_correct / test_total
        test_accuracies.append(test_accuracy)

        # Print epoch summary
        epoch_end_time = time.time()  # Record end time
        elapsed_time = epoch_end_time - epoch_start_time
        print(f"Epoch {epoch + 1} completed in {elapsed_time:.2f}s")
        print(f"  Avg Train Loss: {running_loss / len(train_loader):.4f}")
        print(f"  Avg Test Loss: {avg_test_loss:.4f}")
        print(
            f"  Train Accuracy: {train_accuracy:.2f}%, Test Accuracy: {test_accuracy:.2f}%")

    # Plot the loss
    plt.figure(figsize=(10, 6))

    # Train Loss: Blue line
    train_x, train_y = zip(*train_losses)
    plt.plot(train_x, train_y, label="Train Loss", color="blue", linestyle="-")

    # Test Loss: Red dots
    test_x, test_y = zip(*test_losses)
    plt.scatter(test_x, test_y, label="Test Loss", color="red", marker="o")

    # Configure plot
    plt.xlabel("Number of Training Examples Seen")
    plt.ylabel("Negative Log Likelihood Loss")
    plt.xticks(range(0, max(train_x) + 1, 25000))  # X-axis interval: 25,000
    plt.yticks([i * 0.5 for i in range(6)])  # Y-axis interval: 0.5
    plt.legend()
    plt.grid(True)

    # Save and display the plot
    plt.savefig("loss_plot_combined.png")  # Save as PNG
    plt.show()  # Display the plot

    # Plot the accuracy
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, epochs + 1), train_accuracies,
             label="Train Accuracy", color="blue", marker="o")
    plt.plot(range(1, epochs + 1), test_accuracies,
             label="Test Accuracy", color="red", marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title("Training and Testing Accuracy")
    plt.legend()
    plt.grid(True)
    plt.savefig("accuracy_plot.png")  # Save the accuracy plot
    plt.show()

    return train_losses, test_losses, train_accuracies, test_accuracies


def save_model(model, file_path):
    """
    Save the trained model to the specified file path.

    Args:
        model (torch.nn.Module): The trained PyTorch model.
        file_path (str): Path to save the model file.
    """
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)
    torch.save(model.state_dict(), file_path)
    print(f"Model saved to {file_path}")


def main():
    """
    Main function to load data, train the model, and save the results.
    """
    batch_size = 256
    epochs = 5
    lr = 0.001
    model_path = "./model/mnist_model.pth"

    # Load data
    train_loader, test_loader = load_data(batch_size)

    # Display the first six digits from the test set
    # display_first_six_digits(test_loader)

    # Initialize the model
    model = MyNetwork()
    print("Model Initialized successfully!")

    # Train the model and get loss data
    train_losses, test_losses = train_network(
        model, train_loader, test_loader, epochs, lr)

    # Save the trained model
    save_model(model, model_path)


if __name__ == "__main__":
    main()
