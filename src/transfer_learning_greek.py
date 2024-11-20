# Yanting Lai
# CS5330
# Date: November-19-2024
# Description: transfer_learning_greek.py

import os
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image


# Greek letter data preprocessing
class GreekTransform:
    def __init__(self):
        pass

    def __call__(self, x):
        x = torchvision.transforms.functional.rgb_to_grayscale(
            x)  # Convert to grayscale
        x = torchvision.transforms.functional.affine(
            x, 0, (0, 0), 36 / 128, 0)  # Scale the image
        x = torchvision.transforms.functional.center_crop(
            x, (28, 28))  # Center crop to 28x28
        # Invert pixel intensities
        return torchvision.transforms.functional.invert(x)


# Improved network structure
class ImprovedNetwork(nn.Module):
    def __init__(self):
        """Initialize the network layers."""
        super(ImprovedNetwork, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool2 = nn.MaxPool2d(2, 2)

        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.pool3 = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(64 * 3 * 3, 128)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 3)

    def forward(self, x):
        """Forward pass for the network."""
        x = self.pool1(torch.relu(self.bn1(self.conv1(x))))
        x = self.pool2(torch.relu(self.bn2(self.conv2(x))))
        x = self.pool3(torch.relu(self.bn3(self.conv3(x))))
        x = x.view(-1, 64 * 3 * 3)  # Flatten feature maps
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.log_softmax(self.fc2(x), dim=1)
        return x


def load_greek_data(data_path, batch_size=5):
    """
    Load the Greek letter dataset with preprocessing.
    Args:
        data_path (str): Path to the Greek letter dataset.
        batch_size (int): Number of samples per batch.

    Returns:
        DataLoader: Greek letter dataset DataLoader.
    """
    augmentation = transforms.Compose([
        transforms.RandomRotation(15),  # Random rotation within 15 degrees
        transforms.RandomAffine(
            degrees=0, translate=(0.1, 0.1)),  # Translation
        transforms.RandomHorizontalFlip(),  # Horizontal flip
        transforms.ToTensor(),
        GreekTransform(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    greek_train = torch.utils.data.DataLoader(
        torchvision.datasets.ImageFolder(
            data_path,
            transform=augmentation
        ),
        batch_size=batch_size,
        shuffle=True
    )
    return greek_train


def train_greek_model(model, dataloader, epochs=100, learning_rate=0.001):
    """
    Train the modified network to classify Greek letters.
    Args:
        model: Modified network.
        dataloader: DataLoader for Greek letters.
        epochs (int): Number of epochs to train.
        learning_rate (float): Learning rate for the optimizer.

    Returns:
        train_loss (list): List of training loss for each epoch.
        train_accuracy (list): List of training accuracy for each epoch.
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    train_loss = []
    train_accuracy = []

    for epoch in range(epochs):
        total_loss = 0.0
        correct = 0
        total = 0

        for images, labels in dataloader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        avg_loss = total_loss / len(dataloader)
        accuracy = 100.0 * correct / total

        train_loss.append(avg_loss)
        train_accuracy.append(accuracy)

        print(
            f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")

    return train_loss, train_accuracy


def plot_training_results(train_loss, train_accuracy):
    """
    Plot training loss and accuracy in separate graphs.

    Args:
        train_loss (list): List of training loss for each epoch.
        train_accuracy (list): List of training accuracy for each epoch.
    """
    # Plot training loss
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, len(train_loss) + 1), train_loss,
             marker='o', color='blue', label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Over Epochs')
    plt.legend()
    plt.grid()
    plt.savefig("training_loss_plot.png")  # Save the training loss plot
    plt.show(block=False)

    # Plot training accuracy
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, len(train_accuracy) + 1), train_accuracy,
             marker='o', color='green', label='Training Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Training Accuracy Over Epochs')
    plt.legend()
    plt.grid()
    # Save the training accuracy plot
    plt.savefig("training_accuracy_plot.png")
    plt.show(block=False)


def predict_greek_letter(model, image_path):
    """
    Predict the class of a Greek letter image using the trained model.
    Args:
        model: Trained model.
        image_path (str): Path to the test image.

    Returns:
        str: Predicted class label.
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        GreekTransform(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    image = Image.open(image_path).convert(
        'RGB')  # Open image as RGB using PIL
    image = transform(image).unsqueeze(0)  # Add batch dimension

    model.eval()
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)

    classes = ['alpha', 'beta', 'gamma']
    return classes[predicted.item()]


def main():
    """
    Main function to perform transfer learning on the Greek letter dataset.
    """
    train_data_path = "/Users/sundri/Desktop/CS5330/Project5/data/greek_train"
    test_data_path = "/Users/sundri/Desktop/CS5330/Project5/data/greek_letter"

    model = ImprovedNetwork()
    print("\nModified Network Structure:\n")
    print(model)

    greek_train = load_greek_data(train_data_path)

    train_loss, train_accuracy = train_greek_model(
        model, greek_train, epochs=100)

    plot_training_results(train_loss, train_accuracy)

    print("\nPredictions on Test Images:")
    for letter in ['α.png', 'β.png', 'γ.png']:
        image_path = os.path.join(test_data_path, letter)
        prediction = predict_greek_letter(model, image_path)
        print(f"Image {letter}: Predicted as {prediction}")


if __name__ == "__main__":
    main()
