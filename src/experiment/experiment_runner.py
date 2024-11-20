import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from itertools import product
import sys

# Define the network dynamically


class CustomNetwork(nn.Module):
    def __init__(self, num_conv_layers, num_filters, dropout_rate):
        super(CustomNetwork, self).__init__()
        layers = []
        in_channels = 1

        # Add convolutional layers
        for _ in range(num_conv_layers):
            layers.append(nn.Conv2d(in_channels, num_filters,
                          kernel_size=3, padding=1))
            layers.append(nn.ReLU())
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            in_channels = num_filters

        # Flatten
        self.flatten = nn.Flatten()

        # Fully connected layers
        fc_input_size = num_filters * (28 // (2 ** num_conv_layers))**2
        self.fc = nn.Sequential(
            nn.Linear(fc_input_size, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 10)
        )

        self.conv_layers = nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x


# Data loader
def load_data(batch_size):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    train_data = datasets.FashionMNIST(
        root="./data", train=True, download=True, transform=transform)
    val_data = datasets.FashionMNIST(
        root="./data", train=False, download=True, transform=transform)

    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(
        val_data, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader


# Training function
def train_model(model, train_loader, val_loader, epochs, learning_rate, device):
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    train_loss, val_accuracy = [], []

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        train_loss.append(avg_loss)

        # Validation
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        accuracy = 100 * correct / total
        val_accuracy.append(accuracy)

        print(
            f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}, Validation Accuracy: {accuracy:.2f}%")

    return train_loss, val_accuracy


# Plot results
def plot_results(param, values, train_loss, val_accuracy):
    plt.figure(figsize=(10, 5))
    plt.plot(values, train_loss, label="Training Loss", marker='o')
    plt.plot(values, val_accuracy, label="Validation Accuracy", marker='o')
    plt.xlabel(param)
    plt.ylabel("Metrics")
    plt.title(f"Effect of {param} on Performance")
    plt.legend()
    plt.grid()
    plt.savefig(f"{param}_experiment.png")
    plt.show(block=False)


# Main experiment runner
def main():
    print("Testing print output to log file.", file=sys.stdout)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader = load_data(batch_size=64)

    # Parameters to explore
    num_conv_layers = [2, 3, 4]
    num_filters = [16, 32, 64]
    dropout_rates = [0.2, 0.5, 0.7]

    # Experiment dimensions
    epochs = 10
    learning_rate = 0.001

    results = []

    for conv_layers, filters, dropout in product(num_conv_layers, num_filters, dropout_rates):
        print(
            f"\nExperiment: Conv Layers={conv_layers}, Filters={filters}, Dropout={dropout}")
        model = CustomNetwork(conv_layers, filters, dropout)
        train_loss, val_accuracy = train_model(
            model, train_loader, val_loader, epochs, learning_rate, device)

        results.append({
            "conv_layers": conv_layers,
            "filters": filters,
            "dropout": dropout,
            "final_loss": train_loss[-1],
            "final_accuracy": val_accuracy[-1]
        })

        # Plot individual result
        plot_results(f"Conv={conv_layers}_Filters={filters}_Dropout={dropout}", list(
            range(1, epochs + 1)), train_loss, val_accuracy)

    # Save results
    torch.save(results, "experiment_results.pth")


if __name__ == "__main__":
    main()
