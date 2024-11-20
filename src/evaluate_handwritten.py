import os
import cv2
import torch
import matplotlib.pyplot as plt
from model import MyNetwork


def preprocess_image(image_path):
    """
    Preprocess a single handwritten digit image.

    Args:
        image_path (str): Path to the image file.

    Returns:
        torch.Tensor: Preprocessed image tensor ready for the model.
    """
    # Load the image in grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Resize to 28x28
    img_resized = cv2.resize(img, (28, 28), interpolation=cv2.INTER_AREA)

    # Invert the image if necessary (ensure white digits on black background)
    if img_resized.mean() > 127:  # If background is brighter
        img_resized = 255 - img_resized

    # Normalize pixel values to [-1, 1] to match MNIST dataset
    img_normalized = (img_resized / 255.0) * 2 - 1

    # Convert to a tensor and add a batch dimension (1, 1, 28, 28)
    tensor = torch.tensor(
        img_normalized, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

    return tensor


def evaluate_handwritten_digits(model_path, image_folder):
    """
    Evaluate the trained model on handwritten digit images.

    Args:
        model_path (str): Path to the saved model file.
        image_folder (str): Path to the folder containing handwritten digit images.
    """
    # Load the trained model
    model = MyNetwork()
    model.load_state_dict(torch.load(model_path))
    model.eval()  # Set the model to evaluation mode

    # Get all image paths
    image_paths = sorted([os.path.join(image_folder, f)
                         for f in os.listdir(image_folder) if f.endswith(".png")])

    predictions = []
    images_to_display = []

    print("\nResults on handwritten digits:\n")

    # Process each image
    for image_path in image_paths:
        # Preprocess the image
        img_tensor = preprocess_image(image_path)

        # Run through the model
        with torch.no_grad():
            output = model(img_tensor)
            probabilities = torch.nn.functional.softmax(output, dim=1)
            predicted_label = torch.argmax(probabilities).item()

        # Store results
        predictions.append(predicted_label)
        images_to_display.append(cv2.imread(image_path, cv2.IMREAD_GRAYSCALE))

        # Print results
        print(f"Image: {os.path.basename(image_path)}")
        print(f"  Predicted Label: {predicted_label}")
        print(
            f"  Network Outputs: {[f'{val:.2f}' for val in probabilities[0].tolist()]}")
        print()

    # Display the first 9 images in a 3x3 grid with their predictions
    plt.figure(figsize=(10, 10))
    for i, (img, pred) in enumerate(zip(images_to_display[:9], predictions[:9])):
        plt.subplot(3, 3, i + 1)
        plt.imshow(img, cmap="gray")
        plt.title(f"Predicted: {pred}")
        plt.axis('off')

    plt.suptitle("Handwritten Digits with Predictions")
    plt.savefig("handwritten_predictions_grid.png")
    plt.show()


def main():
    """
    Main function to evaluate handwritten digits.
    """
    # Dynamically construct the relative paths
    script_dir = os.path.dirname(__file__)  # Directory of the current script
    # Relative path to the model
    model_path = os.path.join(script_dir, "model", "mnist_model.pth")
    # Relative path to the handwritten digits
    image_folder = os.path.join(script_dir, "../data/roman_digit")

    # Evaluate the handwritten digits
    evaluate_handwritten_digits(model_path, image_folder)


if __name__ == "__main__":
    main()
