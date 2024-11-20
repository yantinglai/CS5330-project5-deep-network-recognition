import cv2
import numpy as np
import torch
from torchvision import transforms
from model import MyNetwork  # Replace with your trained model class


def load_model(model_path):
    model = MyNetwork()
    model.load_state_dict(torch.load(
        model_path, map_location=torch.device('cpu')))
    model.eval()
    return model


def preprocess_roi(roi):
    # Enhanced preprocessing pipeline
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Grayscale(),
        transforms.Resize((28, 28)),
        # Add padding to help with thin digits like 1
        transforms.Pad(padding=2, fill=0),
        transforms.Resize((28, 28)),  # Resize back to expected input size
        transforms.ToTensor(),
        # Adjust normalization values if needed
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    return transform(roi).unsqueeze(0)


def predict_digit(model, roi):
    with torch.no_grad():
        output = model(roi)
        probabilities = torch.nn.functional.softmax(output, dim=1)
        prob_value, predicted = torch.max(probabilities, 1)

        # Only return prediction if confidence is high enough
        if prob_value.item() > 0.7:  # Adjust confidence threshold as needed
            return predicted.item(), prob_value.item()
        return None, None


def enhance_digit_region(roi):
    # Apply advanced image processing to improve digit clarity
    # Increase contrast
    roi = cv2.convertScaleAbs(roi, alpha=1.5, beta=0)

    # Apply adaptive thresholding
    roi = cv2.adaptiveThreshold(
        roi, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 11, 2
    )

    # Remove noise
    kernel = np.ones((2, 2), np.uint8)
    roi = cv2.morphologyEx(roi, cv2.MORPH_OPEN, kernel)

    # Fill holes
    roi = cv2.morphologyEx(roi, cv2.MORPH_CLOSE, kernel)

    return roi


def detect_single_digit(frame, model):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Enhanced edge detection
    edges = cv2.Canny(blurred, 50, 150)

    # Dilate edges to connect components
    kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(edges, kernel, iterations=1)

    # Find contours
    contours, _ = cv2.findContours(
        dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if contours:
        # Filter contours by area and aspect ratio
        valid_contours = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 100:  # Adjust minimum area as needed
                continue

            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = float(w)/h

            # Filter based on aspect ratio (adjust ranges as needed)
            if aspect_ratio > 0.2 and aspect_ratio < 1.0:
                valid_contours.append(contour)

        if valid_contours:
            # Find the largest valid contour
            largest_contour = max(valid_contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_contour)

            # Add padding around the digit
            padding = 10
            x_start = max(0, x - padding)
            y_start = max(0, y - padding)
            x_end = min(frame.shape[1], x + w + padding)
            y_end = min(frame.shape[0], y + h + padding)

            # Extract ROI
            roi = gray[y_start:y_end, x_start:x_end]

            # Enhance the digit region
            roi_enhanced = enhance_digit_region(roi)

            # Show the enhanced ROI for debugging
            cv2.imshow("Enhanced ROI", roi_enhanced)

            # Preprocess and predict
            roi_tensor = preprocess_roi(roi_enhanced)
            digit, confidence = predict_digit(model, roi_tensor)

            if digit is not None:
                # Draw bounding box
                cv2.rectangle(frame, (x_start, y_start),
                              (x_end, y_end), (0, 255, 0), 2)

                # Display prediction and confidence
                text = f"{digit} ({confidence:.2f})"
                cv2.putText(frame, text, (x_start, y_start - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

    return frame


def live_video_single_digit_recognition(model):
    cap = cv2.VideoCapture(0)

    # Set camera properties for better quality
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)

    if not cap.isOpened():
        print("Error: Cannot access the webcam.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        processed_frame = detect_single_digit(frame, model)
        cv2.imshow("Live Single Digit Recognition", processed_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    model_path = "/Users/sundri/Desktop/CS5330/Project5/src/model/mnist_model.pth"
    model = load_model(model_path)
    live_video_single_digit_recognition(model)
