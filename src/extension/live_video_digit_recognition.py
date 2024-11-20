import cv2
import numpy as np
import torch
from torchvision import transforms
from model import MyNetwork


def load_model(model_path):
    model = MyNetwork()
    model.load_state_dict(torch.load(
        model_path, map_location=torch.device('cpu')))
    model.eval()
    return model


def preprocess_roi(roi):
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Grayscale(),
        transforms.Pad(padding=4, fill=0),
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    return transform(roi).unsqueeze(0)


def enhance_digit_region(roi):
    # Enhance contrast for better digit visibility
    roi = cv2.convertScaleAbs(roi, alpha=2.0, beta=10)

    # Adaptive thresholding with optimized parameters
    roi = cv2.adaptiveThreshold(
        roi, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY_INV, 9, 5
    )

    # Careful noise reduction while preserving digit details
    kernel = np.ones((2, 2), np.uint8)
    roi = cv2.morphologyEx(roi, cv2.MORPH_OPEN, kernel)
    roi = cv2.dilate(roi, kernel, iterations=1)

    return roi


def sort_contours(contours, method="left-to-right"):
    # Sort contours from left to right
    reverse = False
    i = 0
    if method == "right-to-left" or method == "bottom-to-top":
        reverse = True
    if method == "top-to-bottom" or method == "bottom-to-top":
        i = 1

    # Create bounding boxes and sort them
    bounding_boxes = [cv2.boundingRect(c) for c in contours]
    (contours, bounding_boxes) = zip(*sorted(zip(contours, bounding_boxes),
                                             key=lambda b: b[1][i], reverse=reverse))

    return contours


def detect_multiple_digits(frame, model):
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply bilateral filter for noise reduction while preserving edges
    blurred = cv2.bilateralFilter(gray, 9, 75, 75)

    # Otsu's thresholding for robust binarization
    _, binary = cv2.threshold(
        blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Edge detection
    edges = cv2.Canny(binary, 30, 200)

    # Dilate edges to connect components
    kernel = np.ones((2, 2), np.uint8)
    dilated = cv2.dilate(edges, kernel, iterations=1)

    # Find contours
    contours, _ = cv2.findContours(
        dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter and sort valid contours
    valid_contours = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 80:  # Minimum area threshold
            continue

        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = float(w)/h

        # Filter based on aspect ratio and size
        if 0.15 < aspect_ratio < 1.2 and h > 20:  # Added minimum height check
            valid_contours.append(contour)

    # Sort contours left to right
    if valid_contours:
        valid_contours = sort_contours(valid_contours)

        # Process each valid contour
        detected_digits = []
        for i, contour in enumerate(valid_contours):
            x, y, w, h = cv2.boundingRect(contour)

            # Add padding around digit
            padding = int(max(w, h) * 0.2)
            x_start = max(0, x - padding)
            y_start = max(0, y - padding)
            x_end = min(frame.shape[1], x + w + padding)
            y_end = min(frame.shape[0], y + h + padding)

            # Extract and process ROI
            roi = gray[y_start:y_end, x_start:x_end]

            if roi.size > 0 and roi.shape[0] > 10 and roi.shape[1] > 10:
                roi_enhanced = enhance_digit_region(roi)

                # Show individual ROIs for debugging
                cv2.imshow(f"ROI_{i}", roi_enhanced)

                roi_tensor = preprocess_roi(roi_enhanced)

                # Predict digit with confidence check
                with torch.no_grad():
                    output = model(roi_tensor)
                    probabilities = torch.nn.functional.softmax(output, dim=1)
                    values, indices = torch.topk(probabilities, 2)

                    # Adjust confidence threshold based on digit
                    predicted_digit = indices[0][0].item()
                    confidence = values[0][0].item()

                    # Different thresholds for different digits
                    threshold = 0.5 if predicted_digit in [1, 4, 6, 7] else 0.7

                    if confidence > threshold:
                        # Store detection information
                        detected_digits.append({
                            'digit': predicted_digit,
                            'confidence': confidence,
                            'position': (x_start, y_start, x_end, y_end)
                        })

        # Draw all valid detections
        for detection in detected_digits:
            x_start, y_start, x_end, y_end = detection['position']
            digit = detection['digit']
            confidence = detection['confidence']

            # Draw bounding box
            cv2.rectangle(frame, (x_start, y_start),
                          (x_end, y_end), (0, 255, 0), 2)

            # Draw prediction and confidence
            text = f"{digit} ({confidence:.2f})"
            cv2.putText(frame, text, (x_start, y_start - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

    return frame


def live_video_digit_recognition(model):
    cap = cv2.VideoCapture(0)

    # Optimize camera settings
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

        processed_frame = detect_multiple_digits(frame, model)
        cv2.imshow("Multiple Digit Recognition", processed_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    model_path = "/Users/sundri/Desktop/CS5330/Project5/src/model/mnist_model.pth"
    model = load_model(model_path)
    live_video_digit_recognition(model)
