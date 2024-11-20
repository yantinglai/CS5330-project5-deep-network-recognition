# CS5330 Project 5: Recognition using Deep Networks

## Student Information
* Name: Yanting Lai
* Organization: Northeastern University
* Course: CS 5330 Pattern Recognition & Computer Vision
* Time: Nov-19-2024

## Video Submission
* Real-time Digit Recognition Demo: [https://drive.google.com/file/d/1oHGcWV-TwMFkFOAxCgDZvONn645k89me/view?usp=sharing](https://drive.google.com/file/d/1oHGcWV-TwMFkFOAxCgDZvONn645k89me/view?usp=sharing)

## Development Environment
* Operating System: MacOS Sonoma 14.1.1
* IDE: PyCharm Professional 2023.2.5
* Python Version: 3.9.18
* Key Dependencies:
  - PyTorch 2.1.1
  - OpenCV 4.8.1.78
  - NumPy 1.26.2
  - Matplotlib 3.8.2

## Running Instructions

### Basic Setup
1. Clone the repository
2. Install required packages:
```bash
pip install torch torchvision opencv-python numpy matplotlib
```

### Running the Main Tasks
1. Training MNIST Network:
```bash
python train_mnist.py
```

2. Network Examination:
```bash
python examine_network.py
```

3. Transfer Learning for Greek Letters:
```bash
python transfer_learning_greek.py
```

4. Custom Experiments:
```bash
python experiment_runner.py
```

### Running Extensions

1. Gabor Filter Extension:
```bash
python train_mnist.py
```

2. Real-time Digit Recognition:
```bash
python live_video_digit_recognition.py
```
Note: Make sure your webcam is properly connected before running the live recognition.

## Extension Testing Instructions
1. For Gabor Filter Extension:
   * Run the script and compare accuracy with original network
   * View generated filter visualizations

2. For Real-time Recognition:
   * Hold handwritten digits in front of webcam
   * Press 'q' to quit the application
   * Best results with clear, centered digits on white paper

## Time Travel Days
* 2

## Additional Notes
* The model weights are saved in `src/model/mnist_model.pth`
* For best real-time recognition results, ensure good lighting conditions
* Test digits should be written clearly on white paper with black marker
