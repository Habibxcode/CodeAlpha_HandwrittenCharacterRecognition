# Handwritten Character Recognition - CodeAlpha Machine Learning Internship

## Project Overview
This repository contains the implementation of Task 3: Handwritten Character Recognition, completed as part of the CodeAlpha Machine Learning Internship. The primary objective of this project is to develop a deep learning model capable of accurately identifying and classifying handwritten digits (0-9) using image processing and neural network techniques.

## Dataset
This project utilizes the MNIST (Modified National Institute of Standards and Technology) dataset, which is a large database of handwritten digits commonly used for training various image processing systems.
* Training Set: 60,000 grayscale images.
* Testing Set: 10,000 grayscale images.
* Image Format: 28x28 pixels, normalized.

## System Requirements and Installation
To run this project locally, ensure you have Python 3.8 or higher installed on your system. It is recommended to use a virtual environment.

1. Clone the repository to your local machine:
   git clone <your-repository-url>
   cd <your-repository-folder>

2. Install the required Python dependencies using the provided requirements.txt file:
   pip install -r requirements.txt

The core dependencies include:
* torch
* torchvision
* matplotlib
* numpy

## Model Architecture
The solution employs a Convolutional Neural Network (CNN) built from scratch using PyTorch. The architecture is designed to effectively extract spatial features from the input images and classify them into one of the ten digit classes.

The network consists of two main parts:
1. Feature Extractor (Convolutional Layers):
   - Convolutional Layer 1: 1 input channel, 32 output channels, 3x3 kernel, padding of 1. Followed by Batch Normalization, ReLU activation, and 2x2 Max Pooling.
   - Convolutional Layer 2: 32 input channels, 64 output channels, 3x3 kernel, padding of 1. Followed by Batch Normalization, ReLU activation, and 2x2 Max Pooling.

2. Classifier (Fully Connected Layers):
   - Flatten layer to convert the 2D feature maps into a 1D vector.
   - Linear Layer 1: Takes the flattened output (64 * 7 * 7) and maps it to 128 units. Followed by ReLU activation and Dropout (rate = 0.25) to prevent overfitting.
   - Output Linear Layer: Maps the 128 units to the final 10 classes (digits 0-9).

## Training Details
* Loss Function: Cross Entropy Loss (standard for multi-class classification).
* Optimizer: Adam optimizer with a learning rate of 0.001.
* Epochs: 3 (can be adjusted in the script for further fine-tuning).
* Batch Size: 64 for training, 1000 for testing.
* Hardware: The script is configured to automatically utilize a CUDA-enabled GPU if available, otherwise it falls back to the CPU.

## Execution Instructions
To train the model and evaluate its performance, execute the main Python script from your terminal:

   python handwritten_recognition.py

### Expected Outputs
Running the script will execute the following pipeline:
1. Download the MNIST dataset to a local `./data` folder (if not already present).
2. Train the CNN over the specified number of epochs, printing the loss at each step.
3. Evaluate the model against the 10,000 test images and print the final accuracy percentage.
4. Save the trained model parameters to a file named `handwritten_cnn.pth`.
5. Generate and save a matplotlib plot named `sample_predictions.png` displaying 10 test images alongside their true labels and the model's predictions.

## Acknowledgments
This project was developed as a task for the CodeAlpha Machine Learning Internship program.