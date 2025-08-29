# Handwritten Digit Recognition 

This project implements a Handwritten Digit Recognition system using Deep Learning (Convolutional Neural Networks).
It is capable of recognizing digits (0–9) from images of handwritten numbers.

The project is inspired by real-world applications such as bank check recognition, postal code reading, and OCR systems.

##Features

Recognizes handwritten digits from images.

Built with TensorFlow/Keras.

Uses a CNN model for high accuracy.



## 📂 Dataset

Default dataset: MNIST
 (70,000 grayscale images, 28×28 pixels).


## Model Architecture

The CNN model consists of:

Convolutional Layer → extracts features (edges, curves).

MaxPooling Layer → reduces image size while keeping important features.

Flatten Layer → converts image matrix into a 1D vector.

Dense Layer (ReLU) → learns complex patterns.

Output Layer (Softmax) → predicts digit (0–9).
