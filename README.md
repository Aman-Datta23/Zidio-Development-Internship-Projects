# Zidio-Development-Internship-Projects
Machine Learning Projects
This repository contains two of my machine learning projects:

1. Speech Emotion Recognition using CNN and LSTM
Overview
This project implements a Speech Emotion Recognition (SER) system, leveraging Convolutional Neural Networks (CNN) and Long Short-Term Memory (LSTM) models to classify emotions from speech signals. The system recognizes different emotions like happy, sad, angry, etc., based on extracted speech features.

Key Features
CNN and LSTM Integration: Combines CNNs for feature extraction and LSTMs for temporal sequence learning.
Advanced Feature Extraction: Utilizes MFCCs (Mel-Frequency Cepstral Coefficients), HOG (Histogram of Oriented Gradients), and edge detection.
Real-time Data Processing: Real-time recognition pipeline implemented with collaborative filtering, content-based filtering, and hybrid methods.
High Accuracy: Achieved ~98% accuracy after hyperparameter tuning and optimization.
Tools and Technologies
Libraries: Python, TensorFlow, Keras, NumPy, Pandas, Matplotlib
Models: CNN, LSTM
Feature Extraction: MFCC, HOG, Edge Detection
Performance Metrics: Accuracy, Precision, Recall, F1-Score
Optimization Techniques: Hyperparameter tuning, model optimization
Dataset
The dataset contains speech signals labeled with different emotions. The data is preprocessed to extract relevant features, and then split into training and testing sets.
Installation
Clone the repository:https://github.com/Aman-Datta23/Zidio-Development-Internship-Projects/tree/main?tab=readme-ov-file
cd ml-projects/Speech-Emotion-Recognition
Install dependencies: pip install -r requirements.txt
Run the project: python train_model.py
Results
Achieved ~98% accuracy on the test set. Detailed performance metrics can be found in the report.





2. MNIST Handwritten Digit Classification
Overview
This project focuses on classifying handwritten digits from the popular MNIST dataset using a Convolutional Neural Network (CNN). The goal is to identify digits (0-9) from grayscale images.

Key Features
Simple CNN Architecture: Implements a CNN with multiple layers for robust feature learning.
Efficient Training: Trained on the MNIST dataset, a benchmark for image classification.
High Accuracy: Achieved over 99% accuracy on the test set.
Tools and Technologies
Libraries: Python, TensorFlow, Keras, NumPy, Matplotlib, Scikit-learn
Model: CNN
Performance Metrics: Accuracy, Confusion Matrix, Loss
Dataset
The MNIST dataset consists of 60,000 training images and 10,000 test images of handwritten digits. Each image is 28x28 pixels, labeled from 0 to 9.

Installation
Clone the repository: https://github.com/Aman-Datta23/Zidio-Development-Internship-Projects/tree/main?tab=readme-ov-file
cd ml-projects/MNIST-Digit-Classification
Install dependencies: pip install -r requirements.txt
Run the project: python train_model.py
Results
Achieved over 99% accuracy on the MNIST test set. Visualization of results and confusion matrix is available in the output folder.


License:
This project is licensed under the MIT License - see the LICENSE file for details.

Contact:
If you have any questions, feel free to contact me - amandatta02@gmail.com



