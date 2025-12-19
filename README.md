# Handwritten-Digit-Recognition-using-Neural-Network
Handwritten Digit Recognition using Neural Network is a project that trains a deep learning model to recognize digits (0–9) from images of handwritten numbers. The model learns from thousands of labeled grayscale images (such as MNIST‑style 28×28 pixel digits), automatically extracting patterns like strokes and shapes to classify each image into the correct digit class. The repository typically includes data preprocessing, model training, evaluation, and visualization of predictions.
  
This project implements a neural network to classify images of handwritten digits (0–9). The dataset consists of grayscale digit images resized to 28×28 pixels and normalized so that pixel values lie between 0 and 1. A feed‑forward network with fully connected layers (ReLU activations and softmax output) is trained using categorical cross‑entropy loss and the Adam optimizer to learn discriminative features for each digit.  
  
The code covers:
- Loading and preprocessing the dataset (reshaping to $$(28, 28, 1)$$, normalization, and label encoding).  
- Building and compiling the neural network model.  
- Training the model over multiple epochs and tracking accuracy on validation data.  
- Evaluating performance on unseen test data and visualizing sample predictions with their predicted labels.  
  
This repository serves as a concise end‑to‑end example of image classification with neural networks and is a good starting point for experimenting with different architectures, regularization techniques, and optimization settings.

Sources
[1] Neural Network Python Project - Handwritten Digit Recognition https://www.youtube.com/watch?v=bte8Er0QhDg
[2] Keras Fully Connected Neural Network using Python for Digit ... https://github.com/husnainfareed/MNIST-Handwritten-Digit-Recognition
[3] mnist-handwriting-recognition · GitHub Topics https://github.com/topics/mnist-handwriting-recognition
[4] Handwritten digits recognition (using Convolutional Neural Network) https://colab.research.google.com/github/trekhleb/machine-learning-experiments/blob/master/experiments/digits_recognition_cnn/digits_recognition_cnn.ipynb
