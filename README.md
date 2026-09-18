# Three-Spiral Neural Network

An implementation of a neural network trained to solve the
Three-Spiral nonlinear classification problem.

The project was developed from scratch using NumPy, with Matplotlib
used for visualization and training analysis.

## Problem

The Three-Spiral dataset consists of three interleaved classes
forming a nonlinear decision boundary.

[IMAGE / GIF]

## Neural Network

Architecture:

Input: 2
Hidden layer 1: 128
Hidden layer 2: 198
Hidden layer 3: 64
Output: 3

Activation functions: ReLU
Loss function: Cross-Entropy
Optimizer: [ADD]
Learning rate: 0.05
Epochs: 5000

## Results

The model achieved approximately **95% classification accuracy**.

https://github.com/FabioJCSeverino/Three-SpiralML/blob/b362539df40a839098be37c7cc44dd9bc24765da/TrainingLoss.png

https://github.com/FabioJCSeverino/Three-SpiralML/blob/b362539df40a839098be37c7cc44dd9bc24765da/neural_network_learning.gif

## Technologies

- Python
- NumPy
- Matplotlib
