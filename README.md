# Three-Spiral Neural Network

An implementation of a neural network trained to solve the
Three-Spiral nonlinear classification problem.

The project was developed from scratch using NumPy, with Matplotlib
used for visualization and training analysis.

## Problem

The Three-Spiral dataset consists of three interleaved classes
forming a nonlinear decision boundary.

[IMAGE / GIF]
https://github.com/FabioJCSeverino/Three-SpiralML/blob/013599d57bfb64baccb3db3b390e105f23b14755/threespiral.png

## Neural Network

Architecture:

Input: 2
Hidden layer 1: 128
Hidden layer 2: 198
Hidden layer 3: 64
Output: 3

Activation functions: ReLU
Loss function: Cross-Entropy
Learning rate: 0.05
Epochs: 5000

## Results

The model achieved approximately **95% classification accuracy**.

[TRAINING LOSS GRAPH]
https://github.com/FabioJCSeverino/Three-SpiralML/blob/b362539df40a839098be37c7cc44dd9bc24765da/TrainingLoss.png

[DECISION BOUNDARY GIF]
https://github.com/FabioJCSeverino/Three-SpiralML/blob/b362539df40a839098be37c7cc44dd9bc24765da/neural_network_learning.gif

## Technologies

- Python
- NumPy
- Matplotlib
