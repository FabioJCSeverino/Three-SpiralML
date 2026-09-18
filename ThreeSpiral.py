import numpy as np

class dense_layer:
    def __init__(self,n_inputs,n_neurons):
        self.weights = np.random.randn(n_inputs,n_neurons) * np.sqrt(2 / n_inputs)
        self.biases = np.zeros((1,n_neurons))
    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.biases
    def backward(self, dvalues):
        self.dweights = np.dot(
            self.inputs.T,
            dvalues
        )

        self.dbiases = np.sum(
            dvalues,
            axis=0,
            keepdims=True
        )

        self.dinputs = np.dot(
            dvalues,
            self.weights.T
        )

class ReLUAct:
    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.maximum(0,inputs)
    def backward(self, dvalues):
        self.dinputs = dvalues.copy()
        self.dinputs[self.inputs <= 0] = 0

class softmaxAct:
    def forward(self,inputs):
        exp_values = np.exp(inputs - np.max(inputs,axis=1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities
    def backward(self, dvalues, y_true):
        samples = len(dvalues)
        if(len(y_true.shape) == 2):
            y_true = np.argmax(y_true, axis=1)
        self.dinputs = dvalues.copy()
        self.dinputs[range(samples), y_true] -= 1
        self.dinputs = self.dinputs / samples


class Loss:
    def calculate(self,output,y):
        sample_losses = self.forward(output,y)
        data_loss = np.mean(sample_losses)
        return data_loss

class loss_crossEntropy(Loss):
    def forward(self,y_pred,y_true):
        samples = len(y_pred)
        y_pred_clipped = np.clip(y_pred, 1e-7,1-1e-7)

        if len(y_true.shape) == 1:
            correct = y_pred_clipped[range(samples), y_true]

        elif len(y_true.shape) == 2:
            correct = np.sum(y_pred_clipped*y_true, axis=1)

        negative_log = -np.log(correct)
        return negative_log

def accuracy(softmax_output, y_true):
    predictions = np.argmax(softmax_output, axis=1)
    if len(y_true.shape) == 2:
        y_true = np.argmax(y_true, axis=1)
    return np.mean(predictions == y_true)


def spiral_data(points, classes):
    X = np.zeros((points*classes, 2))
    y = np.zeros(points*classes, dtype='uint8')
    for class_number in range(classes):
        ix = range(points*class_number, points*(class_number+1))
        r = np.linspace(0.0, 1, points)  # radius
        t = np.linspace(class_number*4, (class_number+1)*4, points) + np.random.randn(points)*0.2
        X[ix] = np.c_[r*np.sin(t*2.5), r*np.cos(t*2.5)]
        y[ix] = class_number
    return X, y

import matplotlib.pyplot as plt
X,y = spiral_data(100,3)

# Create a grid covering the input space
x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1

xx, yy = np.meshgrid(
    np.linspace(x_min, x_max, 300),
    np.linspace(y_min, y_max, 300)
)

# Turn the grid into a list of (x, y) points
grid = np.c_[xx.ravel(), yy.ravel()]

import os
import glob

os.makedirs("frames", exist_ok=True)

epochs = 5000
learning_rate = 0.05

fc1 = dense_layer(2,128)
activation1 = ReLUAct()

fc2 = dense_layer(128,198)
activation2 = ReLUAct()

fc3 = dense_layer(198,64)
activation3 = ReLUAct()

fc4 = dense_layer(64,3)
activation4 = softmaxAct()

loss_function = loss_crossEntropy()

loss_history = []

for epoch in range(epochs):
    #Forward
    fc1.forward(X)
    activation1.forward(fc1.output)

    fc2.forward(activation1.output)
    activation2.forward(fc2.output)

    fc3.forward(activation2.output)
    activation3.forward(fc3.output)

    fc4.forward(activation3.output)
    activation4.forward(fc4.output)

    loss = loss_function.calculate(activation4.output,y)

    if epoch % 100 == 0:
            acc = accuracy(activation4.output, y)
            print(f"epoch {epoch:4d}  loss {loss:.4f}  acc {acc:.3f}")
    loss_history.append(loss)

    #Backward
    activation4.backward(activation4.output,y)
    fc4.backward(activation4.dinputs)

    activation3.backward(fc4.dinputs)
    fc3.backward(activation3.dinputs)

    activation2.backward(fc3.dinputs)
    fc2.backward(activation2.dinputs)

    activation1.backward(fc2.dinputs)
    fc1.backward(activation1.dinputs)

    #Adjust
    fc1.weights -= learning_rate * fc1.dweights
    fc1.biases -= learning_rate * fc1.dbiases

    fc2.weights -= learning_rate * fc2.dweights
    fc2.biases -= learning_rate * fc2.dbiases

    fc3.weights -= learning_rate * fc3.dweights
    fc3.biases -= learning_rate * fc3.dbiases

    fc4.weights -= learning_rate * fc4.dweights
    fc4.biases -= learning_rate * fc4.dbiases

# VISUALIZE
    if epoch % 50 == 0:
        # Run grid through network
        fc1.forward(grid)
        activation1.forward(fc1.output)

        fc2.forward(activation1.output)
        activation2.forward(fc2.output)

        fc3.forward(activation2.output)
        activation3.forward(fc3.output)

        fc4.forward(activation3.output)
        activation4.forward(fc4.output)

        # Output probabilities
        probabilities = activation4.output

        # Predicted class
        predictions = np.argmax(
            probabilities,
            axis=1
        )

        predictions = predictions.reshape(
            xx.shape
        )

        # Individual class probabilities
        class0 = probabilities[:, 0].reshape(xx.shape)
        class1 = probabilities[:, 1].reshape(xx.shape)
        class2 = probabilities[:, 2].reshape(xx.shape)

        # PLOT
        fig, ax = plt.subplots(figsize=(8, 8))

        # Prediction regions
        ax.contourf(xx, yy, predictions, levels=[-0.5, 0.5, 1.5, 2.5], alpha=0.15)

        #Countours

        # Class 0
        ax.contour(xx, yy, class0, levels=[0.25, 0.5, 0.75], linewidths=1)

        # Class 1
        ax.contour(xx, yy, class1, levels=[0.25, 0.5, 0.75], linewidths=1)

        #Class2
        ax.contour(xx, yy, class2, levels=[0.25, 0.5, 0.75], linewidths=1)

        ax.scatter(X[:, 0], X[:, 1], c=y, edgecolor="black", s=40)

        ax.set_xlim(x_min, x_max)

        ax.set_ylim(y_min, y_max)

        ax.set_xlabel("x1")
        ax.set_ylabel("x2")

        ax.set_title(f"Neural Network - Epoch {epoch}")

        plt.savefig(f"frames/frame_{epoch:05d}.png", dpi=120)

        plt.close()

# Final forward pass

fc1.forward(X)
activation1.forward(fc1.output)

fc2.forward(activation1.output)
activation2.forward(fc2.output)

fc3.forward(activation2.output)
activation3.forward(fc3.output)

fc4.forward(activation3.output)
activation4.forward(fc4.output)

final_loss = loss_function.calculate(activation4.output, y)

final_accuracy = accuracy(activation4.output, y)

print(f"\nfinal loss: {final_loss:.4f}")
print(f"final accuracy: {final_accuracy:.3f}")

from PIL import Image
import glob

files = sorted(glob.glob("frames/frame_*.png"))

images = [
    Image.open(file)
    for file in files
]

images[0].save(
    "neural_network_learning_4layer.gif",
    save_all=True,
    append_images=images[1:],
    duration=100,
    loop=0
)

np.savez(
    "spiral_model_4layer.npz",

    fc1_weights=fc1.weights,
    fc1_biases=fc1.biases,

    fc2_weights=fc2.weights,
    fc2_biases=fc2.biases,

    fc3_weights=fc3.weights,
    fc3_biases=fc3.biases,

    fc4_weights=fc4.weights,
    fc4_biases=fc4.biases
)

plt.figure()
plt.plot(loss_history)
plt.xlabel("epoch")
plt.ylabel("loss")
plt.title("Training loss")
plt.show()

print("Loss: ", loss)