import nnfs
from nnfs.datasets import spiral_data
import numpy as np

nnfs.init()

class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        # Initailize weights and biases
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
        
    def forward(self, inputs):
        # Calculate output values from inputs, weights and biases
        self.output = np.dot(inputs, self.weights) + self.biases

X, y = spiral_data(samples=100, classes=3) # Create dataset in the 2D plane (100 samples, 2 features

# Create dense layer with 2 input features and 3 output values
dense1 = Layer_Dense(2, 3)

# Second "hidden" layer
dense2 = Layer_Dense(3, 1)

# Perform a forward pass of training data through this layer
dense1.forward(X)

# Perform a forward pass of 1st layer outputs through this layer
dense2.forward(dense1.output)

print(dense2.output[:5]) # Shape of (100, 1)
