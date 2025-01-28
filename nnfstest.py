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
        
class Activation_ReLU: # f(x) = max(x, 0)
    # Forward pass
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)

class Activation_Softmax: 
    def forward(self, inputs):
        # Get unormalized probabilities
        exp_values=np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        # Divide 
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)

        self.output = probabilities

X, y = spiral_data(samples=100, classes=3) # Create dataset in the 2D plane (100 samples, 2 features

# Create dense layer with 2 input features and 3 output values
dense1 = Layer_Dense(2, 3)

activation1 = Activation_ReLU()

# Second "hidden" layer
dense2 = Layer_Dense(3, 3)

activation2 = Activation_Softmax()

# Perform a forward pass of training data through this layer
dense1.forward(X)

activation1.forward(dense1.output) # Make a forward pass through the activation function 

# Makes a forward pass through second Dense layer
dense2.forward(activation1.output)

# Makes a forward pass through the Softmax Activation Function
activation2.forward(dense2.output)

print(activation2.output[:5])