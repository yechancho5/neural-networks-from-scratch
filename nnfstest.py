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

class Loss: 
    # Calculates the data and regularization losses
    def calculate(self, output, y):
        
        sample_losses = self.forward(output, y)
       
        # Calculate mean loss
        data_loss = np.mean(sample_losses)

        return data_loss

class Loss_CategoricalCrossentropy(Loss):
    def forward(self, y_pred, y_true):

        samples = len(y_pred)

        # Clip data to prevent division by 0
        # Clip both sides to not drag mean towards any value
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)

        # Probabilities for target values
        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[range(samples), y_true]

        # Mask values - for lists of lists
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(y_pred_clipped * y_true, axis=1) # possible because one-hot encoded lists are ex) [0, 0, 1], so multiplying by 1 cancels out

        # Losses
        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods

X, y = spiral_data(samples=100, classes=3) # Create dataset in the 2D plane (100 samples, 2 features)

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

# Makes a forward pass through the Softmax Activation Function (100, 3) 
activation2.forward(dense2.output)

loss_function = Loss_CategoricalCrossentropy()
loss = loss_function.calculate(activation2.output, y)

print('loss:', loss)

# Calculate accuracy from output of activation2 and targets along the first axis (row)
predictions = np.argmax(activation2.output, axis=1)
if len(y.shape) == 2:
    y = np.argmax(y, axis=1)
accuracy = np.mean(predictions==y)

print('acc:', accuracy)
