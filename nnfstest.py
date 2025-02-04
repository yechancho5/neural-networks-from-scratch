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
        self.inputs = inputs
        # Calculate output values from inputs, weights and biases
        self.output = np.dot(inputs, self.weights) + self.biases

    def backward(self, dvalues):
        # Gradients on parameters
        # w.r.t. weights, transpose inputs to match weights array
        self.dweights = np.dot(self.inputs.T, dvalues)
        
        # w.r.t. biases, sum gradient for each input example 
        # partial derivative with respect to bias is always 1 * previous gradients
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
        
        # Gradient on values, backpropagated to next layer
        # w.r.t. inputs, transpose weights to match inputs array 
        self.dinputs = np.dot(dvalues, self.weights.T) 
        
class Activation_ReLU: # f(x) = max(x, 0)
    # Forward pass
    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.maximum(0, inputs)
    # Backward pass
    def backward(self, dvalues):
        #dvalues is dense2.inputs, the gradient of the loss (prev. layer) w.r.t. the outputs of the ReLU function
        self.dinputs = dvalues.copy()
        # Zero gradient where input values are negative
        self.dinputs[self.inputs <= 0] = 0

class Activation_Softmax: 
    def forward(self, inputs):
        # Get unormalized probabilities
        exp_values=np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        # Divide 
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)

        self.output = probabilities
    
    def backward(self, dvalues):
        # Create uninitialized array
        self.dinputs = np.empty_like(dvalues)

        # Enumerate outputs and gradients
        for index, (single_output, single_dvalues) in enumerate(zip((self.output, dvalues))):
            single_output = single_output.reshape(-1, 1)
            # Calculate Jacobian matrix
            jacobian_matrix = np.diagflat(single_output) - np.dot(single_output, single_output.T)
            # Calculate sample-wise gradient and add it to the array of sample gradients
            self.dinputs[index] = np.dot(jacobian_matrix, single_dvalues)                                                                

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
            # possible because one-hot encoded lists are ex) [0, 0, 1], so multiplying by 1 cancels out
            correct_confidences = np.sum(y_pred_clipped * y_true, axis=1) 

        # Losses
        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods
    
    def backward(self, dvalues, y_true):
        samples = len(dvalues)
        # Number of labels in every sample
        labels = len(dvalues[0])

        if len(y_true.shape) == 1:
            y_true = np.eye(labels)[y_true]

        # Calculate gradient
        self.dinputs = -y_true / dvalues
        # Normalize gradient
        self.dinputs = self.dinputs / samples
    
class Activation_Softmax_Loss_CategoricalCrossentropy():
    # Creates activation and loss function objects
    def __init__(self):
        self.activation = Activation_Softmax()
        self.loss = Loss_CategoricalCrossentropy()
    
    # Forward pass
    def forward(self, inputs, y_true):
        # Output layer's activation function
        self.activation.forward(inputs)
        self.output = self.activation.output
        # Calculate and return loss value
        return self.loss.calculate(self.output, y_true)
    
    # Backward pass
    def backward(self, dvalues, y_true):
        samples = len(dvalues)
    
        # If labels are one-hot encoded, turn them into discrete values
        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis=1)

        self.dinputs = dvalues.copy()
        # Calculate gradient
        self.dinputs[range(samples), y_true] -= 1
        # Normalize
        self.dinputs = self.dinputs / samples # now holds the gradient of the loss w.r.t. the logits of dense2

class Optimizer_SGD: 
    # Initialize with a default learning rate of 1
    def __init__(self, learning_rate=1.0):
        self.learning_rate = learning_rate
    
    def update_params(self, layer):
        layer.weights += -self.learning_rate * layer.dweights # W = W - learning rate * gradient of loss w.r.t. weights
        layer.biases += -self.learning_rate * layer.dbiases # B = B - learning rate * gradient of loss w.r.t. biases

X, y = spiral_data(samples=100, classes=3) # Create dataset in the 2D plane (100 samples, 2 features)

# Create dense layer with 2 input features and 3 output values
dense1 = Layer_Dense(2, 3)

activation1 = Activation_ReLU()

# Second "hidden" layer
dense2 = Layer_Dense(3, 3)

# Create Softmax classifier's combined loss and activation
loss_activation = Activation_Softmax_Loss_CategoricalCrossentropy()

# Create optimizer
optimizer = Optimizer_SGD()

for epoch in range(10001):
    # Perform a forward pass of training data through this layer
    dense1.forward(X)

    activation1.forward(dense1.output) # Make a forward pass through the activation function 

    # Makes a forward pass through second Dense layer
    dense2.forward(activation1.output)

    loss = loss_activation.forward(dense2.output, y)

    # Calculate accuracy from output of activation2 and targets along the first axis (row)
    predictions = np.argmax(loss_activation.output,axis=1)
    if len(y.shape) == 2:
        y = np.argmax(y, axis=1) 
    accuracy = np.mean(predictions==y) # returns an array of Boolean values and converts them to True = 1, False = 0 then takes mean

    if (epoch % 100 == 0):
        print(f'epoch: {epoch}', f'acc: {accuracy:.3f}', f'loss: {loss:.3f}')

    # dvalues = gradient of the loss w.r.t. dense2 logits i.e. softmax outputs
    loss_activation.backward(loss_activation.output, y)
    # dvalues = gradient of loss w.r.t. dense2 outputs
    dense2.backward(loss_activation.dinputs)
    # dvalues = gradient of loss w.r.t. ReLU outputs
    activation1.backward(dense2.dinputs)
    # dvalues = gradient of loss w.r.t. dense1 outputs
    dense1.backward(activation1.dinputs)

    optimizer.update_params(dense1)
    optimizer.update_params(dense2)