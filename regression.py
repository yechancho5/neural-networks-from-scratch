import nnfs
from nnfs.datasets import spiral_data
import numpy as np

nnfs.init()

class Layer_Dense:
    def __init__(self, n_inputs, n_neurons, weight_regularizer_L1=0, weight_regularizer_L2=0, bias_regularizer_L1=0, bias_regularizer_L2=0):
        # Initailize weights and biases
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

        # Set regularization strength
        self.weight_regularizer_l1 = weight_regularizer_L1
        self.weight_regularizer_l2 = weight_regularizer_L2
        self.bias_regularizer_l1 = bias_regularizer_L1
        self.bias_regularizer_l2 = bias_regularizer_L2
        
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
        
        # Gradients on regularization
        # L1 on weights
        if self.weight_regularizer_l1 > 0:
            dL1 = np.ones_like(self.weights)
            dL1[self.weights < 0] = -1
            self.dweights += self.weight_regularizer_l1 * dL1
        # L2 on weights
        if self.weight_regularizer_l2 > 0:
            self.dweights += 2 * self.weight_regularizer_l2 * self.weights
        
        # L1 on biases
        if self.bias_regularizer_l1 > 0:
            dL1 = np.ones_like(self.biases)
            dL1[self.biases < 0] = -1
            self.dbiases += self.bias_regularizer_l1 * dL1
        # L2 on biases
        if self.bias_regularizer_l2 > 0:
            self.dbiases += 2 * self.bias_regularizer_l2 * self.biases
        
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

class Activation_Sigmoid:
    def forward(self, inputs):
        self.inputs = inputs
        self.output = 1 / (1+np.exp(-inputs)) # inputs = z
    
    def backward(self, dvalues):
        self.dinputs = dvalues * (1 - self.output) * self.output

class Loss: 
    # Calculates the data and regularization losses
    def calculate(self, output, y):
        
        sample_losses = self.forward(output, y)
       
        # Calculate mean loss
        data_loss = np.mean(sample_losses)

        return data_loss
    
    def regularization_loss(self, layer):

        # 0 by default
        regularization_loss = 0

        # L1 regularization - weights - sum of all the weights
        if layer.weight_regularizer_l1 > 0:
            regularization_loss += layer.weight_regularizer_l1 * np.sum(np.abs(layer.weights))

        # L2 regularization - weight - sum of squares
        if layer.weight_regularizer_l2 > 0:
            regularization_loss += layer.weight_regularizer_l2 * np.sum(layer.weights * layer.weights)
    
        # L1 regularization - bias
        if layer.bias_regularizer_l1 > 0:
            regularization_loss += layer.bias_regularizer_l1 * np.sum(np.abs(layer.biases))
    
        # L2 regularization - bias
        if layer.bias_regularizer_l2 > 0:
            regularization_loss += layer.bias_regularizer_l2 * np.sum(layer.biases * layer.biases)

        return regularization_loss

class Loss_BinaryCrossentropy(Loss):
    def forward(self, y_pred, y_true):

        # Clip data to prevent division by 0
        # Clip both sides to not drag mean towards any value
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)

        # Calculate sample-wise loss
        sample_losses = -(y_true * np.log(y_pred_clipped) + (1 - y_true) * np.log(1 - y_pred_clipped))
        sample_losses = np.mean(sample_losses, axis=-1)

        return sample_losses
    
    def backward(self, dvalues, y_true):
        samples = len(dvalues)
        outputs = len(dvalues[0])

        clipped_dvalues = np.clip(dvalues, 1e-7, 1 - 1e-7)
        # Calculate gradient
        self.dinputs = -(y_true / clipped_dvalues - (1 - y_true) / (1 -clipped_dvalues)) / outputs
        # Normalize gradient
        self.dinputs = self.dinputs / samples
    
class Optimizer_Adam:

    # Initialize optimizer
    def __init__(self, learning_rate=0.001, decay=0, epsilon=1e-7,beta_1=0.9,beta_2=0.999):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.epsilon = epsilon # A small value to prevent division by 0
        self.beta_1 = beta_1 # Decay rate for the first moment estimate
        self.beta_2 = beta_2 # Decay rate for the second moment estimate

    def pre_update_params(self):
        if self.decay:
            # Gradually reduces learning rate
            self.current_learning_rate = self.learning_rate * (1 / (1 + self.decay * self.iterations)) 
    
    def update_params(self, layer):
        # If layer does not contain cache arrays, create them filled with zeros
        if not hasattr(layer, 'weight_cache'):
            # Track exponentially weighted averages of past gradients
            layer.weight_momentums = np.zeros_like(layer.weights)
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_momentums = np.zeros_like(layer.biases)
            layer.bias_cache = np.zeros_like(layer.biases)
        
        # Update first moment estimate with gradients of loss w.r.t. weights and biases
        # Smooths the gradients by combining past and current gradients
        layer.weight_momentums = self.beta_1 * layer.weight_momentums + (1 - self.beta_1) * layer.dweights
        layer.bias_momentums = self.beta_1 * layer.bias_momentums + (1 - self.beta_1) * layer.dbiases

        # Get bias-corrected momentums 
        weight_momentums_corrected = layer.weight_momentums / (1-self.beta_1 ** (self.iterations + 1))
        bias_momentums_corrected = layer.bias_momentums / (1-self.beta_1 ** (self.iterations + 1))

        # Update cache with squared current gradients (second moment estimate)
        # Prevents large updates in unstable directions
        layer.weight_cache = self.beta_2 * layer.weight_cache + (1 - self.beta_2) * layer.dweights**2
        layer.bias_cache = self.beta_2 * layer.bias_cache + (1 - self.beta_2) * layer.dbiases**2

        # Get bias-corrected cache (second moment estimate)
        weight_cache_corrected = layer.weight_cache / (1- self.beta_2 ** (self.iterations + 1))
        bias_cache_corrected = layer.bias_cache / (1 - self.beta_2 ** (self.iterations + 1))

        # Final parameter updates
        # Learning rate helps prevent overshooting
        # Momentum helps escape local minima when reaching global maximum
        layer.weights += -self.current_learning_rate * weight_momentums_corrected / (np.sqrt(weight_cache_corrected) + self.epsilon)
        layer.biases += -self.current_learning_rate * bias_momentums_corrected / (np.sqrt(bias_cache_corrected) + self.epsilon)
        
    def post_update_params(self):
        self.iterations += 1


X, y = spiral_data(samples=100, classes=2) # Create dataset in the 2D plane 

y = y.reshape(-1, 1)

# Create dense layer with 2 input features and 128 output values
dense1 = Layer_Dense(2, 64, bias_regularizer_L2=5e-4, weight_regularizer_L2=5e-4)

activation1 = Activation_ReLU()

# Second "hidden" layer with 128 input features and 3 output values
dense2 = Layer_Dense(64, 1)

activation2 = Activation_Sigmoid()

loss_function = Loss_BinaryCrossentropy()

# Create optimizer
optimizer = Optimizer_Adam(learning_rate=0.05, decay=5e-5)

for epoch in range(10001):
    # Perform a forward pass of training data through this layer
    dense1.forward(X)

    activation1.forward(dense1.output) # Make a forward pass through the activation function 

    # Makes a forward pass through second Dense layer
    dense2.forward(activation1.output)

    activation2.forward(dense2.output)

    data_loss = loss_function.calculate(activation2.output, y)

    regularization_loss = loss_function.regularization_loss(dense1) + loss_function.regularization_loss(dense2)

    loss = data_loss + regularization_loss

    # Calculate accuracy from output of activation2 and targets along the first axis (row)
    predictions = (activation2.output > 0.5) * 1
    accuracy = np.mean(predictions==y)

    if (epoch % 100 == 0):
        print(f'epoch: {epoch}, acc: {accuracy:.3f}, loss: {loss:.3f}, lr: {optimizer.current_learning_rate}, data_loss: {data_loss: .3f}, reg_loss: {regularization_loss: .3f}')

    # dvalues = gradient of the loss w.r.t. dense2 logits i.e. softmax outputs
    loss_function.backward(activation2.output, y)
    # dvalues = gradient of loss w.r.t. dense2 outputs
    activation2.backward(loss_function.dinputs)
    dense2.backward(activation2.dinputs)
    # dvalues = gradient of loss w.r.t. ReLU outputs
    activation1.backward(dense2.dinputs)
    # dvalues = gradient of loss w.r.t. dense1 outputs
    dense1.backward(activation1.dinputs)

    optimizer.pre_update_params()
    optimizer.update_params(dense1)
    optimizer.update_params(dense2)
    optimizer.post_update_params()

# Validate the model using test sets to simulate unseen data
X_test, y_test = spiral_data(samples=100, classes=2)

y_test = y_test.reshape(-1, 1)

dense1.forward(X_test)

activation1.forward(dense1.output)

dense2.forward(activation1.output)

activation2.forward(dense2.output)


loss = loss_function.calculate(activation2.output, y_test)

predictions = (activation2.output > 0.5) * 1
accuracy = np.mean(predictions==y) 

print(f'validation, acc: {accuracy: .3f}, loss: {loss: .3f}')