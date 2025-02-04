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
        for index, (single_output, single_dvalues) in enumerate(zip(self.output, dvalues)):
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
    def __init__(self, learning_rate=1.0, decay=0, momentum=0):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.momentum = momentum

    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * (1. / (1. + self.decay * self.iterations))
    
    def update_params(self, layer):
        
        # if momentum is used
        if self.momentum: 

            # If layer does not contain momentum arrays, create them filled with zeroes
            if not hasattr(layer, 'weight_momentums'):
                layer.weight_momentums = np.zeros_like(layer.weights)
                # If there is no momentum array for weights, there none for biases
                layer.bias_momentums = np.zeros_like(layer.biases)

            # Build weight updates with momentum using previous update's direction to minimize chances of getting stuck at local min
            # multiply by retain factor and update w/ current gradients
            weight_updates = self.momentum * layer.weight_momentums - self.current_learning_rate * layer.dweights
            layer.weight_momentums = weight_updates

            # Build bias updates
            bias_updates = self.momentum * layer.bias_momentums - self.current_learning_rate * layer.dbiases
            layer.bias_momentums = bias_updates
        # SGD updates (before momentum updates)
        else:
            weight_updates = -self.current_learning_rate * layer.dweights # W = W - learning rate * gradient of loss w.r.t. weights
            bias_updates = -self.current_learning_rate * layer.dbiases # B = B - learning rate * gradient of loss w.r.t. biases

        # Update weights and biases using momentum or vanilla
        layer.weights += weight_updates
        layer.biases += bias_updates
    
    def post_update_params(self):
        self.iterations += 1

class Optimizer_Adam:

    # Initialize optimizer
    def __init__(self, learning_rate=0.001, decay=0, epsilon=1e-7,beta_1=0.9,beta_2=0.999):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.epsilon = epsilon
        self.beta_1 = beta_1
        self.beta_2 = beta_2

    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * (1 / (1 + self.decay * self.iterations))
    
    def update_params(self, layer):
        # If layer does not contain cache arrays, create them filled with zeros
        if not hasattr(layer, 'weight_cache'):
            layer.weight_momentums = np.zeros_like(layer.weights)
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_momentums = np.zeros_like(layer.biases)
            layer.bias_cache = np.zeros_like(layer.biases)
        
        layer.weight_momentums = self.beta_1 * layer.weight_momentums + (1 - self.beta_1) * layer.dweights
        layer.bias_momentums = self.beta_1 * layer.bias_momentums + (1 - self.beta_1) * layer.dbiases

        # Get corrected momentum
        weight_momentums_corrected = layer.weight_momentums / (1-self.beta_1 ** (self.iterations + 1))
        bias_momentums_corrected = layer.bias_momentums / (1-self.beta_1 ** (self.iterations + 1))

        # Update cache with squared current gradients
        layer.weight_cache = self.beta_2 * layer.weight_cache + (1 - self.beta_2) * layer.dweights**2
        layer.bias_cache = self.beta_2 * layer.bias_cache + (1 - self.beta_2) * layer.dbiases**2

        # Get corrected cache
        weight_cache_corrected = layer.weight_cache / (1- self.beta_2 ** (self.iterations + 1))
        bias_cache_corrected = layer.bias_cache / (1 - self.beta_2 ** (self.iterations + 1))

        # Vanilla SGD parameter update + normalization with square rooted cache
        layer.weights += -self.current_learning_rate * weight_momentums_corrected / (np.sqrt(weight_cache_corrected) + self.epsilon)
        layer.biases += -self.current_learning_rate * bias_momentums_corrected / (np.sqrt(bias_cache_corrected) + self.epsilon)
        
    def post_update_params(self):
        self.iterations += 1

class Layer_Dropout:
    def __init__(self, rate):
        # Store rate, invert it
        self.rate = 1 - rate
    
    def forward(self, inputs):
        # Save input values
        self.inputs = inputs
        # Generate and save scaled mask
        self.binary_mask = np.random.binomial(1, self.rate, size=inputs.shape) / self.rate
        # Apply mask
        self.output = inputs * self.binary_mask

    def backward(self, dvalues):
        # Gradient on values
        self.dinputs = dvalues * self.binary_mask

X, y = spiral_data(samples=1000, classes=3) # Create dataset in the 2D plane (100 samples, 2 features)

# Create dense layer with 2 input features and 512 output values
dense1 = Layer_Dense(2, 512, bias_regularizer_L2=5e-4, weight_regularizer_L2=5e-4)

activation1 = Activation_ReLU()

dropout1 = Layer_Dropout(0.1)

# Second "hidden" layer with 512 input features and 3 output values
dense2 = Layer_Dense(512, 3)

# Create Softmax classifier's combined loss and activation
loss_activation = Activation_Softmax_Loss_CategoricalCrossentropy()

# Create optimizer
optimizer = Optimizer_Adam(learning_rate=0.05, decay=5e-5)

for epoch in range(10001):
    # Perform a forward pass of training data through this layer
    dense1.forward(X)

    activation1.forward(dense1.output) # Make a forward pass through the activation function 

    dropout1.forward(activation1.output)

    # Makes a forward pass through second Dense layer
    dense2.forward(dropout1.output)

    data_loss = loss_activation.forward(dense2.output, y)

    regularization_loss = loss_activation.loss.regularization_loss(dense1) + loss_activation.loss.regularization_loss(dense2)

    loss = data_loss + regularization_loss

    # Calculate accuracy from output of activation2 and targets along the first axis (row)
    predictions = np.argmax(loss_activation.output,axis=1)
    if len(y.shape) == 2:
        y = np.argmax(y, axis=1) 
    accuracy = np.mean(predictions==y) # returns an array of Boolean values and converts them to True = 1, False = 0 then takes mean

    if (epoch % 100 == 0):
        print(f'epoch: {epoch}, acc: {accuracy:.3f}, loss: {loss:.3f}, lr: {optimizer.current_learning_rate}, data_loss: {data_loss: .3f}, reg_loss: {regularization_loss: .3f}')

    # dvalues = gradient of the loss w.r.t. dense2 logits i.e. softmax outputs
    loss_activation.backward(loss_activation.output, y)
    # dvalues = gradient of loss w.r.t. dense2 outputs
    dense2.backward(loss_activation.dinputs)
    dropout1.backward(dense2.dinputs)
    # dvalues = gradient of loss w.r.t. ReLU outputs
    activation1.backward(dropout1.dinputs)
    # dvalues = gradient of loss w.r.t. dense1 outputs
    dense1.backward(activation1.dinputs)

    optimizer.pre_update_params()
    optimizer.update_params(dense1)
    optimizer.update_params(dense2)
    optimizer.post_update_params()

# validate the model

X_test, y_test = spiral_data(samples=100, classes=3)

dense1.forward(X_test)

activation1.forward(dense1.output)

dense2.forward(activation1.output)

loss = loss_activation.forward(dense2.output, y_test)

predictions = np.argmax(loss_activation.output,axis=1)
if len(y_test.shape) == 2:
    y_test= np.argmax(y_test, axis=1) 
accuracy = np.mean(predictions==y_test)

print(f'validation, acc: {accuracy: .3f}, loss: {loss: .3f}')