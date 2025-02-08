import nnfs
from nnfs.datasets import spiral_data, sine_data
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

    def forward(self, inputs, training):
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
 
class Layer_Dropout:
    def __init__(self, rate):
        # Store rate, invert it
        self.rate = 1 - rate
    
    def forward(self, inputs, training):
        # Save input values
        self.inputs = inputs

        # If not in training mode - return values
        if not training:
            self.output = inputs.copy()
            return

        # Generate and save scaled mask
        self.binary_mask = np.random.binomial(1, self.rate, size=inputs.shape) / self.rate
        # Apply mask
        self.output = inputs * self.binary_mask

    def backward(self, dvalues):
        # Gradient on values
        self.dinputs = dvalues * self.binary_mask

class Layer_Input:
    def forward(self, inputs, training):
        self.output = inputs

class Activation_ReLU: # f(x) = max(x, 0)
    # Forward pass
    def forward(self, inputs, training):
        self.inputs = inputs
        self.output = np.maximum(0, inputs)
    # Backward pass
    def backward(self, dvalues):
        #dvalues is dense2.inputs, the gradient of the loss (prev. layer) w.r.t. the outputs of the ReLU function
        self.dinputs = dvalues.copy()
        # Zero gradient where input values are negative
        self.dinputs[self.inputs <= 0] = 0
    def predictions(self, outputs):
        return outputs
    
class Activation_Softmax: 
    def forward(self, inputs, training):
        self.inputs = inputs
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

    def predictions(self, outputs):
        return np.argmax(outputs, axis=1)
    

class Activation_Sigmoid:
    def forward(self, inputs, training):
        self.inputs = inputs
        self.output = 1 / (1+np.exp(-inputs)) # inputs = z
    
    def backward(self, dvalues):
        self.dinputs = dvalues * (1 - self.output) * self.output

    def predictions(self, outputs):
        return (outputs > 0.5) * 1

class Activation_Linear:
    def forward(self, inputs, training):
        self.inputs = inputs
        self.output = inputs

    def backward(self, dvalues):
        # Derivative is 1, 1 * dvalues = dvalues by chain rule
        self.dinputs = dvalues.copy()

    def predictions(self, outputs):
        return outputs

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

class Loss: 

    def regularization_loss(self):
        # 0 by default
        regularization_loss = 0

        for layer in self.trainable_layers:

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
    
    
    def remember_trainable_layers(self, trainable_layers):
        self.trainable_layers = trainable_layers
    
    def calculate(self, output, y, *, include_regularization=False):
        
        sample_losses = self.forward(output, y)
       
        # Calculate mean loss
        data_loss = np.mean(sample_losses)

        if not include_regularization:
            return data_loss

        return data_loss, self.regularization_loss()
    
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

class Loss_MeanSquaredError(Loss):
    def forward(self, y_pred, y_true):
        sample_losses = np.mean((y_true - y_pred)**2, axis=-1)
        return sample_losses
    
    def backward(self, dvalues, y_true):
        samples = len(dvalues)
        outputs = len(dvalues[0])

        # Gradient on values
        self.dinputs = -2 * (y_true - dvalues) / outputs
        # Normalize gradient
        self.dinputs = self.dinputs / samples

class Loss_MeanAbsoluteError(Loss):
    def forward(self, y_pred, y_true):
        sample_losses = np.mean(np.abs(y_true - y_pred), axis=1)
        return sample_losses
    
    def backward(self, dvalues, y_true):
        samples = len(dvalues)
        outputs = len(dvalues[0])
        
        self.dinputs = np.sign(y_true - dvalues) / outputs
        self.dinputs = self.dinputs / samples

class Accuracy: 
    def calculate(self, predictions, y):

        # Get comparison results
        comparisons = self.compare(predictions, y)

        # Calculate accuracy
        accuracy = np.mean(comparisons)

        return accuracy

class Accuracy_Categorical(Accuracy):
    def init(self, y):
        pass

    # Compares predictions to ground truth values
    def compare(self, predictions, y):
        if len(y.shape) == 2:
            y = np.argmax(y, axis=1)
        return predictions == y

class Accuracy_Regression(Accuracy):
    def init(self):
        self.precision = None # Create precision property
    
    # Calculates precision value
    def init(self, y, reinit=False):
        if self.precision is None or reinit:
            self.precision = np.std(y) / 250
    
    def compare(self, predictions, y):
        return np.absolute(predictions - y) < self.precision
    
class Model:
    def __init__(self): 
        # Create a list of network objects
        self.layers = []
        self.softmax_classifier_output = None
    
    def add(self, layer):
        # Add objects to model
        self.layers.append(layer)

    def set(self, *, loss, optimizer, accuracy):
        self.loss = loss
        self.optimizer = optimizer
        self. accuracy = accuracy
    
    # Finalize the model
    def finalize(self):

        self.input_layer = Layer_Input()

        # Count all objects
        layer_count = len(self.layers)

        self.trainable_layers = []

        for i in range(layer_count):
            if i == 0:
                self.layers[i].prev = self.input_layer
                self.layers[i].next = self.layers[i+1]
            
            # All layers except first and last
            elif i < layer_count - 1:
                self.layers[i].prev = self.layers[i-1]
                self.layers[i].next = self.layers[i+1]
            
            # Last layer where the next object is the loss
            else: 
                self.layers[i].prev = self.layers[i-1]
                self.layers[i].next = self.loss
                self.output_layer_activation = self.layers[i]
        
        if hasattr(self.layers[i], 'weights'):
            self.trainable_layers.append(self.layers[i])
        
        # Update loss object with trainable layers
        self.loss.remember_trainable_layers(self.trainable_layers)

        if isinstance(self.layers[-1], Activation_Softmax) and isinstance(self.loss, Loss_CategoricalCrossentropy):
            # Create an object that combines both for faster calculations
            self.softmax_classifier_output = Activation_Softmax_Loss_CategoricalCrossentropy()

    def train(self, X, y, *, epochs=1, print_every=1, validation_data=None):

        # Initialize accuracy object
        self.accuracy.init(y)

        for epoch in range(1, epochs+1):
            
            output = self.forward(X, training=True)
            
            data_loss, regularization_loss = self.loss.calculate(output, y, include_regularization=True)
            loss = data_loss + regularization_loss

            # Get predictions and calculate an accuracy
            predictions = self.output_layer_activation.predictions(output)
            accuracy = self.accuracy.calculate(predictions, y)

            # Backward pass
            self.backward(output, y)

            self.optimizer.pre_update_params()
            for layer in self.trainable_layers:
                self.optimizer.update_params(layer)
            self.optimizer.post_update_params() 

            if (epoch % 100 == 0):
                print(f'epoch: {epoch}, acc: {accuracy:.3f}, loss: {loss:.3f}, lr: {self.optimizer.current_learning_rate}, data_loss: {data_loss: .3f}, reg_loss: {regularization_loss: .3f}')
            
            if validation_data is not None:
                X_val, y_val = validation_data

                output = self.forward(X_val, training=False)

                loss = self.loss.calculate(output, y_val)

                predictions = self.output_layer_activation.predictions(output)

                accuracy = self.accuracy.calculate(predictions, y_val)

    def forward(self, X, training):
        self.input_layer.forward(X, training)

        for layer in self.layers:
            layer.forward(layer.prev.output, training)

        return layer.output

    def backward(self, output, y):
        if self.softmax_classifier_output is not None:
            # First call backward method on combined activation/loss
            self.softmax_classifier_output.backward(output, y)

            # Since not going to last layer, set dinputs in this one
            self.layers[-1].dinputs = self.softmax_classifier_output.dinputs
        
            # Call backward method going through all layers but last in reversed order
            for layer in reversed(self.layers[:-1]):
                layer.backward(layer.next.dinputs)
        
            return 
        
        # First call backward method on the loss
        self.loss.backward(output, y)

        for layer in reversed(self.layers):
            layer.backward(layer.next.dinputs)


X, y = spiral_data(samples=1000, classes=3)
X_test, y_test, = spiral_data(samples=100, classes=3)

model = Model()

# add layers
model.add(Layer_Dense(2, 256, bias_regularizer_L2=5e-4, weight_regularizer_L2=5e-4))

model.add(Activation_ReLU())
model.add(Layer_Dropout(0.1))
model.add(Layer_Dense(256, 3))
model.add(Activation_Softmax())

model.set(loss=Loss_CategoricalCrossentropy(), optimizer=Optimizer_Adam(learning_rate=0.05, decay=5e-5), accuracy=Accuracy_Categorical())

model.finalize()

model.train(X, y, validation_data=(X_test, y_test), epochs=10000, print_every=100)


