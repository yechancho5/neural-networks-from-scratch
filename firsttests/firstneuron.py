# Coding the first neuron

inputs = [1, 2, 3, 2.5]
weights = [[0.2, 0.8, -0.5, 1], [0.5, -0.91, 0.26, -0.5], [-0.26, -0.27, 0.17, 0.87]]
biases = [2, 3, 0.5]

layer_outputs = []
# for each neuron
for neuron_weights, neuron_bias in zip(weights, biases): # groups together the weights and biases
    # Zerod output of given neuron
    neuron_output = 0
    for n_input, weight in zip(inputs, neuron_weights): # groups together the inputs and neuron weights [[1, 0.2], [2, 0.8]...]
        neuron_output += n_input*weight
        print(neuron_output)
    # add the bias
    neuron_output+= neuron_bias

    layer_outputs.append(neuron_output)

print(layer_outputs)

