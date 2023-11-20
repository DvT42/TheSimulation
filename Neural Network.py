"""
This file is supposed to include all the mechanical processes needed for the neural networks.

IMPORTANT: Currently the neural network works with vectors, and not batches. I might want to reconsier this in the
future for efficiency purposes.
"""

import numpy as np


class NeuralNetwork:
    def __init__(self):
        self.layers = []

    def add_layer(self, neuron_num, input_num, activation: str):
        self.layers.append(
            NeuronLayer(neuron_num, input_num=input_num, activation=activation)
        )

    def feed(self, input):
        """
        This function runs the neural network.

        Input: a touple or a list of all the inp needed to be passed to the inp layer
        Output: an array of the resulting numbers coming out of the output layer.
        """
        output = self.layers[0].forward(inputs=input)

        for layer in self.layers[1:]:
            layer.forward(inputs=output)
            output = layer.output

        return output

    def __repr__(self):
        return f'{self.layers}'


class NeuronLayer:
    def __init__(self, neuron_num, activation: str, input_num):
        self.neuron_num = neuron_num

        self.weights = 0.1 * np.random.randn(input_num, neuron_num)
        self.biases = np.zeros((1, neuron_num))
        self.output = 0

        self.activation = activation

    def forward(self, inputs):
        """
        This function is to be run on each layer during a feed into the network.

        Input: array of the values fed into the layer, either from the input to the network or from the former layer.
        Output: a tensor representing the output of the entire layer.
        """
        self.output = np.dot(inputs, self.weights) + self.biases
        self.activation_function()

        return self.output

    def activation_function(self):
        if self.activation == 'softmax':
            self.output = NeuronLayer.softmax(self.output)
        elif self.activation == "relu":
            self.output = NeuronLayer.relu(self.output)

    @staticmethod
    def relu(x):
        return np.maximum(x, 0)

    @staticmethod
    def softmax(x):
        """Compute softmax values for each sets of scores in x."""
        pre = x - np.max(x)  # protects my code from exponential overflow. has no impact on the result.

        # If I want to convert to batch-oriented neural network, add axis=1 and keepdims=True to np.sum and np.max.
        return np.exp(pre) / np.sum(np.exp(pre))

    def __repr__(self):
        return f'({self.neuron_num})->|{self.activation}|'
