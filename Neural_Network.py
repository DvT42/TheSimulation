"""
This file is supposed to include all the mechanical processes needed for the neural networks.
"""
import numpy as np


class NeuralNetwork:
    """
        Represents a dense neural network (DNN) model for machine learning tasks.

        This class implements a feedforward neural network architecture with multiple layers of neurons. Each layer applies a non-linear activation function to its inputs and passes the results to the next layer. The network can be trained on a dataset to learn complex patterns and make predictions.

        Attributes:
            layers (list): A list of `Layer` objects representing the different layers of the neural network.
    """
    def __init__(self, layers=None):
        """
            Initializes the neural network with the specified layers.

            This constructor sets up the neural network architecture by defining the layers that make up the network. Each layer could represent a fully connected layer, a convolutional layer, or other types of layers commonly used in deep learning.

            Args:
                layers (list, optional): A list of `Layer` objects defining the network architecture. If None, an empty list is created. Defaults to None.
            """
        self.layers = [] if not layers else layers.copy()

    def add_layer(self, neuron_num, input_num, activation: str):
        """
            Adds a new layer to the existing neural network.

            This method creates a new `NeuronLayer` object with the specified parameters and appends it to the `self.layers` list. The `NeuronLayer` class represents a fully connected layer in the neural network.

            Args:
                neuron_num (int): The number of neurons in the new layer.
                input_num (int): The number of neurons in the previous layer.
                activation (str): The activation function to be applied to the layer.
            """
        self.layers.append(
            NeuronLayer(neuron_num, input_num=input_num, activation=activation)
        )

    def feed(self, input):
        """
        This function runs the neural network.

        Input: a tuple or a list of all the inp needed to be passed to the inp layer
        Output: an array of the resulting numbers coming out of the output layer.
        """
        output = self.layers[0].forward(inputs=input)

        for layer in self.layers[1:]:
            layer.forward(inputs=output)
            output = layer.output

        return output

    def get_weights(self):
        """
            Retrieves all the weights of the neural network.

            This method iterates through the layers of the network and extracts the weights from each layer. The weights represent the connection strengths between neurons in different layers.

            Returns:
                list: A list of weight matrices, where each matrix corresponds to the weights of a layer.
            """
        return [layer.weights for layer in self.layers]

    def get_biases(self):
        """
            Retrieves all the biases of the neural network.

            This method iterates through the layers of the network and extracts the biases from each layer. The biases represent the constant values added to the weighted sums of inputs before applying the activation function.

            Returns:
                list: A list of bias vectors, where each vector corresponds to the biases of a layer.
            """
        return [layer.biases for layer in self.layers]

    def set_weights(self, new_weights):
        """
            Replaces the existing weights of the neural network with the provided weights.

            This method iterates through the layers of the network and updates their weights with the corresponding values from the `new_weights` list. The `new_weights` list should contain weight matrices for each layer, matching the structure and dimensions of the original weights.

            Args:
                new_weights (list): A list of weight matrices, where each matrix corresponds to the weights of a layer.
            """

        for index, layer in enumerate(self.layers):
            layer.weights = new_weights[index]

    def set_biases(self, new_biases):
        """
            Replaces the existing biases of the neural network with the provided biases.

            This method iterates through the layers of the network and updates their biases with the corresponding values from the `new_biases` list. The `new_biases` list should contain bias vectors for each layer, matching the length of the original biases.

            Args:
                new_biases (list): A list of bias vectors, where each vector corresponds to the biases of a layer.
            """
        for index, layer in enumerate(self.layers):
            layer.biases = new_biases[index]

    def __repr__(self):
        """
            Represents the neural network as a string for debugging purposes.

            This method provides a detailed representation of the neural network's architecture and parameters. It includes information about the layers, their activation functions, weights, and biases.

            Returns:
                str: A string representation of the neural network.
            """
        return f'{self.layers}'

    def copy(self):
        """
            Creates a deep copy of the current neural network.

            This method generates a new neural network object with an identical structure and parameters as the original network. It performs a deep copy of the network's layers, ensuring that any modifications made to the copy do not affect the original network.

            Returns:
                NeuralNetwork: A new neural network object that is a deep copy of the original.
            """
        return NeuralNetwork(self.layers)


class NeuronLayer:
    """
        Represents a single layer of neurons in a neural network.

        This class encapsulates the essential operations of a neural network layer, including:

        * **Storing neuron parameters:**
            * `neuron_num`: The number of neurons in the layer.
            * `weights`: A 2D matrix representing the weights connecting neurons in this layer to neurons in the previous layer.
            * `biases`: A 1D vector representing the biases for each neuron in this layer.

        * **Performing forward propagation:**
            * `forward()`: Computes the weighted sum of inputs and applies the activation function to produce the layer's output.
            * `output`: Stores the resulting output values for each neuron in the layer.

        * **Storing activation function:**
            * `activation`: A string specifying the activation function to be applied to the layer's output (e.g., "ReLU", "Sigmoid").

        This class provides the foundation for building more complex neural network architectures.
        """

    def __init__(self, neuron_num, activation: str, input_num):
        """
            Initializes the NeuronLayer object.

            This method sets up the essential parameters for the neuron layer, including:

            Args:
                neuron_num (int): Number of neurons in the layer.
                activation (str): Activation function to apply to the layer's output.
                input_num (int): Number of neurons in the previous layer or input data size.
            """
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
        """
            Applies the activation function specified for the layer to its output.

            This method retrieves the activation function string stored in the `self.activation` attribute and dynamically calls the corresponding static activation function from the `Activations` module. It then applies this function to the layer's output (`self.output`).

            Returns:
                None: The method doesn't explicitly return any value.
            """
        if self.activation == 'softmax':
            self.output = NeuronLayer.softmax(self.output)
        elif self.activation == "relu":
            self.output = NeuronLayer.relu(self.output)
        elif self.activation == "sigmoid":
            self.output = NeuronLayer.sigmoid(self.output)

    @staticmethod
    def relu(x):
        """
            Applies the ReLU (Rectified Linear Unit) activation function to the input.

            This function implements the ReLU activation function, which replaces negative inputs with zero and passes positive inputs through unchanged. It is a common activation function in neural networks due to its simplicity and effectiveness.

            Args:
                x (numpy.ndarray or float): The input value or array to apply the ReLU function to.

            Returns:
                numpy.ndarray or float: The output after applying the ReLU function.
            """

        return np.maximum(x, 0)

    @staticmethod
    def softmax(x):
        """
            Applies the softmax activation function to the input.

            This function implements the softmax activation function, which normalizes a vector of real numbers into a probability distribution. It is commonly used in multi-class classification tasks in neural networks.

            Args:
                x (numpy.ndarray or float): The input value or array to apply the softmax function to.

            Returns:
                numpy.ndarray or float: The output after applying the softmax function.
        """
        pre = x - np.max(x)  # protects my code from exponential overflow. has no impact on the result.

        # If I want to convert to batch-oriented neural network, add axis=1 and keepdims=True to np.sum and np.max.
        return np.exp(pre) / np.sum(np.exp(pre))

    @staticmethod
    def sigmoid(x):
        """
            Applies the sigmoid activation function to the input.

            This function implements the sigmoid activation function, also known as the logistic function. It squashes each element of the input between 0 and 1, making it suitable for tasks like binary classification.

            Args:
                x (numpy.ndarray or float): The input value or array to apply the sigmoid function to.

            Returns:
                numpy.ndarray or float: The output after applying the sigmoid function.
            """
        return 1 / (1 + np.exp(-x))

    def __repr__(self):
        """
            Represents the NeuronLayer object as a string.

            This method provides a concise and informative string representation of the neuron layer, including its essential attributes.

            Returns:
                str: The string representation of the neuron layer.
            """
        return f'({self.neuron_num})->|{self.activation}|'
