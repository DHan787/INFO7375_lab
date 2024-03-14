import numpy as np
from activation import ActivationFunction as Activation

class Layer:
    def __init__(self, input_size, output_size, activation_func):
        self.weights = np.random.randn(input_size, output_size) * 0.01
        self.bias = np.zeros((1, output_size))
        self.activation_func = activation_func
        self.output = None
        self.input = None
        self.activation_output = None

    def forward(self, inputs):
        self.input = inputs
        self.output = np.dot(inputs, self.weights) + self.bias
        if self.activation_func == 'relu':
            self.activation_output = Activation.relu(self.output)
        elif self.activation_func == 'sigmoid':
            self.activation_output = Activation.sigmoid(self.output)
        return self.activation_output
