import numpy as np


class Neuron:
    def __init__(self, weights, bias, activation_func):
        self.weights = weights
        self.bias = bias
        self.activation_func = activation_func

    def activate(self, inputs):
        # Simplified representation of the weighted input and activation for a single neuron
        z = np.dot(self.weights, inputs) + self.bias
        return self.activation_func(z)
