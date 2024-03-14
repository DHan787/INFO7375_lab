import numpy as np


class Parameters:
    def __init__(self, layer_dims):
        self.weights = {}
        self.biases = {}
        for l in range(1, len(layer_dims)):
            self.weights[f'W{l}'] = np.random.randn(layer_dims[l], layer_dims[l-1]) * 0.01
            self.biases[f'b{l}'] = np.zeros((layer_dims[l], 1))
