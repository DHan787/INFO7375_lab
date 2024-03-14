import numpy as np

class ForwardPropagation:
    @staticmethod
    def compute(layers, X):
        A = X
        for layer in layers:
            A_prev = A
            Z = np.dot(layer.weights, A_prev) + layer.b
            layer.Z = Z
            layer.activate(Z)
            A = layer.A
        return A
