import numpy as np
from activation import ActivationFunction

class BackwardPropagation:
    @staticmethod
    def compute(layers, Y, output):
        m = Y.shape[1]
        dA = -2 * (Y - output) / m  # Derivative of MSE Loss
        grads = {}
        for l in reversed(range(len(layers))):
            layer = layers[l]
            if layer.activation == "relu":
                _, dZ = ActivationFunction.relu(layer.Z)
            elif layer.activation == "sigmoid":
                _, dZ = ActivationFunction.sigmoid(layer.Z)
            dZ *= dA
            A_prev = layers[l-1].A if l > 0 else Y
            grads[f'dW{l+1}'] = np.dot(dZ, A_prev.T)
            grads[f'db{l+1}'] = np.sum(dZ, axis=1, keepdims=True)
            if l > 0:
                dA = np.dot(layers[l].weights.T, dZ)
        return grads
