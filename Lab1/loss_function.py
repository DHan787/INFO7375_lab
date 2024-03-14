import numpy as np


class LossFunction:
    @staticmethod
    def mse(predictions, targets):
        m = targets.shape[1]
        loss = np.sum((predictions - targets) ** 2) / m
        return loss

    @staticmethod
    def mse_derivative(predictions, targets):
        m = targets.shape[1]
        dA = -2 * (targets - predictions) / m
        return dA
