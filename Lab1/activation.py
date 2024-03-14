import numpy as np

class ActivationFunction:
    @staticmethod
    def relu(x):
        return np.maximum(0, x), np.where(x > 0, 1, 0)

    @staticmethod
    def sigmoid(x):
        exp_x = np.exp(-x)
        return 1 / (1 + exp_x), exp_x / ((1 + exp_x) ** 2)
