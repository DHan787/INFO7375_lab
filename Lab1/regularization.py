import numpy as np


class Regularization:
    @staticmethod
    def l2(weights, lambda_val, m):
        l2_cost = 0
        for w in weights.values():
            l2_cost += np.sum(np.square(w))
        l2_cost = (lambda_val / (2 * m)) * l2_cost
        return l2_cost

    @staticmethod
    def l2_grad(weights, lambda_val, m):
        l2_grads = {}
        for key, w in weights.items():
            l2_grads[key] = (lambda_val / m) * w
        return l2_grads
