import numpy as np


class Dropout:
    @staticmethod
    def apply(layer_output, keep_prob):
        D = np.random.rand(layer_output.shape[0], layer_output.shape[1]) < keep_prob
        layer_output *= D
        layer_output /= keep_prob
        return layer_output, D
