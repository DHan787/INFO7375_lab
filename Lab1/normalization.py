import numpy as np


class Normalization:
    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, data):

        self.mean = np.mean(data, axis=0)
        self.std = np.std(data, axis=0)

    def transform(self, data):

        if self.mean is None or self.std is None:
            raise ValueError("Normalization parameters have not been computed. Call fit() first.")
        normalized_data = (data - self.mean) / self.std
        return normalized_data

    def fit_transform(self, data):

        self.fit(data)
        return self.transform(data)
