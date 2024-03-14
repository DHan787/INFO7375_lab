class GradientDescent:
    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate

    def update(self, layers, gradients):
        for i, layer in enumerate(layers):
            layer.weights -= self.learning_rate * gradients[f'dW{i+1}']
            layer.biases -= self.learning_rate * gradients[f'db{i+1}']
