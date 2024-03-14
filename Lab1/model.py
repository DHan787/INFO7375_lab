from layers import Layer
from forward_propagation import ForwardPropagation
from backward_propagation import BackwardPropagation

class Model:
    def __init__(self):
        self.layers = []

    def add_layer(self, layer):
        self.layers.append(layer)

    def forward(self, inputs):
        return ForwardPropagation.compute(self.layers, inputs)

    def backward(self, loss_grad):
        BackwardPropagation.compute(self.layers, loss_grad)
