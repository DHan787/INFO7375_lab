from model import Model
from layers import Layer
from activation import ActivationFunction
from normalization import Normalization
from training import Training
from gradient_descent import GradientDescent
from loss_function import LossFunction
import numpy as np


def main():
    # Model setup

    model = Model()
    model.add_layer(Layer(10, 1,activation_func=ActivationFunction.relu))  # Layer 1
    model.add_layer(Layer(8,1, activation_func=ActivationFunction.relu))  # Layer 2
    model.add_layer(Layer(8,1, activation_func=ActivationFunction.relu))  # Layer 3
    model.add_layer(Layer(4, 1,activation_func=ActivationFunction.relu))  # Layer 4
    model.add_layer(Layer(1,1, activation_func=ActivationFunction.sigmoid))  # Layer 5

    # Data setup
    X_train = np.random.randn(100, 10)
    Y_train = np.random.randint(2, size=(1, 100))

    # Normalize data
    normalizer = Normalization()
    X_train_normalized = normalizer.fit_transform(X_train)

    # Training setup
    optimizer = GradientDescent(learning_rate=0.01)
    trainer = Training(model=model, optimizer=optimizer, loss_function=LossFunction.mse)

    # Train the model
    trainer.train(X_train_normalized, Y_train, epochs=1000)


if __name__ == "__main__":
    main()
