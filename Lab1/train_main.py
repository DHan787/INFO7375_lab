from model import Model
from activation import ActivationFunction
from loss_function import LossFunction
from training import Training

def main():
    # Initialize your model, optimizer, and loss function
    model = Model()
    active = ActivationFunction()# Make sure to initialize your optimizer with any required parameters
    loss_function = LossFunction.mse  # Or replace with your loss function

    # Create an instance of the Training class
    training_instance =Training(model, active, loss_function)

    # Load or generate your training data
    X_train = ...  # Your training inputs
    Y_train = ...  # Your training targets

    # Define the number of epochs for training
    epochs = 1000  # Or any number you deem appropriate

    # Start training your model
    training_instance.train(X_train, Y_train, epochs)

if __name__ == "__main__":
    main()
