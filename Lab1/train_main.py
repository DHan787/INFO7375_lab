import string

from Lab1.data_Gen import handwriting_recognition_data_generator
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
    # Initialize the data generator
    data_generator = handwriting_recognition_data_generator(
        chars=string.ascii_letters + string.digits,  # Characters to include
        batch_size=32,  # Number of images/labels per batch
        image_size=(28, 28),  # Size of the images
        font_path='HappySwirly-KVB7l.ttf'  # Path to the font file
    )



    # Define the number of epochs for training
    epochs = 1000  # Or any number you deem appropriate

    for epoch in range(epochs):
        # print(f"Epoch {epoch + 1}/{epochs}")
        for step in range(epochs):
            # Generate a batch of data
            X_batch, Y_batch = next(data_generator)

    # Start training your model
    training_instance.train(X_batch, Y_batch, epochs)

if __name__ == "__main__":
    main()
