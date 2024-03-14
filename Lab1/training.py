class Training:
    def __init__(self, model, optimizer, loss_function):
        self.model = model
        self.optimizer = optimizer  # This could be an instance of GradientDescent or any other optimization class
        self.loss_function = loss_function  # This expects a function, e.g., LossFunction.mse

    def train(self, X_train, Y_train, epochs):
        for epoch in range(epochs):
            # Forward pass
            predictions = self.model.forward(X_train)

            # Compute loss
            loss = self.loss_function(predictions, Y_train)

            # Backward pass to compute gradients
            grads = self.model.backward(Y_train, predictions)

            # Update parameters
            self.optimizer.update(self.model.parameters, grads)

            if epoch % 100 == 0:  # Example: print every 100 epochs
                print(f"Epoch {epoch}, Loss: {loss}")
