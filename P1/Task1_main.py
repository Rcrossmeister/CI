import torch


class ManualPerceptron:
    def __init__(self, input_dim):
        self.weights = torch.randn(input_dim, 1) * 0.01
        self.bias = torch.randn(1) * 0.01

    def sigmoid(self, x):
        return 1 / (1 + torch.exp(-x))

    def forward(self, x):
        linear_output = torch.mm(x, self.weights) + self.bias
        return self.sigmoid(linear_output)

    def squared_loss(self, y_true, y_pred):
        return torch.mean((y_true - y_pred) ** 2)

    def mse(self, y_true, y_pred):
        # Manual computation of Mean Squared Error
        return torch.mean((y_true - y_pred) ** 2)

    def train(self, X, y, learning_rate, epochs):
        for epoch in range(epochs):
            # Forward pass
            predictions = self.forward(X)

            # Compute BCE loss
            loss = self.squared_loss(y, predictions)

            # Compute MSE
            mse_value = self.mse(y, predictions)

            # Compute gradients
            dloss_dpred = -(y / predictions) + (1 - y) / (1 - predictions)
            dpred_dlinear = predictions * (1 - predictions)
            dloss_dlinear = dloss_dpred * dpred_dlinear
            dloss_dweights = torch.mm(X.t(), dloss_dlinear)
            dloss_dbias = torch.sum(dloss_dlinear)

            # Update weights and bias using SGD
            self.weights -= learning_rate * dloss_dweights
            self.bias -= learning_rate * dloss_dbias

            # Print the BCE loss and MSE for each epoch
            print(f"Epoch [{epoch + 1}/{epochs}], Squared Loss: {loss.item():.4f}, MSE: {mse_value.item():.4f}")

        # Return the final weights and bias
        return self.weights, self.bias

if __name__ == '__main__':
    data = torch.tensor([
        [24, 56, 78, 1],
        [5, 66, 52, 0],
        [92, 42, 82, 1],
        [31, 63, 67, 1],
        [15, 70, 44, 0],
        [12, 45, 35, 0]
    ], dtype=torch.float32)

    X = data[:, :-1]
    y = data[:, -1].unsqueeze(1)

    model = ManualPerceptron(input_dim=3)
    final_weights_sigmoid_mse, final_bias_sigmoid_mse = model.train(X, y, learning_rate=0.0001, epochs=3)


