import torch
import torch.nn as nn
import torch.optim as optim
class Perceptron(nn.Module):
    def __init__(self, input_dim):
        super(Perceptron, self).__init__()
        self.fc = nn.Linear(input_dim, 1)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        return self.sig(self.fc(x))

def train_and_evaluate(data, lr, epochs, avg=False):
    X = data[:, :-1]
    y = data[:, -1].unsqueeze(1)
    learning_rate = lr
    epochs = epochs

    model = Perceptron(input_dim=3)
    criterion = nn.BCELoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    losses = []
    for epoch in range(epochs):
        outputs = model(X)
        loss = criterion(outputs, y)
        losses.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if not avg:
            print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}")

    with torch.no_grad():
        predictions = model(X)
        print(predictions)
        print(y)
        mse = nn.MSELoss()(predictions, y)
        if not avg:
            print(f"Final MSE: {mse.item():.4f}")

    return sum(losses) / len(losses), mse.item()

if __name__ == '__main__':
    data = torch.tensor([
        [24, 56, 78, 1],
        [5, 66, 52, 0],
        [92, 42, 82, 1],
        [31, 63, 67, 1],
        [15, 70, 44, 0],
        [12, 45, 35, 0]
    ], dtype=torch.float32)
    train_and_evaluate(data, 0.01, 3)

