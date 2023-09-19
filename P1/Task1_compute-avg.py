from Task1_main import train_and_evaluate
import torch

losses = []
mses = []
times = 50
data = torch.tensor([
        [24, 56, 78, 1],
        [5, 66, 52, 0],
        [92, 42, 82, 1],
        [31, 63, 67, 1],
        [15, 70, 44, 0],
        [12, 45, 35, 0]
    ], dtype=torch.float32)

for _ in range(times):
    avg_loss, mse = train_and_evaluate(data, 0.01, 3, avg=True)
    losses.append(avg_loss)
    mses.append(mse)

average_loss = sum(losses) / times
average_mse = sum(mses) / times

print(average_loss)
print(average_mse)
