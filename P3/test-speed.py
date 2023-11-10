import numpy as np
import torch
import time

# 生成数据
np_vector = np.random.rand(768)
np_matrix = np.random.rand(10000, 768)

# NumPy实现及时间测量
start_time_np = time.time()
np_inner_product = np_matrix.dot(np_vector)
np_top5_indices = np.argsort(np_inner_product)[-5:][::-1]
end_time_np = time.time()
np_time_taken = end_time_np - start_time_np

# 转换成PyTorch数据类型
torch_vector = torch.tensor(np_vector, dtype=torch.float32)
torch_matrix = torch.tensor(np_matrix, dtype=torch.float32)

# PyTorch实现及时间测量
start_time_torch = time.time()
torch_inner_product = torch.matmul(torch_matrix, torch_vector)
torch_top5_values, torch_top5_indices = torch.topk(torch_inner_product, 5)
end_time_torch = time.time()
torch_time_taken = end_time_torch - start_time_torch

np_time_taken, torch_time_taken, np_top5_indices, torch_top5_indices
