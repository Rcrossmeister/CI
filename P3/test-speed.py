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

# 定义函数以便于重复测试
def numpy_inner_product_topk(vector, matrix, k=5):
    start_time = time.time()
    inner_product = matrix.dot(vector)
    topk_indices = np.argsort(inner_product)[-k:][::-1]
    end_time = time.time()
    return end_time - start_time

def torch_inner_product_topk(vector, matrix, k=5):
    start_time = time.time()
    inner_product = torch.matmul(matrix, vector)
    topk_values, topk_indices = torch.topk(inner_product, k)
    end_time = time.time()
    return end_time - start_time

# 运行1000次测试并计算平均速度
num_trials = 1000
numpy_times = [numpy_inner_product_topk(np_vector, np_matrix) for _ in range(num_trials)]
torch_times = [torch_inner_product_topk(torch_vector, torch_matrix) for _ in range(num_trials)]

avg_numpy_time = np.mean(numpy_times)
avg_torch_time = np.mean(torch_times)

print(avg_numpy_time)
print(avg_torch_time)
