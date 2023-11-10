import torch

# 假设有以下向量和矩阵
vector = torch.tensor([1, 2, 3])
matrix = torch.tensor([[2, 0, 1],
                       [1, 3, 1],
                       [0, 0, 5],
                       [4, 2, 1]], dtype=torch.float32)  # dtype指定为float32以适配torch.matmul

# 计算内积
inner_product = torch.matmul(matrix, vector)

# 找到内积最大的k个值的索引，这里设k为2
k = 2
# 使用torch.topk来得到最大的k个值和索引
topk_values, topk_indices = torch.topk(inner_product, k)

topk_indices, topk_values
