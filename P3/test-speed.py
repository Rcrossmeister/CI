import numpy as np


vector = np.array([1, 2, 3])
matrix = np.array([[2, 0, 1],
                   [1, 3, 1],
                   [0, 0, 5],
                   [4, 2, 1]])

# 计算内积
inner_product = matrix.dot(vector)

# 找到内积最大的k个值的索引，这里设k为2
k = 2
# 使用argpartition进行部分排序
# 注意argpartition返回的是部分排序的索引，对于我们的用途是足够的
# 我们感兴趣的k个最大值的索引将是数组的最后k个元素
# 因为numpy.argpartition不保证顺序，所以我们还需要对这最后k个索引做一个排序来得到真正的顺序
top_k_indices = np.argpartition(inner_product, -k)[-k:]
# 最后对这k个索引按照内积的实际值降序排序
top_k_indices_sorted = top_k_indices[np.argsort(inner_product[top_k_indices])[::-1]]

top_k_indices_sorted, inner_product[top_k_indices_sorted]
