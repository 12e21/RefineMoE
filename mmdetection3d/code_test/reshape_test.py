import torch

# 假设 m=2, n=3, k=4
m, n, k = 2, 3, 4
tensor = torch.randn(m, n, k)  # 创建一个形状为 [m, n, k] 的张量

# 使用 view 或 reshape 将其重塑为 [mxn, k]
reshaped_tensor = tensor.reshape(-1,tensor.shape[-1]) # 或者使用 tensor.reshape(m * n, k)
import ipdb;ipdb.set_trace()
print(reshaped_tensor.shape)  # 输出应为 torch.Size([6, 4])