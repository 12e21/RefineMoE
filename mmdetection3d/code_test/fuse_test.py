import torch

def normalize_by_column(input_tensor):
    """
    将输入张量按列归一化，即每个元素除以其所在列的总和
    
    参数:
        input_tensor: 形状为[M, N]的输入张量
        
    返回:
        形状为[M, N]的归一化后的张量，每个元素是原始值除以其所在列的总和
    """
    # 计算每列的总和，维度为[1, N]
    column_sums = torch.sum(input_tensor, dim=0, keepdim=True)
    
    # 通过广播机制进行除法运算
    normalized_tensor = input_tensor / column_sums
    
    return normalized_tensor

def weighted_combination(weights, tensor):
    """
    使用权重对张量进行线性组合
    
    参数:
        weights: 形状为[M, N]的权重张量
        tensor: 形状为[M, N, C]的输入张量
        
    返回:
        形状为[N, C]的加权组合后的张量
    """
    # 将weights从[M, N]变形为[M, N, 1]以便广播
    weights_expanded = weights.unsqueeze(-1)  # 形状变为[M, N, 1]
    
    # 对tensor应用权重
    weighted_tensor = tensor * weights_expanded  # 形状[M, N, C]
    
    # 沿着第一个维度(M)求和，得到[N, C]的结果
    result = torch.sum(weighted_tensor, dim=0)  # 形状[N, C]
    
    return result

# 示例演示
# 假设我们有以下输入
M, N, C = 2, 3, 4
scores = torch.tensor([
    [1.0, 2.0, 3.0],
    [4.0, 5.0, 6.0]
], dtype=torch.float32)  # 形状[M, N]

# 生成一个随机的[M, N, C]张量作为示例
features = torch.rand(M, N, C)  # 形状[M, N, C]
print("原始特征张量 (形状 [M, N, C]):")
print(features)


# 计算权重
weights = normalize_by_column(scores)
print("权重张量 (形状 [M, N]):")
print(weights)

# 使用权重进行线性组合
combined_features = weighted_combination(weights, features)
print("\n组合后的特征张量 (形状 [N, C]):")
print(combined_features)

# 验证第一列第一个特征的计算结果
print("\n验证计算:")
col_idx = 0
feature_idx = 0
manual_calc = (features[0, col_idx, feature_idx] * weights[0, col_idx] + 
              features[1, col_idx, feature_idx] * weights[1, col_idx])
print(f"手动计算第{col_idx}列第{feature_idx}个特征: {manual_calc}")
print(f"函数输出的相应值: {combined_features[col_idx, feature_idx]}")