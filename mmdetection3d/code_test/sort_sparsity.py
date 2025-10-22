import torch
import unittest

class TestGroupMask(unittest.TestCase):
    def test_mask_grouping(self):
        """测试基于索引的分组运算算法的正确性"""
        # 1. 设置测试参数
        batch_size = 10
        feat_dim = 16
        num_groups = 3  # 索引范围 0-2
        
        # 2. 创建测试输入
        # 创建随机特征，形状为 [batch_size, feat_dim]
        features = torch.rand(batch_size, feat_dim)
        
        # 创建固定的分组索引，确保每个组都有样本
        # 我们创建一个确定性的索引分布来验证算法
        indices = torch.tensor([0, 1, 2, 0, 1, 2, 0, 1, 2, 0], dtype=torch.long)
        
        # 3. 创建恒等"网络层"（仅用于测试，实际上只是添加一个标识符）
        identity_layers = []
        for i in range(num_groups):
            # 每个层都是恒等变换，但会添加一个标记以便识别
            identity_layers.append(lambda x, idx=i: x + idx * 0.1)
        
        # 4. 实现分组处理逻辑
        # 初始化输出张量
        output = torch.zeros_like(features)
        
        # 对每个组应用相应的处理
        for idx in range(num_groups):
            # 获取当前组的掩码
            mask = (indices == idx)
            
            if not mask.any():
                continue
                
            # 获取属于当前组的特征
            current_feats = features[mask]
            
            # 应用"网络层"（恒等变换加标识符）
            processed_feats = identity_layers[idx](current_feats)
            
            # 将结果放回原始位置
            output[mask] = processed_feats
        
        # 5. 进行断言验证
        # 遍历每个样本，确保它们被正确地分配到对应的组并处理
        for i in range(batch_size):
            idx = indices[i].item()
            # 验证输出是否等于输入加上组索引标识符
            expected = features[i] + idx * 0.1
            self.assertTrue(torch.allclose(output[i], expected), 
                            f"样本 {i} (组 {idx}) 处理不正确")
        
        # 6. 打印结果以进行直观验证
        print("===== 分组处理测试结果 =====")
        for i in range(batch_size):
            idx = indices[i].item()
            print(f"样本 {i}, 组 {idx}:")
            print(f"  输入: {features[i, 0].item():.4f}")
            print(f"  输出: {output[i, 0].item():.4f}")
            print(f"  期望: {(features[i, 0] + idx * 0.1).item():.4f}")
            print()

        return True

    def test_random_indices(self):
        """使用随机生成的索引测试分组算法"""
        # 参数设置
        batch_size = 12
        feat_dim = 8
        num_groups = 3
        
        # 创建随机特征
        features = torch.rand(batch_size, feat_dim)
        
        # 随机生成索引
        indices = torch.randint(0, num_groups, (batch_size,))
        
        # 创建恒等"网络层"
        identity_layers = []
        for i in range(num_groups):
            identity_layers.append(lambda x, idx=i: x * (idx + 1))  # 乘以(组索引+1)作为唯一标识
        
        # 初始化输出
        output = torch.zeros_like(features)
        
        # 针对每个组进行处理
        for idx in range(num_groups):
            mask = (indices == idx)
            if not mask.any():
                continue
                
            current_feats = features[mask]
            processed_feats = identity_layers[idx](current_feats)
            output[mask] = processed_feats
        
        # 验证结果
        # 构建预期输出
        expected_output = torch.zeros_like(features)
        for i in range(batch_size):
            idx = indices[i].item()
            expected_output[i] = features[i] * (idx + 1)
        
        # 验证输出与预期输出是否相同
        self.assertTrue(torch.allclose(output, expected_output),
                        "随机索引测试失败：输出与预期不符")
        
        # 打印分组统计
        for idx in range(num_groups):
            count = (indices == idx).sum().item()
            print(f"组 {idx} 包含 {count} 个样本")
            
        return True

if __name__ == "__main__":
    unittest.main()