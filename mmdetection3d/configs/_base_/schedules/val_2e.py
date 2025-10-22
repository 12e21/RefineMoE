# 设置初始学习率
lr = 0.0018

# 使用 AdamW 优化器，保留合理参数
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=lr, betas=(0.9, 0.999), weight_decay=0.01),
    clip_grad=dict(max_norm=10, norm_type=2)
)

# 学习率与动量调度（短周期）
param_scheduler = [
    # 学习率先上升后下降，适应2个epoch
    dict(
        type='CosineAnnealingLR',
        T_max=2,
        eta_min=lr * 0.1,  # 最小学习率为初始的 1/10
        begin=0,
        end=2,
        by_epoch=True,
        convert_to_iter_based=True
    ),
    # 动量从 0.85 到 0.95 缓慢提升
    dict(
        type='CosineAnnealingMomentum',
        T_max=2,
        eta_min=0.85,
        begin=0,
        end=2,
        by_epoch=True,
        convert_to_iter_based=True
    )
]

# 训练设置：只运行 2 个 epoch，每个 epoch 验证一次
train_cfg = dict(by_epoch=True, max_epochs=2, val_interval=1)
val_cfg = dict()
test_cfg = dict()

# 自动缩放学习率（可选关闭）
auto_scale_lr = dict(enable=False, base_batch_size=48)