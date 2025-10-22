lr = 0.0018

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=lr, betas=(0.95, 0.99), weight_decay=0.01),
    clip_grad=dict(max_norm=10, norm_type=2)
)

param_scheduler = [
    # Learning rate schedule
    dict(
        type='CosineAnnealingLR',
        T_max=24,
        eta_min=lr * 10,
        begin=0,
        end=24,
        by_epoch=True,
        convert_to_iter_based=True),
    dict(
        type='CosineAnnealingLR',
        T_max=36,
        eta_min=lr * 1e-4,
        begin=24,
        end=60,
        by_epoch=True,
        convert_to_iter_based=True),
    
    # Momentum schedule
    dict(
        type='CosineAnnealingMomentum',
        T_max=24,
        eta_min=0.85 / 0.95,
        begin=0,
        end=24,
        by_epoch=True,
        convert_to_iter_based=True),
    dict(
        type='CosineAnnealingMomentum',
        T_max=36,
        eta_min=1,
        begin=24,
        end=60,
        by_epoch=True,
        convert_to_iter_based=True)
]

# 修改训练配置为 60 个 epoch
train_cfg = dict(by_epoch=True, max_epochs=60, val_interval=2)
val_cfg = dict()
test_cfg = dict()

auto_scale_lr = dict(enable=False, base_batch_size=48)
