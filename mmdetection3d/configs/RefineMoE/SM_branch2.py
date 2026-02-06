_base_ = ["./SM.py"]

# Sparsity-MoE expert/branch count (trainable architecture parameter).
model = dict(
    roi_head=dict(
        num_branch=2,
        bbox_head=dict(sparsity_branch_num=2),
    )
)

# Training-only config: disable visualization hook.
default_hooks = dict(visualization=dict(draw=False))

# Use standard KITTI evaluator for AP (bbox/bev/3d).
val_evaluator = dict(
    type="KittiMetric",
    ann_file="data/kitti/kitti_infos_val.pkl",
    metric="bbox",
    backend_args=None,
)
test_evaluator = val_evaluator

# Keep runs separated for traceability.
work_dir = "./work_dirs/RefineMoE/SM_branch2"
