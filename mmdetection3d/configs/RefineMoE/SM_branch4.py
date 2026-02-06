_base_ = ["./SM.py"]

# Sparsity-MoE expert/branch count (trainable architecture parameter).
model = dict(
    roi_head=dict(
        num_branch=4,
        bbox_head=dict(sparsity_branch_num=4),
    )
)

# Training-only config: disable visualization hook.
default_hooks = dict(visualization=dict(draw=False))

# Keep runs separated for traceability.
work_dir = "./work_dirs/RefineMoE/SM_branch4"
