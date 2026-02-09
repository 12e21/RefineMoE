_base_ = ["./SM_branch2.py"]

# Learned router gating on sparsity (points_count).
# Minimal router input: log(points_count + 1).
model = dict(
    roi_head=dict(
        gating_cfg=dict(
            type="LearnedRouterGating",
            num_branch=2,
            hidden_channels=16,
            num_layers=2,
            temperature=1.0,
        ),
    )
)

work_dir = "./work_dirs/RefineMoE/SM_branch2_learnedrouter"
