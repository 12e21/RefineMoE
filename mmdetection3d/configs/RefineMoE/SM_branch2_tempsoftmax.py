_base_ = ["./SM_branch2.py"]

# Temperature-softmax gating on sparsity (points_count).
# NOTE: branch count is fixed to 2 for this ablation.
model = dict(
    roi_head=dict(
        gating_cfg=dict(
            type="TemperatureSoftmaxGating",
            num_branch=2,
            global_min=0.0,
            global_max=10.0,
            sigma_scale=1.0,
            temperature=0.5,
        ),
    )
)

work_dir = "./work_dirs/RefineMoE/SM_branch2_tempsoftmax_tau0p5"
