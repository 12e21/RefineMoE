import torch
import torch.nn as nn
from ...ops.pointnet2.pointnet2_stack import voxel_pool_modules as voxelpool_stack_modules
from ...utils import common_utils
from .roi_head_template import RoIHeadTemplate
import torch.nn.functional as F
from ...utils import loss_utils

class SampleDecouplingVoxelRCNNHead(RoIHeadTemplate):
    def __init__(self, backbone_channels, model_cfg, point_cloud_range, voxel_size, num_class=1, **kwargs):
        super().__init__(num_class=num_class, model_cfg=model_cfg)
        self.model_cfg = model_cfg
        self.pool_cfg = model_cfg.ROI_GRID_POOL
        LAYER_cfg = self.pool_cfg.POOL_LAYERS
        self.point_cloud_range = point_cloud_range
        self.voxel_size = voxel_size
        
        # Sample Decoupling parameters
        self.num_experts = model_cfg.get('NUM_EXPERTS', 4)
        self.sparsity_config = model_cfg.get('SPARSITY_CONFIG', {
            'GLOBAL_MIN': 0.0,
            'GLOBAL_MAX': 600.0,
            'SIGMA_SCALE': 0.5
        })

        # Shared ROI pooling layers
        c_out = 0
        self.roi_grid_pool_layers = nn.ModuleList()
        for src_name in self.pool_cfg.FEATURES_SOURCE:
            mlps = LAYER_cfg[src_name].MLPS
            for k in range(len(mlps)):
                mlps[k] = [backbone_channels[src_name]] + mlps[k]
            pool_layer = voxelpool_stack_modules.NeighborVoxelSAModuleMSG(
                query_ranges=LAYER_cfg[src_name].QUERY_RANGES,
                nsamples=LAYER_cfg[src_name].NSAMPLE,
                radii=LAYER_cfg[src_name].POOL_RADIUS,
                mlps=mlps,
                pool_method=LAYER_cfg[src_name].POOL_METHOD,
            )
            self.roi_grid_pool_layers.append(pool_layer)
            c_out += sum([x[-1] for x in mlps])

        GRID_SIZE = self.model_cfg.ROI_GRID_POOL.GRID_SIZE
        pre_channel = GRID_SIZE * GRID_SIZE * GRID_SIZE * c_out

        # Create multiple expert branches
        self.expert_branches = nn.ModuleList()
        for expert_idx in range(self.num_experts):
            expert_branch = self._create_expert_branch(pre_channel)
            self.expert_branches.append(expert_branch)

        self.init_weights()

    def _create_expert_branch(self, pre_channel):
        """Create a single expert branch with shared_fc, cls_fc, and reg_fc"""
        branch = nn.ModuleDict()
        
        # Shared FC layers
        shared_fc_list = []
        current_channel = pre_channel
        for k in range(0, self.model_cfg.SHARED_FC.__len__()):
            shared_fc_list.extend([
                nn.Linear(current_channel, self.model_cfg.SHARED_FC[k], bias=False),
                nn.BatchNorm1d(self.model_cfg.SHARED_FC[k]),
                nn.ReLU(inplace=True)
            ])
            current_channel = self.model_cfg.SHARED_FC[k]

            if k != self.model_cfg.SHARED_FC.__len__() - 1 and self.model_cfg.DP_RATIO > 0:
                shared_fc_list.append(nn.Dropout(self.model_cfg.DP_RATIO))
        branch['shared_fc'] = nn.Sequential(*shared_fc_list)

        # Classification FC layers
        cls_fc_list = []
        cls_channel = current_channel
        for k in range(0, self.model_cfg.CLS_FC.__len__()):
            cls_fc_list.extend([
                nn.Linear(cls_channel, self.model_cfg.CLS_FC[k], bias=False),
                nn.BatchNorm1d(self.model_cfg.CLS_FC[k]),
                nn.ReLU()
            ])
            cls_channel = self.model_cfg.CLS_FC[k]

            if k != self.model_cfg.CLS_FC.__len__() - 1 and self.model_cfg.DP_RATIO > 0:
                cls_fc_list.append(nn.Dropout(self.model_cfg.DP_RATIO))
        branch['cls_fc'] = nn.Sequential(*cls_fc_list)
        branch['cls_pred'] = nn.Linear(cls_channel, self.num_class, bias=True)

        # Regression FC layers
        reg_fc_list = []
        reg_channel = current_channel
        for k in range(0, self.model_cfg.REG_FC.__len__()):
            reg_fc_list.extend([
                nn.Linear(reg_channel, self.model_cfg.REG_FC[k], bias=False),
                nn.BatchNorm1d(self.model_cfg.REG_FC[k]),
                nn.ReLU()
            ])
            reg_channel = self.model_cfg.REG_FC[k]

            if k != self.model_cfg.REG_FC.__len__() - 1 and self.model_cfg.DP_RATIO > 0:
                reg_fc_list.append(nn.Dropout(self.model_cfg.DP_RATIO))
        branch['reg_fc'] = nn.Sequential(*reg_fc_list)
        branch['reg_pred'] = nn.Linear(reg_channel, self.box_coder.code_size * self.num_class, bias=True)

        return branch

    def init_weights(self):
        init_func = nn.init.xavier_normal_
        for expert_branch in self.expert_branches:
            for module_list in [expert_branch['shared_fc'], expert_branch['cls_fc'], expert_branch['reg_fc']]:
                for m in module_list.modules():
                    if isinstance(m, nn.Linear):
                        init_func(m.weight)
                        if m.bias is not None:
                            nn.init.constant_(m.bias, 0)
                        
            nn.init.normal_(expert_branch['cls_pred'].weight, 0, 0.01)
            nn.init.constant_(expert_branch['cls_pred'].bias, 0)
            nn.init.normal_(expert_branch['reg_pred'].weight, mean=0, std=0.001)
            nn.init.constant_(expert_branch['reg_pred'].bias, 0)

    def gaussian_fixed_interval_scoring(self,
                                        x: torch.Tensor, 
                                        K: int, 
                                        global_min: float = 0.0, 
                                        global_max: float = 600.0, 
                                        sigma_scale: float = 0.5) -> torch.Tensor:
        """
        Generate a K-dimensional score vector for each scalar input using fixed interval Gaussian scoring.

        The value range [global_min, global_max] is divided evenly into (K - 1) intervals.
        Each of the first K - 1 groups focuses on a specific subrange, with samples in the interval scored as 1,
        and others decaying based on a Gaussian kernel. The K-th group focuses on values greater than global_max.

        Args:
            x (torch.Tensor): A 1D tensor of shape (N,) containing input scalar values.
            K (int): The number of scoring groups. The last group targets (global_max, +inf).
            global_min (float): The lower bound of the fixed scoring range.
            global_max (float): The upper bound of the fixed scoring range.
            sigma_scale (float): Scaling factor for Gaussian kernel width (sigma = sigma_scale * interval width).

        Returns:
            torch.Tensor: A (K, N) tensor where each row contains the scores for one group.
        """
        x = x.view(1, -1)  # shape: (1, N)
        N = x.shape[1]
        device = x.device

        width = (global_max - global_min) / (K - 1)
        sigma = width * sigma_scale

        centers = torch.linspace(global_min + width / 2, global_max - width / 2, K - 1, device=device)
        centers = torch.cat([centers, torch.tensor([global_max + width / 2], device=device)])  # shape: (K,)

        a = torch.linspace(global_min, global_max - width, K - 1, device=device)
        b = a + width
        a = torch.cat([a, torch.tensor([global_max], device=device)])
        b = torch.cat([b, torch.tensor([float('inf')], device=device)])  # shape: (K,)

        x_expanded = x.expand(K, -1)  # shape: (K, N)
        centers_expanded = centers.view(-1, 1)  # shape: (K, 1)

        # Gaussian kernel scoring
        scores = torch.exp(-((x_expanded - centers_expanded) ** 2) / (2 * sigma ** 2))

        # Assign score 1.0 to samples within the focused interval
        mask_in_range = (x_expanded >= a.view(-1, 1)) & (x_expanded <= b.view(-1, 1))
        scores[mask_in_range] = 1.0

        return scores  # shape: (K, N)

    def box_query_point_count_parallel(self, batch_dict):
        rois = batch_dict['rois']
        points = batch_dict['points']

        batch_size, num_rois, _ = rois.shape
        num_points = points.shape[0]

        expanded_rois = rois.unsqueeze(2)
        
        batch_indices = points[:, 0].long()
        point_coords = points[:, 1:4]
        
        max_points_per_batch = torch.bincount(batch_indices).max()
        
        padded_points = torch.zeros(batch_size, max_points_per_batch, 3, device=rois.device)
        points_mask = torch.zeros(batch_size, max_points_per_batch, dtype=torch.bool, device=rois.device)

        for b_idx in range(batch_size):
            batch_mask = (batch_indices == b_idx)
            batch_points = point_coords[batch_mask]
            num_batch_points = batch_points.shape[0]
            if num_batch_points > 0:
                padded_points[b_idx, :num_batch_points] = batch_points
                points_mask[b_idx, :num_batch_points] = True
                
        expanded_points = padded_points.unsqueeze(1)
        
        roi_centers = expanded_rois[..., :3]
        roi_sizes = expanded_rois[..., 3:6]
        roi_angles = expanded_rois[..., 6]

        translated_points = expanded_points - roi_centers

        cos_angles = torch.cos(-roi_angles)
        sin_angles = torch.sin(-roi_angles)

        x_coords = translated_points[..., 0]
        y_coords = translated_points[..., 1]
        
        rotated_x = x_coords * cos_angles - y_coords * sin_angles
        rotated_y = x_coords * sin_angles + y_coords * cos_angles
        rotated_z = translated_points[..., 2]

        half_sizes = roi_sizes / 2.0
        
        inside_mask = (torch.abs(rotated_x) <= half_sizes[..., 0]) & \
                    (torch.abs(rotated_y) <= half_sizes[..., 1]) & \
                    (torch.abs(rotated_z) <= half_sizes[..., 2])

        valid_inside_mask = inside_mask & points_mask.unsqueeze(1)
        
        point_counts = valid_inside_mask.sum(dim=-1).float()
        
        return point_counts.flatten()
    def box_query_point_count(self, batch_dict):
        """
        Compute the number of points in each proposal box.
        
        Args:
            batch_dict: dict containing rois and point_coords
            
        Returns:
            point_counts: tensor of shape (total_rois,) containing point counts
        """
        rois = batch_dict['rois']  # (B, N, 7)
        point_coords = batch_dict['points'][:,:4]  # (num_points, 4) [bs_idx, x, y, z]
        
        batch_size = rois.shape[0]
        num_rois = rois.shape[1]
        
        point_counts = []
        
        for bs_idx in range(batch_size):
            # Get points for current batch
            batch_mask = point_coords[:, 0] == bs_idx
            batch_points = point_coords[batch_mask, 1:4]  # (N_points, 3)
            
            if batch_points.shape[0] == 0:
                # No points in this batch
                point_counts.append(torch.zeros(num_rois, device=rois.device))
                continue
                
            batch_rois = rois[bs_idx]  # (N, 7)
            batch_counts = torch.zeros(num_rois, device=rois.device)
            
            for roi_idx in range(num_rois):
                roi = batch_rois[roi_idx]  # (7,) [x, y, z, dx, dy, dz, ry]
                
                # Transform points to box coordinate system
                roi_center = roi[:3]
                roi_size = roi[3:6]
                roi_ry = roi[6]
                
                # Translate points to box center
                translated_points = batch_points - roi_center.unsqueeze(0)
                
                # Rotate points (inverse rotation)
                cos_ry = torch.cos(-roi_ry)
                sin_ry = torch.sin(-roi_ry)
                
                rotated_x = translated_points[:, 0] * cos_ry - translated_points[:, 1] * sin_ry
                rotated_y = translated_points[:, 0] * sin_ry + translated_points[:, 1] * cos_ry
                rotated_z = translated_points[:, 2]
                
                # Check if points are inside box
                half_size = roi_size / 2
                inside_mask = (torch.abs(rotated_x) <= half_size[0]) & \
                             (torch.abs(rotated_y) <= half_size[1]) & \
                             (torch.abs(rotated_z) <= half_size[2])
                
                batch_counts[roi_idx] = inside_mask.sum().float()
            
            point_counts.append(batch_counts)
        
        # Flatten to (total_rois,)
        point_counts = torch.cat(point_counts, dim=0)
        return point_counts

    def roi_grid_pool(self, batch_dict):
        """
        Shared ROI pooling (same as original implementation)
        """
        rois = batch_dict['rois']
        batch_size = batch_dict['batch_size']
        with_vf_transform = batch_dict.get('with_voxel_feature_transform', False)
        
        roi_grid_xyz, _ = self.get_global_grid_points_of_roi(
            rois, grid_size=self.pool_cfg.GRID_SIZE
        )  # (BxN, 6x6x6, 3)
        # roi_grid_xyz: (B, Nx6x6x6, 3)
        roi_grid_xyz = roi_grid_xyz.view(batch_size, -1, 3)  

        # compute the voxel coordinates of grid points
        roi_grid_coords_x = (roi_grid_xyz[:, :, 0:1] - self.point_cloud_range[0]) // self.voxel_size[0]
        roi_grid_coords_y = (roi_grid_xyz[:, :, 1:2] - self.point_cloud_range[1]) // self.voxel_size[1]
        roi_grid_coords_z = (roi_grid_xyz[:, :, 2:3] - self.point_cloud_range[2]) // self.voxel_size[2]
        # roi_grid_coords: (B, Nx6x6x6, 3)
        roi_grid_coords = torch.cat([roi_grid_coords_x, roi_grid_coords_y, roi_grid_coords_z], dim=-1)

        batch_idx = rois.new_zeros(batch_size, roi_grid_coords.shape[1], 1)
        for bs_idx in range(batch_size):
            batch_idx[bs_idx, :, 0] = bs_idx
        roi_grid_batch_cnt = rois.new_zeros(batch_size).int().fill_(roi_grid_coords.shape[1])

        pooled_features_list = []
        for k, src_name in enumerate(self.pool_cfg.FEATURES_SOURCE):
            pool_layer = self.roi_grid_pool_layers[k]
            cur_stride = batch_dict['multi_scale_3d_strides'][src_name]
            cur_sp_tensors = batch_dict['multi_scale_3d_features'][src_name]

            if with_vf_transform:
                cur_sp_tensors = batch_dict['multi_scale_3d_features_post'][src_name]
            else:
                cur_sp_tensors = batch_dict['multi_scale_3d_features'][src_name]

            # compute voxel center xyz and batch_cnt
            cur_coords = cur_sp_tensors.indices
            cur_voxel_xyz = common_utils.get_voxel_centers(
                cur_coords[:, 1:4],
                downsample_times=cur_stride,
                voxel_size=self.voxel_size,
                point_cloud_range=self.point_cloud_range
            )
            cur_voxel_xyz_batch_cnt = cur_voxel_xyz.new_zeros(batch_size).int()
            for bs_idx in range(batch_size):
                cur_voxel_xyz_batch_cnt[bs_idx] = (cur_coords[:, 0] == bs_idx).sum()
            # get voxel2point tensor
            v2p_ind_tensor = common_utils.generate_voxel2pinds(cur_sp_tensors)
            # compute the grid coordinates in this scale, in [batch_idx, x y z] order
            cur_roi_grid_coords = roi_grid_coords // cur_stride
            cur_roi_grid_coords = torch.cat([batch_idx, cur_roi_grid_coords], dim=-1)
            cur_roi_grid_coords = cur_roi_grid_coords.int()
            # voxel neighbor aggregation
            pooled_features = pool_layer(
                xyz=cur_voxel_xyz.contiguous(),
                xyz_batch_cnt=cur_voxel_xyz_batch_cnt,
                new_xyz=roi_grid_xyz.contiguous().view(-1, 3),
                new_xyz_batch_cnt=roi_grid_batch_cnt,
                new_coords=cur_roi_grid_coords.contiguous().view(-1, 4),
                features=cur_sp_tensors.features.contiguous(),
                voxel2point_indices=v2p_ind_tensor
            )

            pooled_features = pooled_features.view(
                -1, self.pool_cfg.GRID_SIZE ** 3,
                pooled_features.shape[-1]
            )  # (BxN, 6x6x6, C)
            pooled_features_list.append(pooled_features)
        
        ms_pooled_features = torch.cat(pooled_features_list, dim=-1)
        return ms_pooled_features

    def get_global_grid_points_of_roi(self, rois, grid_size):
        rois = rois.view(-1, rois.shape[-1])
        batch_size_rcnn = rois.shape[0]

        local_roi_grid_points = self.get_dense_grid_points(rois, batch_size_rcnn, grid_size)  # (B, 6x6x6, 3)
        global_roi_grid_points = common_utils.rotate_points_along_z(
            local_roi_grid_points.clone(), rois[:, 6]
        ).squeeze(dim=1)
        global_center = rois[:, 0:3].clone()
        global_roi_grid_points += global_center.unsqueeze(dim=1)
        return global_roi_grid_points, local_roi_grid_points

    @staticmethod
    def get_dense_grid_points(rois, batch_size_rcnn, grid_size):
        faked_features = rois.new_ones((grid_size, grid_size, grid_size))
        dense_idx = faked_features.nonzero()  # (N, 3) [x_idx, y_idx, z_idx]
        dense_idx = dense_idx.repeat(batch_size_rcnn, 1, 1).float()  # (B, 6x6x6, 3)

        local_roi_size = rois.view(batch_size_rcnn, -1)[:, 3:6]
        roi_grid_points = (dense_idx + 0.5) / grid_size * local_roi_size.unsqueeze(dim=1) \
                          - (local_roi_size.unsqueeze(dim=1) / 2)  # (B, 6x6x6, 3)
        return roi_grid_points

    def forward(self, batch_dict):
        """
        Forward pass with sample decoupling logic
        """
        targets_dict = self.proposal_layer(
            batch_dict, nms_config=self.model_cfg.NMS_CONFIG['TRAIN' if self.training else 'TEST']
        )
        if self.training:
            targets_dict = self.assign_targets(batch_dict)
            batch_dict['rois'] = targets_dict['rois']
            batch_dict['roi_labels'] = targets_dict['roi_labels']

        # Step 1: Compute point counts for each proposal
        point_counts = self.box_query_point_count_parallel(batch_dict)  # (total_rois,)

        # Step 2: Generate sparsity scores using Gaussian fixed interval scoring
        sparsity_scores = self.gaussian_fixed_interval_scoring(
            point_counts, 
            self.num_experts, 
            self.sparsity_config['GLOBAL_MIN'],
            self.sparsity_config['GLOBAL_MAX'],
            self.sparsity_config['SIGMA_SCALE']
        )  # (K, total_rois)
        
        # Step 3: Shared ROI pooling
        pooled_features = self.roi_grid_pool(batch_dict)  # (BxN, 6x6x6, C)
        pooled_features = pooled_features.view(pooled_features.size(0), -1)  # (total_rois, flattened_features)

        if self.training:
            # Training phase: Use weighted loss for each expert
            expert_cls_preds = []
            expert_reg_preds = []
            
            for expert_idx in range(self.num_experts):
                expert_branch = self.expert_branches[expert_idx]
                
                # Forward through expert branch
                shared_features = expert_branch['shared_fc'](pooled_features)
                cls_pred = expert_branch['cls_pred'](expert_branch['cls_fc'](shared_features))
                reg_pred = expert_branch['reg_pred'](expert_branch['reg_fc'](shared_features))
                
                expert_cls_preds.append(cls_pred)
                expert_reg_preds.append(reg_pred)

            # Store predictions and sparsity scores for loss computation
            targets_dict['expert_cls_preds'] = expert_cls_preds
            targets_dict['expert_reg_preds'] = expert_reg_preds
            targets_dict['sparsity_scores'] = sparsity_scores
            self.forward_ret_dict = targets_dict

        else:
            # Test phase: Compute weighted combination
            expert_cls_preds = []
            expert_reg_preds = []
            
            for expert_idx in range(self.num_experts):
                expert_branch = self.expert_branches[expert_idx]
                
                # Forward through expert branch
                shared_features = expert_branch['shared_fc'](pooled_features)
                cls_pred = expert_branch['cls_pred'](expert_branch['cls_fc'](shared_features))
                reg_pred = expert_branch['reg_pred'](expert_branch['reg_fc'](shared_features))
                
                expert_cls_preds.append(cls_pred)
                expert_reg_preds.append(reg_pred)

            # Stack predictions
            stacked_cls_preds = torch.stack(expert_cls_preds, dim=0)  # (K, total_rois, num_class)
            stacked_reg_preds = torch.stack(expert_reg_preds, dim=0)  # (K, total_rois, code_size * num_class)
            
            # Normalize sparsity scores across experts
            branch_sum_scores = torch.sum(sparsity_scores, dim=0, keepdim=True)  # (1, total_rois)
            fuse_weights = sparsity_scores / (branch_sum_scores + 1e-8)  # (K, total_rois)
            
            # Weighted combination
            def weighted_combination(weights, tensor):
                weights_expanded = weights.unsqueeze(-1)  # (K, total_rois, 1)
                weighted_tensor = tensor * weights_expanded  # (K, total_rois, C)
                result = torch.sum(weighted_tensor, dim=0)  # (total_rois, C)
                return result

            final_cls_pred = weighted_combination(fuse_weights, stacked_cls_preds)
            final_reg_pred = weighted_combination(fuse_weights, stacked_reg_preds)

            mode = 'refine_reg'
            assert mode in ['normal', 'skip_roi', 'refine_reg', 'refine_cls']
            if mode == 'normal':
                batch_cls_preds, batch_box_preds = self.generate_predicted_boxes(
                    batch_size=batch_dict['batch_size'], rois=batch_dict['rois'], cls_preds=final_cls_pred, box_preds=final_reg_pred
                )
                batch_dict['batch_cls_preds'] = batch_cls_preds
                batch_dict['batch_box_preds'] = batch_box_preds
                batch_dict['cls_preds_normalized'] = False
            elif mode == "refine_reg":
                batch_cls_preds, batch_box_preds = self.generate_predicted_boxes(
                    batch_size=batch_dict['batch_size'], rois=batch_dict['rois'], cls_preds=final_cls_pred, box_preds=final_reg_pred
                )
                batch_dict['batch_cls_preds'] = batch_dict['roi_scores'].unsqueeze(-1)
                batch_dict['batch_box_preds'] = batch_box_preds
                batch_dict['cls_preds_normalized'] = True
            
            elif mode == "refine_cls":
                batch_cls_preds, batch_box_preds = self.generate_predicted_boxes(
                    batch_size=batch_dict['batch_size'], rois=batch_dict['rois'], cls_preds=final_cls_pred, box_preds=final_reg_pred
                )
                batch_dict['batch_cls_preds'] = batch_cls_preds
                batch_dict['batch_box_preds'] = batch_dict['rois']
                batch_dict['cls_preds_normalized'] = False

            elif mode == "skip_roi":
                batch_dict['batch_cls_preds'] = batch_dict['roi_scores'].unsqueeze(-1)
                batch_dict['batch_box_preds'] = batch_dict['rois']
                batch_dict['cls_preds_normalized'] = True            
            
            # # Generate final predictions
            # batch_cls_preds, batch_box_preds = self.generate_predicted_boxes(
            #     batch_size=batch_dict['batch_size'], 
            #     rois=batch_dict['rois'], 
            #     cls_preds=final_cls_pred, 
            #     box_preds=final_reg_pred
            # )
            # batch_dict['batch_cls_preds'] = batch_cls_preds
            # batch_dict['batch_box_preds'] = batch_box_preds
            # batch_dict['cls_preds_normalized'] = False

        return batch_dict

    def get_box_cls_layer_loss(self, forward_ret_dict):
        """Compute classification loss for all experts with sparsity weighting"""
        loss_cfgs = self.model_cfg.LOSS_CONFIG
        expert_cls_preds = forward_ret_dict['expert_cls_preds']
        sparsity_scores = forward_ret_dict['sparsity_scores']  # (K, total_rois)
        rcnn_cls_labels = forward_ret_dict['rcnn_cls_labels'].view(-1)
        
        total_cls_loss = 0
        tb_dict = {}
        
        for expert_idx in range(self.num_experts):
            rcnn_cls = expert_cls_preds[expert_idx]
            expert_sparsity_weight = sparsity_scores[expert_idx]  # (total_rois,)
            
            if loss_cfgs.CLS_LOSS == 'BinaryCrossEntropy':
                rcnn_cls_flat = rcnn_cls.view(-1)
                batch_loss_cls = F.binary_cross_entropy(
                    torch.sigmoid(rcnn_cls_flat), rcnn_cls_labels.float(), reduction='none'
                )
            elif loss_cfgs.CLS_LOSS == 'CrossEntropy':
                batch_loss_cls = F.cross_entropy(
                    rcnn_cls, rcnn_cls_labels, reduction='none', ignore_index=-1
                )
            else:
                raise NotImplementedError
            
            # Apply sparsity weighting
            cls_valid_mask = (rcnn_cls_labels >= 0).float()
            weighted_loss = batch_loss_cls * expert_sparsity_weight * cls_valid_mask
            expert_cls_loss = weighted_loss.sum() / torch.clamp(
                (expert_sparsity_weight * cls_valid_mask).sum(), min=1.0
            )
            
            expert_cls_loss = expert_cls_loss * loss_cfgs.LOSS_WEIGHTS['rcnn_cls_weight']
            total_cls_loss += expert_cls_loss
            tb_dict[f'expert_{expert_idx}_cls_loss'] = expert_cls_loss.item()
        
        tb_dict['rcnn_loss_cls'] = total_cls_loss.item()
        return total_cls_loss, tb_dict

    def get_box_reg_layer_loss(self, forward_ret_dict):
        """Compute regression loss for all experts with sparsity weighting"""
        loss_cfgs = self.model_cfg.LOSS_CONFIG
        expert_reg_preds = forward_ret_dict['expert_reg_preds']
        sparsity_scores = forward_ret_dict['sparsity_scores']  # (K, total_rois)
        
        code_size = self.box_coder.code_size
        reg_valid_mask = forward_ret_dict['reg_valid_mask'].view(-1)
        gt_boxes3d_ct = forward_ret_dict['gt_of_rois'][..., 0:code_size]
        gt_of_rois_src = forward_ret_dict['gt_of_rois_src'][..., 0:code_size].view(-1, code_size)
        roi_boxes3d = forward_ret_dict['rois']
        rcnn_batch_size = gt_boxes3d_ct.view(-1, code_size).shape[0]

        fg_mask = (reg_valid_mask > 0)
        fg_sum = fg_mask.long().sum().item()

        total_reg_loss = 0
        tb_dict = {}

        if loss_cfgs.REG_LOSS == 'smooth-l1':
            rois_anchor = roi_boxes3d.clone().detach().view(-1, code_size)
            rois_anchor[:, 0:3] = 0
            rois_anchor[:, 6] = 0
            reg_targets = self.box_coder.encode_torch(
                gt_boxes3d_ct.view(rcnn_batch_size, code_size), rois_anchor
            )

            for expert_idx in range(self.num_experts):
                rcnn_reg = expert_reg_preds[expert_idx]
                expert_sparsity_weight = sparsity_scores[expert_idx]  # (total_rois,)

                rcnn_loss_reg = self.reg_loss_func(
                    rcnn_reg.view(rcnn_batch_size, -1).unsqueeze(dim=0),
                    reg_targets.unsqueeze(dim=0),
                )  # [B, M, 7]
                
                # Apply sparsity weighting
                weighted_reg_loss = rcnn_loss_reg.view(rcnn_batch_size, -1) * expert_sparsity_weight.unsqueeze(dim=-1) * fg_mask.unsqueeze(dim=-1).float()
                expert_reg_loss = weighted_reg_loss.sum() / max(
                    (expert_sparsity_weight * fg_mask.float()).sum().item(), 1
                )
                expert_reg_loss = expert_reg_loss * loss_cfgs.LOSS_WEIGHTS['rcnn_reg_weight']
                
                total_reg_loss += expert_reg_loss
                tb_dict[f'expert_{expert_idx}_reg_loss'] = expert_reg_loss.item()

                # Corner loss (if enabled)
                if loss_cfgs.CORNER_LOSS_REGULARIZATION and fg_sum > 0:
                    fg_rcnn_reg = rcnn_reg.view(rcnn_batch_size, -1)[fg_mask]
                    fg_roi_boxes3d = roi_boxes3d.view(-1, code_size)[fg_mask]
                    fg_sparsity_weight = expert_sparsity_weight[fg_mask]

                    fg_roi_boxes3d = fg_roi_boxes3d.view(1, -1, code_size)
                    batch_anchors = fg_roi_boxes3d.clone().detach()
                    roi_ry = fg_roi_boxes3d[:, :, 6].view(-1)
                    roi_xyz = fg_roi_boxes3d[:, :, 0:3].view(-1, 3)
                    batch_anchors[:, :, 0:3] = 0
                    rcnn_boxes3d = self.box_coder.decode_torch(
                        fg_rcnn_reg.view(batch_anchors.shape[0], -1, code_size), batch_anchors
                    ).view(-1, code_size)

                    rcnn_boxes3d = common_utils.rotate_points_along_z(
                        rcnn_boxes3d.unsqueeze(dim=1), roi_ry
                    ).squeeze(dim=1)
                    rcnn_boxes3d[:, 0:3] += roi_xyz

                    # Get ground truth boxes
                    fg_gt_boxes3d = gt_of_rois_src[fg_mask]

                    # Compute corner loss with sparsity weighting
                    corner_loss = loss_utils.get_corner_loss_lidar(
                        rcnn_boxes3d[:, 0:7],
                        fg_gt_boxes3d[:, 0:7]
                    )
                    corner_loss = corner_loss.mean()
                    weighted_corner_loss = corner_loss * fg_sparsity_weight.mean()
                    weighted_corner_loss = weighted_corner_loss * loss_cfgs.LOSS_WEIGHTS['rcnn_corner_weight']
                    
                    total_reg_loss += weighted_corner_loss
                    tb_dict[f'expert_{expert_idx}_corner_loss'] = weighted_corner_loss.item()

        else:
            raise NotImplementedError(f"Regression loss {loss_cfgs.REG_LOSS} not implemented")

        tb_dict['rcnn_loss_reg'] = total_reg_loss.item()
        return total_reg_loss, tb_dict