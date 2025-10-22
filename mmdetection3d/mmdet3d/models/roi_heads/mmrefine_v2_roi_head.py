# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Optional, Dict, Tuple
from torch.nn import ModuleList
import torch
from mmdet.models.task_modules import AssignResult
from mmdet.models.task_modules.samplers import SamplingResult
from torch.nn import functional as F

from mmdet3d.models.roi_heads.base_3droi_head import Base3DRoIHead
from mmdet3d.registry import MODELS, TASK_UTILS
from mmdet3d.structures import bbox3d2roi
from mmdet3d.structures.det3d_data_sample import SampleList
from mmdet3d.utils import InstanceList
from mmdet3d.structures.bbox_3d import rotation_3d_in_axis, LiDARInstance3DBoxes
from mmcv.ops.points_in_boxes import points_in_boxes_all
import numpy as np


@MODELS.register_module()
class MMrefineV2RoiHead(Base3DRoIHead):
    """Multi-stage Multi-branch RoI head combining DecoupleRefineRoiHead and SoftSparsityBranchRoiHead.
    
    Each stage performs multi-branch computation with sparsity-aware weighting.

    Args:
        num_stage (int): Number of refinement stages. Defaults to 3.
        stage_loss_weights (list): Loss weights for each stage. Defaults to [1.0, 1.0, 1.0].
        num_classes (int): The number of classes. Defaults to 3.
        semantic_head (dict, optional): Config of semantic head. Defaults to None.
        bbox_roi_extractor (dict, optional): Config of roi_extractor. Defaults to None.
        bbox_head (dict, optional): Config of bbox_head. Defaults to None.
        bbox_coder (dict, optional): Config of bbox_coder. Defaults to None.
        train_cfg (dict, optional): Train config of model. Defaults to None.
        test_cfg (dict, optional): Test config of model. Defaults to None.
        init_cfg (dict, optional): Initialize config of model. Defaults to None.
        sparsity_branch_num (int): Number of sparsity branches. Defaults to 2.
        sparsity_global_max (float): Global max for sparsity scoring. Defaults to 10.0.
        sparsity_sigma_scale (float): Sigma scale for sparsity scoring. Defaults to 1.0.
    """

    def __init__(self,
                 num_stage: int = 3,
                 stage_loss_weights: list = [1.0, 1.0, 1.0],
                 num_classes: int = 3,
                 semantic_head: Optional[dict] = None,
                 bbox_roi_extractor: Optional[dict] = None,
                 bbox_head: Optional[dict] = None,
                 bbox_coder: Optional[dict] = None,
                 train_cfg: Optional[dict] = None,
                 test_cfg: Optional[dict] = None,
                 init_cfg: Optional[dict] = None,
                 sparsity_branch_num: int = 2,
                 sparsity_global_max: float = 10.0,
                 sparsity_sigma_scale: float = 1.0):
        
        self.num_stage = num_stage
        self.stage_loss_weights = stage_loss_weights
        self.sparsity_branch_num = sparsity_branch_num
        self.sparsity_global_max = sparsity_global_max
        self.sparsity_sigma_scale = sparsity_sigma_scale
        
        super(MMrefineV2RoiHead, self).__init__(
            bbox_head=bbox_head,
            bbox_roi_extractor=bbox_roi_extractor,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            init_cfg=init_cfg)
        
        self.num_classes = num_classes
        self.semantic_head = MODELS.build(semantic_head)
        self.bbox_coder = TASK_UTILS.build(bbox_coder)
        self.init_assigner_sampler()

    def init_bbox_head(self, bbox_roi_extractor: dict, bbox_head: dict) -> None:
        """Initialize box head and box roi extractor for multi-stage setup.

        Args:
            bbox_roi_extractor (dict or ConfigDict): Config of box roi extractor.
            bbox_head (dict or ConfigDict): Config of box in box head.
        """
        self.bbox_roi_extractor = ModuleList()
        self.bbox_head = ModuleList()
        
        if not isinstance(bbox_roi_extractor, list):
            bbox_roi_extractor = [bbox_roi_extractor for _ in range(self.num_stage)]
        if not isinstance(bbox_head, list):
            bbox_head = [bbox_head for _ in range(self.num_stage)]
            
        assert len(bbox_roi_extractor) == len(bbox_head) == self.num_stage
        
        for roi_extractor, head in zip(bbox_roi_extractor, bbox_head):
            self.bbox_head.append(MODELS.build(head))
            self.bbox_roi_extractor.append(MODELS.build(roi_extractor))

    @property
    def with_semantic(self):
        """bool: whether the head has semantic branch"""
        return hasattr(self, 'semantic_head') and self.semantic_head is not None

    def loss(self, feats_dict: dict, rpn_results_list: InstanceList,
             batch_data_samples: SampleList, **kwargs) -> dict:
        """Training forward function of MMrefineV2RoiHead.

        Args:
            feats_dict (dict): Contains point-wise features.
            rpn_results_list (List[:obj:`InstanceData`]): Detection results of rpn head.
            batch_data_samples (List[:obj:`Det3DDataSample`]): The Data samples.

        Returns:
            dict: losses from each head including multi-stage multi-branch losses.
        """
        losses = dict()
        batch_gt_instances_3d = []
        batch_gt_instances_ignore = []
        
        for data_sample in batch_data_samples:
            batch_gt_instances_3d.append(data_sample.gt_instances_3d)
            if 'ignored_instances' in data_sample:
                batch_gt_instances_ignore.append(data_sample.ignored_instances)
            else:
                batch_gt_instances_ignore.append(None)

        # Semantic head forward
        if self.with_semantic:
            semantic_results = self._semantic_forward_train(
                feats_dict['keypoint_features'], feats_dict['keypoints'],
                batch_gt_instances_3d)
            losses['loss_semantic'] = semantic_results['loss_semantic']

        # Sample results
        sample_results = self._assign_and_sample(rpn_results_list, batch_gt_instances_3d)
        
        # Generate sparsity scores
        source_points_list = feats_dict['points']
        source_points_list = [points[:, :3] for points in source_points_list]
        bbox_list = [res.bboxes for res in sample_results]
        points_count_list = self.count_points_in_bbox(source_points_list, bbox_list)
        sparsity_scores_list = self.sparsity_scoring(
            points_count_list=points_count_list,
            branch_num=self.sparsity_branch_num,
            global_max=self.sparsity_global_max,
            sigma_scale=self.sparsity_sigma_scale)
        stacked_sparsity_scores = torch.stack(sparsity_scores_list).reshape(-1, sparsity_scores_list[0].shape[-1])

        # Multi-stage multi-branch bbox forward
        if self.with_bbox:
            bbox_results_list, loss_merge_corner = self._bbox_forward_train(
                semantic_results['seg_preds'],
                feats_dict['fusion_keypoint_features'],
                feats_dict['keypoints'], 
                sample_results,
                stacked_sparsity_scores)

            # Aggregate stage losses
            for stage in range(self.num_stage):
                stage_loss_weight = self.stage_loss_weights[stage]
                stage_losses = bbox_results_list[stage]['loss_bbox']
                for name, value in stage_losses.items():
                    losses[f's{stage}.{name}'] = (
                        value * stage_loss_weight if 'loss' in name else value)

            losses.update(loss_merge_corner)

        return losses

    def predict(self, feats_dict: dict, rpn_results_list: InstanceList,
                batch_data_samples: SampleList, **kwargs) -> SampleList:
        """Perform forward propagation and predict detection results.

        Args:
            feats_dict (dict): Contains point-wise features.
            rpn_results_list (List[:obj:`InstanceData`]): Detection results of rpn head.
            batch_data_samples (List[:obj:`Det3DDataSample`]): The Data samples.

        Returns:
            list[:obj:`InstanceData`]: Detection results of each sample.
        """
        assert self.with_bbox, 'Bbox head must be implemented.'
        assert self.with_semantic, 'Semantic head must be implemented.'

        batch_input_metas = [
            data_samples.metainfo for data_samples in batch_data_samples
        ]

        # Semantic forward
        semantic_results = self.semantic_head(feats_dict['keypoint_features'])
        point_features = feats_dict[
            'fusion_keypoint_features'] * semantic_results[
                'seg_preds'].sigmoid().max(
                    dim=-1, keepdim=True).values

        # Generate sparsity scores
        source_points_list = feats_dict['points']
        source_points_list = [points[:, :3] for points in source_points_list]
        bbox_list = [res.bboxes_3d.tensor for res in rpn_results_list]
        points_count_list = self.count_points_in_bbox(source_points_list, bbox_list)
        sparsity_scores_list = self.sparsity_scoring(
            points_count_list=points_count_list,
            branch_num=self.sparsity_branch_num,
            global_max=self.sparsity_global_max,
            sigma_scale=self.sparsity_sigma_scale)
        stacked_sparsity_scores = torch.stack(sparsity_scores_list).reshape(-1, sparsity_scores_list[0].shape[-1])

        rois = bbox3d2roi([res['bboxes_3d'].tensor for res in rpn_results_list])
        labels_3d = [res['labels_3d'] for res in rpn_results_list]

        # Multi-stage multi-branch bbox forward
        bbox_results_list = []
        for stage in range(self.num_stage):
            bbox_results = self._bbox_forward(stage, point_features, feats_dict['keypoints'], rois)
            bbox_results_list.append(bbox_results)

        # Merge multi-stage predictions (assume 3 stages: center, size, direction)
        assert self.num_stage == 3
        merge_bbox_results = bbox_results_list[0]['bbox_reg']
        for i in range(len(merge_bbox_results)):
            merge_bbox_results[i][:, 3:6] = bbox_results_list[1]['bbox_reg'][i][:, 3:6]
            merge_bbox_results[i][:, 6] = bbox_results_list[2]['bbox_reg'][i][:, 6]

        # Multi-branch fusion for final scores
        def weighted_combination(weights, tensor_list):
            weights_expanded = weights.unsqueeze(-1)  # [M, N, 1]
            stacked_tensors = torch.stack(tensor_list, dim=0)  # [M, N, C]
            weighted_tensor = stacked_tensors * weights_expanded  # [M, N, C]
            result = torch.sum(weighted_tensor, dim=0)  # [N, C]
            return result

        # Compute branch fusion weights
        branch_sum_scores = torch.sum(stacked_sparsity_scores, dim=0, keepdim=True)
        fuse_weights = stacked_sparsity_scores / branch_sum_scores

        # Combine multi-stage multi-branch scores
        final_bbox_scores = (
            weighted_combination(fuse_weights, bbox_results_list[0]['bbox_scores_list']) * 0.34 +
            weighted_combination(fuse_weights, bbox_results_list[1]['bbox_scores_list']) * 0.33 +
            weighted_combination(fuse_weights, bbox_results_list[2]['bbox_scores_list']) * 0.33
        )

        # Combine multi-branch regression
        final_bbox_results = weighted_combination(fuse_weights, merge_bbox_results)

        results_list = self.bbox_head[0].get_results(
            rois, final_bbox_scores, final_bbox_results,
            labels_3d, batch_input_metas, self.test_cfg)

        return results_list

    def _bbox_forward_train(self, seg_preds: torch.Tensor,
                            keypoint_features: torch.Tensor,
                            keypoints: torch.Tensor,
                            sampling_results: SamplingResult,
                            stacked_sparsity_scores: torch.Tensor) -> Tuple[List[Dict], Dict]:
        """Forward training function for multi-stage multi-branch bbox head.

        Args:
            seg_preds (torch.Tensor): Point-wise semantic features.
            keypoint_features (torch.Tensor): key points features from points encoder.
            keypoints (torch.Tensor): Coordinate of key points.
            sampling_results (:obj:`SamplingResult`): Sampled results used for training.
            stacked_sparsity_scores (torch.Tensor): Sparsity scores for each branch.

        Returns:
            tuple: (bbox_results_list, loss_merge_corner)
        """
        keypoint_features = keypoint_features * seg_preds.sigmoid().max(
            dim=-1, keepdim=True).values

        bbox_results_list = []
        rois = bbox3d2roi([res.bboxes for res in sampling_results])

        # Multi-stage forward
        for stage in range(self.num_stage):
            bbox_results = self._bbox_forward(stage, keypoint_features, keypoints, rois)
            bbox_targets = self.bbox_head[stage].get_targets(sampling_results, self.train_cfg)
            
            # Multi-branch loss computation
            loss_bbox = self.bbox_head[stage].loss(
                True,  # with_cls_pred
                bbox_results['bbox_scores_list'],
                bbox_results['bbox_reg_list'],
                rois,
                stacked_sparsity_scores,
                *bbox_targets
            )
            bbox_results.update(loss_bbox=loss_bbox)
            bbox_results_list.append(bbox_results)

        # Compute merged corner loss
        loss_merge_corner = {}
        if self.train_cfg.get('merge_corner_loss', False):
            assert self.num_stage == 3
            
            # Get final stage targets for corner loss
            final_bbox_targets = self.bbox_head[-1].get_targets(sampling_results, self.train_cfg)
            
            # Merge predictions from different stages
            merge_bbox_results = bbox_results_list[0]['bbox_reg']
            for i in range(len(merge_bbox_results)):
                merge_bbox_results[i][:, 3:6] = bbox_results_list[1]['bbox_reg'][i][:, 3:6]
                merge_bbox_results[i][:, 6] = bbox_results_list[2]['bbox_reg'][i][:, 6]


            loss_merge_corner = self.cal_avg_corner_loss_from_list(
                merge_bbox_results, rois, *final_bbox_targets
            )

        return bbox_results_list, loss_merge_corner

    def _bbox_forward(self, stage: int, keypoint_features: torch.Tensor,
                      keypoints: torch.Tensor, rois: torch.Tensor) -> dict:
        """Forward function for multi-branch bbox head at specific stage.

        Args:
            stage (int): Current stage index.
            keypoint_features (torch.Tensor): key points features from points encoder.
            keypoints (torch.Tensor): Coordinate of key points.
            rois (torch.Tensor): Roi boxes.

        Returns:
            dict: Contains multi-branch predictions of bbox_head.
        """
        pooled_keypoint_features = self.bbox_roi_extractor[stage](
            keypoint_features, keypoints[..., 1:], keypoints[..., 0].int(), rois)
        
        # Multi-branch forward (assuming bbox_head returns lists)
        bbox_score_list, bbox_reg_list = self.bbox_head[stage](pooled_keypoint_features)
        
        bbox_results = dict(
            bbox_scores_list=bbox_score_list,
            bbox_reg_list=bbox_reg_list,
            bbox_scores=bbox_score_list,
            bbox_reg=bbox_reg_list
        )
        return bbox_results

    def cal_avg_corner_loss_from_list(self,
                                    bbox_pred_list: list,
                                    rois: torch.Tensor,
                                    labels: torch.Tensor,
                                    bbox_targets: torch.Tensor,
                                    pos_gt_bboxes: torch.Tensor,
                                    reg_mask: torch.Tensor,
                                    label_weights: torch.Tensor,
                                    bbox_weights: torch.Tensor):
        """
        计算多个 bbox_pred 的 corner loss，并对它们取平均。

        Args:
            bbox_pred_list (list of torch.Tensor): 多个 bbox_pred tensor。
            其余参数与 self.cal_corner_loss 相同。

        Returns:
            dict: {'loss_merge_corner': mean_corner_loss}
        """
        corner_losses = []
        for bbox_pred in bbox_pred_list:
            loss_dict = self.cal_corner_loss(
                bbox_pred=bbox_pred,
                rois=rois,
                labels=labels,
                bbox_targets=bbox_targets,
                pos_gt_bboxes=pos_gt_bboxes,
                reg_mask=reg_mask,
                label_weights=label_weights,
                bbox_weights=bbox_weights
            )
            corner_losses.append(loss_dict['loss_merge_corner'])

        # Stack and average
        mean_corner_loss = torch.stack(corner_losses).mean()
        return {'loss_merge_corner': mean_corner_loss}


    def cal_corner_loss(self,bbox_pred: torch.Tensor,
             rois: torch.Tensor, labels: torch.Tensor,
             bbox_targets: torch.Tensor, pos_gt_bboxes: torch.Tensor,
             reg_mask: torch.Tensor, label_weights: torch.Tensor,
             bbox_weights: torch.Tensor):
        losses = dict()
        rcnn_batch_size = bbox_pred.shape[0]
        code_size = self.bbox_coder.code_size
        pos_inds = (reg_mask > 0)
        if pos_inds.any() == 0:
            losses['loss_merge_corner'] = 0 * bbox_pred.sum()
        else:
            pos_bbox_pred = bbox_pred.view(rcnn_batch_size, -1)[pos_inds]
            bbox_weights_flat = bbox_weights[pos_inds].view(-1, 1).repeat(
                1, pos_bbox_pred.shape[-1])

            pos_roi_boxes3d = rois[..., 1:].view(-1, code_size)[pos_inds]
            pos_roi_boxes3d = pos_roi_boxes3d.view(-1, code_size)
            batch_anchors = pos_roi_boxes3d.clone().detach()
            pos_rois_rotation = pos_roi_boxes3d[..., 6].view(-1)
            roi_xyz = pos_roi_boxes3d[..., 0:3].view(-1, 3)
            batch_anchors[..., 0:3] = 0
            # decode boxes
            pred_boxes3d = self.bbox_coder.decode(
                batch_anchors,
                pos_bbox_pred.view(-1, code_size)).view(-1, code_size)

            pred_boxes3d[..., 0:3] = rotation_3d_in_axis(
                pred_boxes3d[..., 0:3].unsqueeze(1),
                pos_rois_rotation,
                axis=2).squeeze(1)

            pred_boxes3d[:, 0:3] += roi_xyz

            # calculate corner loss
            loss_corner = self.get_corner_loss_lidar(
                pred_boxes3d, pos_gt_bboxes)
            losses['loss_merge_corner'] = loss_corner.mean()

        return losses

    def get_corner_loss_lidar(self,
                              pred_bbox3d: torch.Tensor,
                              gt_bbox3d: torch.Tensor,
                              delta: float = 1.0) -> torch.Tensor:
        """Calculate corner loss of given boxes.

        Args:
            pred_bbox3d (torch.FloatTensor): Predicted boxes in shape (N, 7).
            gt_bbox3d (torch.FloatTensor): Ground truth boxes in shape (N, 7).
            delta (float, optional): huber loss threshold. Defaults to 1.0

        Returns:
            torch.FloatTensor: Calculated corner loss in shape (N).
        """
        assert pred_bbox3d.shape[0] == gt_bbox3d.shape[0]

        # This is a little bit hack here because we assume the box for
        # Part-A2 is in LiDAR coordinates
        gt_boxes_structure = LiDARInstance3DBoxes(gt_bbox3d)
        pred_box_corners = LiDARInstance3DBoxes(pred_bbox3d).corners
        gt_box_corners = gt_boxes_structure.corners

        # This flip only changes the heading direction of GT boxes
        gt_bbox3d_flip = gt_boxes_structure.clone()
        gt_bbox3d_flip.tensor[:, 6] += np.pi
        gt_box_corners_flip = gt_bbox3d_flip.corners

        corner_dist = torch.min(
            torch.norm(pred_box_corners - gt_box_corners, dim=2),
            torch.norm(pred_box_corners - gt_box_corners_flip,
                       dim=2))  # (N, 8)
        # huber loss
        abs_error = torch.abs(corner_dist)
        corner_loss = torch.where(abs_error < delta,
                                  0.5 * abs_error**2 / delta,
                                  abs_error - 0.5 * delta)
        return corner_loss.mean(dim=1)

    def count_points_in_bbox(self,points_list, bboxes_list):
        assert len(points_list) == len(bboxes_list), "points_list and bboxes_list must have the same length"
        results = []

        for points, bboxes in zip(points_list, bboxes_list):
            bboxes_bottom_center = bboxes.clone()
            bboxes_bottom_center[:, 2] -= bboxes_bottom_center[:, 5] / 2

            batched_points = points.unsqueeze(0)
            batched_boxes = bboxes_bottom_center.unsqueeze(0)

            mask = points_in_boxes_all(batched_points, batched_boxes)

            count = mask[0].sum(dim=0)
            results.append(count)

        return results
    
    def sparsity_scoring(self, 
                     points_count_list: List[torch.Tensor], 
                     branch_num: int, global_min: float = 0.0, 
                     global_max: float = 600.0, 
                     sigma_scale: float = 0.5) -> List[torch.Tensor]:
        sparsity_scores_list = []
        for points_count in points_count_list:
            sparsity_scores_list.append(self.gaussian_fixed_interval_scoring(points_count, branch_num, global_min, global_max, sigma_scale))

        return sparsity_scores_list
        
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

    def _assign_and_sample(
            self, proposal_list: InstanceList,
            batch_gt_instances_3d: InstanceList) -> List[SamplingResult]:
        """Assign and sample proposals for training.

        Args:
            proposal_list (list[:obj:`InstancesData`]): Proposals produced by
                rpn head.
            batch_gt_instances_3d (list[:obj:`InstanceData`]): Batch of
                gt_instances. It usually includes ``bboxes_3d`` and
                ``labels_3d`` attributes.

        Returns:
            list[:obj:`SamplingResult`]: Sampled results of each training
                sample.
        """
        sampling_results = []
        # bbox assign
        for batch_idx in range(len(proposal_list)):
            cur_proposal_list = proposal_list[batch_idx]
            cur_boxes = cur_proposal_list['bboxes_3d']
            cur_labels_3d = cur_proposal_list['labels_3d']
            cur_gt_instances_3d = batch_gt_instances_3d[batch_idx]
            cur_gt_instances_3d.bboxes_3d = cur_gt_instances_3d.\
                bboxes_3d.tensor
            cur_gt_bboxes = batch_gt_instances_3d[batch_idx].bboxes_3d.to(
                cur_boxes.device)
            cur_gt_labels = batch_gt_instances_3d[batch_idx].labels_3d

            batch_num_gts = 0
            # 0 is bg
            batch_gt_indis = cur_gt_labels.new_full((len(cur_boxes), ), 0)
            batch_max_overlaps = cur_boxes.tensor.new_zeros(len(cur_boxes))
            # -1 is bg
            batch_gt_labels = cur_gt_labels.new_full((len(cur_boxes), ), -1)

            # each class may have its own assigner
            if isinstance(self.bbox_assigner, list):
                for i, assigner in enumerate(self.bbox_assigner):
                    gt_per_cls = (cur_gt_labels == i)
                    pred_per_cls = (cur_labels_3d == i)
                    cur_assign_res = assigner.assign(
                        cur_proposal_list[pred_per_cls],
                        cur_gt_instances_3d[gt_per_cls])
                    # gather assign_results in different class into one result
                    batch_num_gts += cur_assign_res.num_gts
                    # gt inds (1-based)
                    gt_inds_arange_pad = gt_per_cls.nonzero(
                        as_tuple=False).view(-1) + 1
                    # pad 0 for indice unassigned
                    gt_inds_arange_pad = F.pad(
                        gt_inds_arange_pad, (1, 0), mode='constant', value=0)
                    # pad -1 for indice ignore
                    gt_inds_arange_pad = F.pad(
                        gt_inds_arange_pad, (1, 0), mode='constant', value=-1)
                    # convert to 0~gt_num+2 for indices
                    gt_inds_arange_pad += 1
                    # now 0 is bg, >1 is fg in batch_gt_indis
                    batch_gt_indis[pred_per_cls] = gt_inds_arange_pad[
                        cur_assign_res.gt_inds + 1] - 1
                    batch_max_overlaps[
                        pred_per_cls] = cur_assign_res.max_overlaps
                    batch_gt_labels[pred_per_cls] = cur_assign_res.labels

                assign_result = AssignResult(batch_num_gts, batch_gt_indis,
                                             batch_max_overlaps,
                                             batch_gt_labels)
            else:  # for single class
                assign_result = self.bbox_assigner.assign(
                    cur_proposal_list, cur_gt_instances_3d)
            # sample boxes
            sampling_result = self.bbox_sampler.sample(assign_result,
                                                       cur_boxes.tensor,
                                                       cur_gt_bboxes,
                                                       cur_gt_labels)
            sampling_results.append(sampling_result)
        return sampling_results

    def _semantic_forward_train(self, keypoint_features: torch.Tensor,
                                keypoints: torch.Tensor,
                                batch_gt_instances_3d: InstanceList) -> dict:
        """Train semantic head.

        Args:
            keypoint_features (torch.Tensor): key points features
                from points encoder.
            keypoints (torch.Tensor): Coordinate of key points.
            batch_gt_instances_3d (list[:obj:`InstanceData`]): Batch of
                gt_instances. It usually includes ``bboxes_3d`` and
                ``labels_3d`` attributes.

        Returns:
            dict: Segmentation results including losses
        """
        semantic_results = self.semantic_head(keypoint_features)
        semantic_targets = self.semantic_head.get_targets(
            keypoints, batch_gt_instances_3d)
        loss_semantic = self.semantic_head.loss(semantic_results,
                                                semantic_targets)
        semantic_results.update(loss_semantic)
        return semantic_results
