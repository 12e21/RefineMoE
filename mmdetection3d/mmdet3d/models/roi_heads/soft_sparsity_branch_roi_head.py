# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Optional

import torch
from mmdet.models.task_modules import AssignResult
from mmdet.models.task_modules.samplers import SamplingResult
from torch.nn import functional as F

from mmdet3d.models.roi_heads.base_3droi_head import Base3DRoIHead
from mmdet3d.registry import MODELS
from mmdet3d.structures import bbox3d2roi
from mmdet3d.structures.det3d_data_sample import SampleList
from mmdet3d.utils import InstanceList
from mmcv.ops.points_in_boxes import points_in_boxes_all

@MODELS.register_module()
class SoftSparsityBranchRoiHead(Base3DRoIHead):
    """RoI head for PV-RCNN.

    Args:
        num_classes (int): The number of classes. Defaults to 3.
        semantic_head (dict, optional): Config of semantic head.
            Defaults to None.
        bbox_roi_extractor (dict, optional): Config of roi_extractor.
            Defaults to None.
        bbox_head (dict, optional): Config of bbox_head. Defaults to None.
        train_cfg (dict, optional): Train config of model.
            Defaults to None.
        test_cfg (dict, optional): Train config of model.
            Defaults to None.
        init_cfg (dict, optional): Initialize config of
            model. Defaults to None.
    """

    def __init__(self,
                 num_classes: int = 3,
                 num_branch: int = 2,
                 global_max: int = 10,
                 sigma_scale: float = 1.0,
                 semantic_head: Optional[dict] = None,
                 bbox_roi_extractor: Optional[dict] = None,
                 bbox_head: Optional[dict] = None,
                 train_cfg: Optional[dict] = None,
                 test_cfg: Optional[dict] = None,
                 init_cfg: Optional[dict] = None):
        super(SoftSparsityBranchRoiHead, self).__init__(
            bbox_head=bbox_head,
            bbox_roi_extractor=bbox_roi_extractor,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            init_cfg=init_cfg)
        self.num_classes = num_classes
        self.semantic_head = MODELS.build(semantic_head)
        self.num_branch = num_branch
        self.global_max = global_max
        self.sigma_scale = sigma_scale

        self.init_assigner_sampler()

    @property
    def with_semantic(self):
        """bool: whether the head has semantic branch"""
        return hasattr(self,
                       'semantic_head') and self.semantic_head is not None

    def loss(self, feats_dict: dict, rpn_results_list: InstanceList,
             batch_data_samples: SampleList, **kwargs) -> dict:
        """Training forward function of PVRCNNROIHead.

        Args:
            feats_dict (dict): Contains point-wise features.
            rpn_results_list (List[:obj:`InstanceData`]): Detection results
                of rpn head.
            batch_data_samples (List[:obj:`Det3DDataSample`]): The Data
                samples. It usually includes information such as
                `gt_instance_3d`, `gt_panoptic_seg_3d` and `gt_sem_seg_3d`.

        Returns:
            dict: losses from each head.

            - loss_semantic (torch.Tensor): loss of semantic head.
            - loss_bbox (torch.Tensor): loss of bboxes.
            - loss_cls (torch.Tensor): loss of object classification.
            - loss_corner (torch.Tensor): loss of bboxes corners.
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
        if self.with_semantic:
            semantic_results = self._semantic_forward_train(
                feats_dict['keypoint_features'], feats_dict['keypoints'],
                batch_gt_instances_3d)
            losses['loss_semantic'] = semantic_results['loss_semantic']

        sample_results = self._assign_and_sample(rpn_results_list,
                                                 batch_gt_instances_3d)
        # generate sparsity group index
        source_points_list = feats_dict['points']
        source_points_list = [points[:,:3] for points in source_points_list]
        bbox_list = [res.bboxes for res in sample_results]
        points_count_list =self.count_points_in_bbox(source_points_list, bbox_list)

        sparsity_scores_list = self.sparsity_scoring(points_count_list=points_count_list, 
                                                     branch_num=self.num_branch,
                                                     global_max=self.global_max,
                                                     sigma_scale=self.sigma_scale)
        stacked_sparsity_scores = torch.stack(sparsity_scores_list).reshape(-1,sparsity_scores_list[0].shape[-1])

        if self.with_bbox:
            bbox_results = self._bbox_forward_train(
                semantic_results['seg_preds'],
                feats_dict['fusion_keypoint_features'],
                feats_dict['keypoints'], 
                sample_results,
                stacked_sparsity_scores
                )
            losses.update(bbox_results['loss_bbox'])

        return losses

    def predict(self, feats_dict: dict, rpn_results_list: InstanceList,
                batch_data_samples: SampleList, **kwargs) -> SampleList:
        """Perform forward propagation of the roi head and predict detection
        results on the features of the upstream network.

        Args:
            feats_dict (dict): Contains point-wise features.
            rpn_results_list (List[:obj:`InstanceData`]): Detection results
                of rpn head.
            batch_data_samples (List[:obj:`Det3DDataSample`]): The Data
                samples. It usually includes information such as
                `gt_instance_3d`, `gt_panoptic_seg_3d` and `gt_sem_seg_3d`.

        Returns:
            list[:obj:`InstanceData`]: Detection results of each sample
            after the post process.
            Each item usually contains following keys.

            - scores_3d (Tensor): Classification scores, has a shape
              (num_instances, )
            - labels_3d (Tensor): Labels of bboxes, has a shape
              (num_instances, ).
            - bboxes_3d (BaseInstance3DBoxes): Prediction of bboxes,
              contains a tensor with shape (num_instances, C), where
              C >= 7.
        """
        assert self.with_bbox, 'Bbox head must be implemented.'
        assert self.with_semantic, 'Semantic head must be implemented.'

        batch_input_metas = [
            data_samples.metainfo for data_samples in batch_data_samples
        ]

        semantic_results = self.semantic_head(feats_dict['keypoint_features'])
        point_features = feats_dict[
            'fusion_keypoint_features'] * semantic_results[
                'seg_preds'].sigmoid().max(
                    dim=-1, keepdim=True).values
        
         # generate sparsity group index
        source_points_list = feats_dict['points']
        source_points_list = [points[:,:3] for points in source_points_list]
        bbox_list = [res.bboxes_3d.tensor for res in rpn_results_list]
        points_count_list =self.count_points_in_bbox(source_points_list, bbox_list)

        sparsity_scores_list = self.sparsity_scoring(points_count_list=points_count_list, 
                                                     branch_num=self.num_branch,
                                                     global_max=self.global_max,
                                                     sigma_scale=self.sigma_scale)
        
        stacked_sparsity_scores = torch.stack(sparsity_scores_list).reshape(-1,sparsity_scores_list[0].shape[-1])
        

        rois = bbox3d2roi(
            [res['bboxes_3d'].tensor for res in rpn_results_list])
        labels_3d = [res['labels_3d'] for res in rpn_results_list]
        bbox_results = self._bbox_forward(point_features,
                                          feats_dict['keypoints'], rois)
        
        branch_sum_scores = torch.sum(stacked_sparsity_scores, dim=0, keepdim=True)
        fuse_weights = stacked_sparsity_scores / branch_sum_scores
        
        def weighted_combination(weights, tensor):
            weights_expanded = weights.unsqueeze(-1)  # 形状变为[M, N, 1]
            weighted_tensor = tensor * weights_expanded  # 形状[M, N, C]
            result = torch.sum(weighted_tensor, dim=0)  # 形状[N, C]
            return result
        
        final_bbox_scores = weighted_combination(fuse_weights, torch.stack(bbox_results['bbox_scores_list'], dim=0))
        final_bbox_reg = weighted_combination(fuse_weights, torch.stack(bbox_results['bbox_reg_list'], dim=0))

        # final_bbox_scores = bbox_results['bbox_scores_list'][1]
        # final_bbox_reg = bbox_results['bbox_reg_list'][1]

        # <<< 构造1 stage的分数 <<<
        # roi_scores = torch.cat([
        #     res['scores_3d'].unsqueeze(1) for res in rpn_results_list
        # ],dim=1)
        # roi_scores = torch.log(roi_scores / (1 - roi_scores))
        # >>> 构造1 stage的分数 >>>
        
        # <<< 构造zero reg <<<
        # zero_reg = torch.zeros_like(bbox_results['bbox_reg'])
        # >>> 构造zero reg >>>
        results_list = self.bbox_head.get_results(rois,
                                                  final_bbox_scores,
                                                #   roi_scores,
                                                  final_bbox_reg,
                                                #   zero_reg,
                                                  labels_3d, batch_input_metas,
                                                  self.test_cfg)

        return results_list

    def _bbox_forward_train(self, seg_preds: torch.Tensor,
                            keypoint_features: torch.Tensor,
                            keypoints: torch.Tensor,
                            sampling_results: SamplingResult,
                            stacked_sparsity_scores: torch.Tensor) -> dict:
        """Forward training function of roi_extractor and bbox_head.

        Args:
            seg_preds (torch.Tensor): Point-wise semantic features.
            keypoint_features (torch.Tensor): key points features
                from points encoder.
            keypoints (torch.Tensor): Coordinate of key points.
            sampling_results (:obj:`SamplingResult`): Sampled results used
                for training.

        Returns:
            dict: Forward results including losses and predictions.
        """

        rois = bbox3d2roi([res.bboxes for res in sampling_results])
        keypoint_features = keypoint_features * seg_preds.sigmoid().max(
            dim=-1, keepdim=True).values
        bbox_results = self._bbox_forward(keypoint_features, keypoints, rois)

        bbox_targets = self.bbox_head.get_targets(sampling_results,
                                                  self.train_cfg)
        
        loss_bbox = self.bbox_head.loss(bbox_results['bbox_scores_list'],
                                        bbox_results['bbox_reg_list'], rois,
                                        stacked_sparsity_scores,
                                        *bbox_targets
                                        )

        bbox_results.update(loss_bbox=loss_bbox)
        return bbox_results

    def _bbox_forward(self, keypoint_features: torch.Tensor,
                      keypoints: torch.Tensor, rois: torch.Tensor) -> dict:
        """Forward function of roi_extractor and bbox_head used in both
        training and testing.

        Args:
            rois (Tensor): Roi boxes.
            keypoint_features (torch.Tensor): key points features
                from points encoder.
            keypoints (torch.Tensor): Coordinate of key points.
            rois (Tensor): Roi boxes.

        Returns:
            dict: Contains predictions of bbox_head and
                features of roi_extractor.
        """
        pooled_keypoint_features = self.bbox_roi_extractor(
            keypoint_features, keypoints[..., 1:], keypoints[..., 0].int(),
            rois)
        bbox_score_list, bbox_reg_list = self.bbox_head(pooled_keypoint_features)

        bbox_results = dict(bbox_scores_list=bbox_score_list, bbox_reg_list=bbox_reg_list)
        return bbox_results

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