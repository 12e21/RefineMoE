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
import math
from torch.nn import ModuleList
from mmengine.structures import InstanceData
from mmcv.ops.points_in_boxes import points_in_boxes_all

@MODELS.register_module()
class SparsityBranchRoiHead(Base3DRoIHead):
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
                 semantic_head: Optional[dict] = None,
                 bbox_roi_extractor: Optional[dict] = None,
                 bbox_head: Optional[dict] = None,
                 train_cfg: Optional[dict] = None,
                 test_cfg: Optional[dict] = None,
                 init_cfg: Optional[dict] = None):
        super(SparsityBranchRoiHead, self).__init__(
            bbox_head=bbox_head,
            bbox_roi_extractor=bbox_roi_extractor,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            init_cfg=init_cfg)
        self.num_classes = num_classes
        self.semantic_head = MODELS.build(semantic_head)

        self.init_assigner_sampler()

    def init_bbox_head(self, bbox_roi_extractor: dict,
                       bbox_head: dict) -> None:
        """Initialize box head and box roi extractor.

        Args:
            bbox_roi_extractor (dict or ConfigDict): Config of box
                roi extractor.
            bbox_head (dict or ConfigDict): Config of box in box head.
        """
        self.num_branch = 2
        self.bbox_roi_extractor = ModuleList()
        self.bbox_head = ModuleList()
        bbox_roi_extractor = [
            bbox_roi_extractor for _ in range(self.num_branch)
        ]
        bbox_head = [bbox_head for _ in range(self.num_branch)]

        assert len(bbox_roi_extractor) == len(bbox_head) == self.num_branch
        for roi_extractor, head in zip(bbox_roi_extractor, bbox_head):
            self.bbox_head.append(MODELS.build(head))
            self.bbox_roi_extractor.append(MODELS.build(roi_extractor))

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


        source_points_list = feats_dict['points']
        source_points_list = [points[:,:3] for points in source_points_list]
        
        bbox_list = [res.bboxes_3d.tensor for res in rpn_results_list]

        points_count_list =self.count_points_in_bbox(source_points_list, bbox_list)

        sparse_mask_list = [points_count < 100 for points_count in points_count_list]

        sparse_rpn_results_list = [rpn_results_list[i][sparse_mask_list[i]] for i in range(len(rpn_results_list))]
        dense_rpn_results_list = [rpn_results_list[i][~sparse_mask_list[i]] for i in range(len(rpn_results_list))]

        sparse_batch_gt_instances_3d = [batch_gt_instance.clone() for batch_gt_instance in batch_gt_instances_3d]
        dense_batch_gt_instances_3d = [batch_gt_instance.clone() for batch_gt_instance in batch_gt_instances_3d]
        sparse_sample_results = self._assign_and_sample(sparse_rpn_results_list,
                                                 sparse_batch_gt_instances_3d)
        dense_sample_results = self._assign_and_sample(dense_rpn_results_list,
                                                    dense_batch_gt_instances_3d)

        # currently only support 1 batch size, because bbox head compute with  batch size, 
        # if you have implementation for batch size, please open a issue
        if self.with_bbox:
            if not sparse_mask_list[0].all():
                bbox_results = self._bbox_forward_train(
                    1,
                    semantic_results['seg_preds'],
                    feats_dict['fusion_keypoint_features'],
                    feats_dict['keypoints'], dense_sample_results)
                
                # multiply dense branch loss by 0.5
                bbox_results['loss_bbox'] = {k: v * 0.5 for k, v in bbox_results['loss_bbox'].items()}
                losses.update(bbox_results['loss_bbox'])

            if not (~sparse_mask_list[0]).all():
                bbox_results = self._bbox_forward_train(
                    0,
                    semantic_results['seg_preds'],
                    feats_dict['fusion_keypoint_features'],
                    feats_dict['keypoints'], sparse_sample_results)
                
                # multiply sparse branch loss by 0.5
                bbox_results['loss_bbox'] = {k: v * 0.5 for k, v in bbox_results['loss_bbox'].items()}
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

        source_points_list = feats_dict['points']
        source_points_list = [points[:,:3] for points in source_points_list]
        
        bbox_list = [res.bboxes_3d.tensor for res in rpn_results_list]

        points_count_list =self.count_points_in_bbox(source_points_list, bbox_list)

        sparse_mask_list = [points_count < 100 for points_count in points_count_list]

        sparse_rpn_results_list = [rpn_results_list[i][sparse_mask_list[i]] for i in range(len(rpn_results_list))]
        dense_rpn_results_list = [rpn_results_list[i][~sparse_mask_list[i]] for i in range(len(rpn_results_list))]

        if not (~sparse_mask_list[0]).all():

            sparse_rois = bbox3d2roi(
                [res['bboxes_3d'].tensor for res in sparse_rpn_results_list])
            sparse_labels_3d = [res['labels_3d'] for res in sparse_rpn_results_list]
            sparse_bbox_results = self._bbox_forward(0,point_features,
                                            feats_dict['keypoints'], sparse_rois)

            #  <<< construct 1 stage cls score and reg <<<
            # sparse_roi_scores = torch.cat([
            # res['scores_3d'].unsqueeze(1) for res in sparse_rpn_results_list
            # ],dim=1)
            # sparse_roi_scores = torch.log(sparse_roi_scores / (1 - sparse_roi_scores))

            # sparse_zero_reg = torch.zeros_like(sparse_bbox_results['bbox_reg'])
            #  >>> construct 1 stage cls score and reg >>>

            sparse_results_list = self.bbox_head[0].get_results(sparse_rois,
                                                    sparse_bbox_results['bbox_scores'],
                                                    # sparse_roi_scores,
                                                    sparse_bbox_results['bbox_reg'],
                                                    # sparse_zero_reg,
                                                    sparse_labels_3d, batch_input_metas,
                                                    self.test_cfg)
        if not sparse_mask_list[0].all():
            dense_rois = bbox3d2roi(
                [res['bboxes_3d'].tensor for res in dense_rpn_results_list])
            dense_labels_3d = [res['labels_3d'] for res in dense_rpn_results_list]
            dense_bbox_results = self._bbox_forward(1,point_features,
                                                feats_dict['keypoints'], dense_rois)
            
            #  <<< construct 1 stage cls score and reg <<<
            # dense_roi_scores = torch.cat([
            # res['scores_3d'].unsqueeze(1) for res in dense_rpn_results_list
            # ],dim=1)
            # dense_roi_scores = torch.log(dense_roi_scores / (1 - dense_roi_scores))

            # dense_zero_reg = torch.zeros_like(dense_bbox_results['bbox_reg'])
            #  >>> construct 1 stage cls score and reg >>>


            dense_results_list = self.bbox_head[1].get_results(dense_rois,
                                                        dense_bbox_results['bbox_scores'],
                                                        # dense_roi_scores,
                                                        dense_bbox_results['bbox_reg'],
                                                        # dense_zero_reg,
                                                        dense_labels_3d, batch_input_metas,
                                                        self.test_cfg)

        # merge sparse and dense results
        results_list = []
        if sparse_mask_list[0].all():
            results_list = sparse_results_list
        elif (~sparse_mask_list[0]).all():
            results_list = dense_results_list
        else:        
            for i in range(len(rpn_results_list)):
                results_list.append(InstanceData.cat([sparse_results_list[i], dense_results_list[i]]))

        return results_list

    def _bbox_forward_train(self,branch,seg_preds: torch.Tensor,
                            keypoint_features: torch.Tensor,
                            keypoints: torch.Tensor,
                            sampling_results: SamplingResult) -> dict:
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
        bbox_results = self._bbox_forward(branch,keypoint_features, keypoints, rois)

        bbox_targets = self.bbox_head[branch].get_targets(sampling_results,
                                                  self.train_cfg)
        loss_bbox = self.bbox_head[branch].loss(bbox_results['bbox_scores'],
                                        bbox_results['bbox_reg'], rois,
                                        *bbox_targets)

        if branch == 0:
            # 将loss_bbox字典中的所有key添加前缀sparse
            loss_bbox = {f'sparse_{key}': value for key, value in loss_bbox.items()}
        else:
            loss_bbox = {f'dense_{key}': value for key, value in loss_bbox.items()}
        bbox_results.update(loss_bbox=loss_bbox)
        return bbox_results

    def _bbox_forward(self,branch,keypoint_features: torch.Tensor,
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
        pooled_keypoint_features = self.bbox_roi_extractor[branch](
            keypoint_features, keypoints[..., 1:], keypoints[..., 0].int(),
            rois)
        bbox_score, bbox_reg = self.bbox_head[branch](pooled_keypoint_features)

        bbox_results = dict(bbox_scores=bbox_score, bbox_reg=bbox_reg)
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
    
    def compute_bbox_point_densities(self,points_list, bboxes_list):
        """
        Compute the point density (points per unit volume) for each bounding box.

        Args:
            points_list (List[Tensor]): List of point clouds, each with shape [N, 3]
            bboxes_list (List[Tensor]): List of bounding boxes, each with shape [T, 7]

        Returns:
            List[Tensor]: A list of tensors, each tensor has shape [T], indicating
                        the point density (point count / volume) of each bbox.
        """
        # Step 1: 获取每个 box 中的点数
        point_counts_list = self.count_points_in_bbox(points_list, bboxes_list)

        densities_list = []

        for bboxes, point_counts in zip(bboxes_list, point_counts_list):
            # Step 2: 获取 box 的尺寸
            x_sizes = bboxes[:, 3]
            y_sizes = bboxes[:, 4]
            z_sizes = bboxes[:, 5]

            # Step 3: 计算每个 box 的体积
            volumes = x_sizes * y_sizes * z_sizes

            # Step 4: 计算点密度（避免除以零）
            densities = point_counts.float() / (volumes + 1e-6)

            densities_list.append(densities)

        return densities_list
