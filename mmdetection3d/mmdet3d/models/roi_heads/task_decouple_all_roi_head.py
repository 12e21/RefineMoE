# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Optional
from torch.nn import ModuleList
import torch
from mmdet.models.task_modules import AssignResult
from mmdet.models.task_modules.samplers import SamplingResult
from torch.nn import functional as F

from mmdet3d.models.roi_heads.base_3droi_head import Base3DRoIHead
from mmdet3d.registry import MODELS,TASK_UTILS
from mmdet3d.structures import bbox3d2roi
from mmdet3d.structures.det3d_data_sample import SampleList
from mmdet3d.utils import InstanceList
from mmdet3d.structures.bbox_3d import rotation_3d_in_axis,LiDARInstance3DBoxes
import numpy as np

@MODELS.register_module()
class TaskDecoupleAllRoiHead(Base3DRoIHead):
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
                 num_stage: int = 1,
                 stage_loss_weights: list = [1.0],
                 num_classes: int = 3,
                 semantic_head: Optional[dict] = None,
                 bbox_roi_extractor: Optional[dict] = None,
                 bbox_head: Optional[dict] = None,
                 bbox_coder: Optional[dict] = None,
                 train_cfg: Optional[dict] = None,
                 test_cfg: Optional[dict] = None,
                 init_cfg: Optional[dict] = None):
        self.num_stage = num_stage
        self.stage_loss_weights = stage_loss_weights
        super(TaskDecoupleAllRoiHead, self).__init__(
            bbox_head=bbox_head,
            bbox_roi_extractor=bbox_roi_extractor,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            init_cfg=init_cfg)
        self.num_classes = num_classes
        self.semantic_head = MODELS.build(semantic_head)
        self.bbox_coder = TASK_UTILS.build(bbox_coder)
        self.init_assigner_sampler()

    def init_bbox_head(self, bbox_roi_extractor: dict,
                       bbox_head: dict) -> None:
        """Initialize box head and box roi extractor.

        Args:
            bbox_roi_extractor (dict or ConfigDict): Config of box
                roi extractor.
            bbox_head (dict or ConfigDict): Config of box in box head.
        """

        self.bbox_roi_extractor = ModuleList()
        self.bbox_head = ModuleList()
        if not isinstance(bbox_roi_extractor, list):
            bbox_roi_extractor = [
                bbox_roi_extractor for _ in range(self.num_stage)
            ]
        if not isinstance(bbox_head, list):
            bbox_head = [bbox_head for _ in range(self.num_stage)]
        assert len(bbox_roi_extractor) == len(bbox_head) == self.num_stage
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

        sample_results = self._assign_and_sample(rpn_results_list,
                                                 batch_gt_instances_3d)
        if self.with_bbox:
            bbox_results_list , loss_merge_corner = self._bbox_forward_train(
                semantic_results['seg_preds'],
                feats_dict['fusion_keypoint_features'],
                feats_dict['keypoints'], sample_results)

            losses_list = [bbox_res['loss_bbox'] for bbox_res in bbox_results_list]
            for stage in range(self.num_stage):
                stage_loss_weight = self.stage_loss_weights[stage]
                for name, value in losses_list[stage].items():
                    losses[f's{stage}.{name}'] = (
                        value * stage_loss_weight if 'loss' in name else value)

            losses.update(loss_merge_corner)
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
        rois = bbox3d2roi(
            [res['bboxes_3d'].tensor for res in rpn_results_list])
        labels_3d = [res['labels_3d'] for res in rpn_results_list]


        bbox_results_list=[]

        for stage in range(self.num_stage):
            # 新的bbox存放在bbox_results中
            bbox_results = self._bbox_forward(stage,point_features, feats_dict['keypoints'], rois)
            bbox_results_list.append(bbox_results)

        assert self.num_stage == 7
        # change this to only refine specific subtask
        refine_reg_option = "all"
        assert refine_reg_option in ["all", "center","shape","orient"]
        if refine_reg_option == "all":
            merge_bbox_results = bbox_results_list[0]['bbox_reg']
            merge_bbox_results[:, 1] = bbox_results_list[1]['bbox_reg'][:, 1]
            merge_bbox_results[:, 2] = bbox_results_list[2]['bbox_reg'][:, 2]
            merge_bbox_results[:, 3] = bbox_results_list[3]['bbox_reg'][:, 3]
            merge_bbox_results[:, 4] = bbox_results_list[4]['bbox_reg'][:, 4]
            merge_bbox_results[:, 5] = bbox_results_list[5]['bbox_reg'][:, 5]
            merge_bbox_results[:, 6] = bbox_results_list[6]['bbox_reg'][:, 6]
        # elif refine_reg_option == "center":
        #     merge_bbox_results = bbox_results_list[0]['bbox_reg']
        # elif refine_reg_option == "shape":
        #     merge_bbox_results = bbox_results_list[1]['bbox_reg']
        # elif refine_reg_option == "orient":
        #     merge_bbox_results = bbox_results_list[2]['bbox_reg']
        # construct final bbox scores, linear combination of 3 stage score
        final_bbox_scores = (
                                bbox_results_list[0]['bbox_scores'] * 0.34 +
                                bbox_results_list[1]['bbox_scores'] * 0.33+
                                bbox_results_list[2]['bbox_scores'] * 0.33+
                                bbox_results_list[3]['bbox_scores'] * 0.34 +
                                bbox_results_list[4]['bbox_scores'] * 0.33+
                                bbox_results_list[5]['bbox_scores'] * 0.33+
                                bbox_results_list[6]['bbox_scores'] * 0.33
                            )
        
        #  <<< construct 1 stage cls score and reg <<<
        # roi_scores = torch.cat([
        # res['scores_3d'].unsqueeze(1) for res in rpn_results_list
        # ],dim=1)
        # roi_scores = torch.log(roi_scores / (1 - roi_scores))

        # zero_reg = torch.zeros_like(merge_bbox_results)
        #  >>> construct 1 stage cls score and reg >>>

        results_list = self.bbox_head[0].get_results(rois,
                                                    # bbox_results_list[-1]['bbox_scores'],
                                                    final_bbox_scores,
                                                    # roi_scores,
                                                    merge_bbox_results,
                                                    # zero_reg,
                                                    labels_3d, batch_input_metas,
                                                    self.test_cfg)

        return results_list

    def _bbox_forward_train(self, seg_preds: torch.Tensor,
                            keypoint_features: torch.Tensor,
                            keypoints: torch.Tensor,
                            sampling_results: SamplingResult):
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

        keypoint_features = keypoint_features * seg_preds.sigmoid().max(
            dim=-1, keepdim=True).values

        bbox_results_list=[]

        rois = bbox3d2roi([res.bboxes for res in sampling_results])


        for stage in range(self.num_stage):
            # 新的bbox存放在bbox_results中
            bbox_results = self._bbox_forward(stage,keypoint_features, keypoints,rois)
            bbox_targets = self.bbox_head[stage].get_targets(sampling_results, self.train_cfg)
            loss_bbox = self.bbox_head[stage].loss(
                True,
                bbox_results['bbox_scores'],
                bbox_results['bbox_reg'],
                rois,
                *bbox_targets
            )
            bbox_results.update(loss_bbox=loss_bbox)
            bbox_results_list.append(bbox_results)

        # 形成三个stage的差量预测的合并
        if self.train_cfg.get('merge_corner_loss', False):
            assert self.num_stage == 7
            merge_bbox_results = bbox_results_list[0]['bbox_reg']
            merge_bbox_results[:, 1] = bbox_results_list[1]['bbox_reg'][:, 1]
            merge_bbox_results[:, 2] = bbox_results_list[2]['bbox_reg'][:, 2]
            merge_bbox_results[:, 3] = bbox_results_list[3]['bbox_reg'][:, 3]
            merge_bbox_results[:, 4] = bbox_results_list[4]['bbox_reg'][:, 4]
            merge_bbox_results[:, 5] = bbox_results_list[5]['bbox_reg'][:, 5]
            merge_bbox_results[:, 6] = bbox_results_list[6]['bbox_reg'][:, 6]

            loss_merge_corner = self.cal_corner_loss(
                merge_bbox_results,
                rois,
                *bbox_targets
            )


        return bbox_results_list , loss_merge_corner

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

    def _bbox_forward(self, stage ,keypoint_features: torch.Tensor,
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
        pooled_keypoint_features = self.bbox_roi_extractor[stage](
            keypoint_features, keypoints[..., 1:], keypoints[..., 0].int(),
            rois)
        bbox_score, bbox_reg = self.bbox_head[stage](pooled_keypoint_features)

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
