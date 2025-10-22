# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Optional
from torch.nn import ModuleList
import torch
from mmdet.models.task_modules import AssignResult
from mmdet.models.task_modules.samplers import SamplingResult
from torch.nn import functional as F

from mmdet3d.models.roi_heads.base_3droi_head import Base3DRoIHead
from mmdet3d.registry import MODELS
from mmdet3d.structures import bbox3d2roi
from mmdet3d.structures.det3d_data_sample import SampleList
from mmdet3d.utils import InstanceList
from mmengine.structures import InstanceData
import math

@MODELS.register_module()
class MMrefineRoiHead(Base3DRoIHead):
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
                 sparse_dense_loss_proportion: list = [],
                 divide_method='dynamic_median',
                 num_classes: int = 3,
                 semantic_head: Optional[dict] = None,
                 bbox_roi_extractor: Optional[dict] = None,
                 bbox_head: Optional[dict] = None,
                 train_cfg: Optional[dict] = None,
                 test_cfg: Optional[dict] = None,
                 init_cfg: Optional[dict] = None):
        self.num_stage = num_stage
        self.stage_loss_weights = stage_loss_weights
        super(Base3DRoIHead, self).__init__(
            bbox_head=bbox_head,
            bbox_roi_extractor=bbox_roi_extractor,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            init_cfg=init_cfg)
        self.num_classes = num_classes
        self.semantic_head = MODELS.build(semantic_head)
        self.with_sparsity_bbox_head = True
        self.sparse_dense_loss_proportion = sparse_dense_loss_proportion
        self.divide_method = divide_method
        self.init_assigner_sampler()

    def init_bbox_head(self, bbox_roi_extractor: dict,
                       bbox_head: dict) -> None:
        """Initialize box head and box roi extractor.

        Args:
            bbox_roi_extractor (dict or ConfigDict): Config of box
                roi extractor.
            bbox_head (dict or ConfigDict): Config of box in box head.
        """

        self.sparse_bbox_roi_extractor = ModuleList()
        self.sparse_bbox_head = ModuleList()
        self.dense_bbox_roi_extractor = ModuleList()
        self.dense_bbox_head = ModuleList()


        if not isinstance(bbox_roi_extractor, list):
            bbox_roi_extractor = [
                bbox_roi_extractor for _ in range(self.num_stage)
            ]
        if not isinstance(bbox_head, list):
            bbox_head = [bbox_head for _ in range(self.num_stage)]
        assert len(bbox_roi_extractor) == len(bbox_head) == self.num_stage
        for roi_extractor, head in zip(bbox_roi_extractor, bbox_head):
            # self.bbox_head.append(MODELS.build(head))
            # self.bbox_roi_extractor.append(MODELS.build(roi_extractor))
            self.sparse_bbox_roi_extractor.append(MODELS.build(roi_extractor))
            self.sparse_bbox_head.append(MODELS.build(head))
            self.dense_bbox_roi_extractor.append(MODELS.build(roi_extractor))
            self.dense_bbox_head.append(MODELS.build(head))

    @property
    def with_semantic(self):
        """bool: whether the head has semantic branch"""
        return hasattr(self,
                       'semantic_head') and self.semantic_head is not None

    def loss(self, feats_dict: dict, rpn_results_list: InstanceList,
             batch_data_samples: SampleList, **kwargs) -> dict:
        # for name, param in self.named_parameters():
        #     if param.grad is None:
        #         print(name)
        # import ipdb
        # ipdb.set_trace()
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

        keypoints = feats_dict['keypoints']

        assert self.divide_method in ["dynamic_median", "clswise_dynamic_median"]
        if self.divide_method == "dynamic_median":
            sparse_mask_list = self.dynamic_median_divide(keypoints, rpn_results_list)
        elif self.divide_method == "clswise_dynamic_median":
            sparse_mask_list = self.clswise_dynamic_median_divide(keypoints, rpn_results_list)
        sparse_count =dict()
        sparse_count["sparse_count"] = torch.tensor(sparse_mask_list[0].sum().item(),dtype=torch.float)
        sparse_count["dense_count"] = torch.tensor((~sparse_mask_list[0]).sum().item(),dtype=torch.float)

        sparse_rpn_results_list = [rpn_results_list[i][sparse_mask_list[i]] for i in range(len(rpn_results_list))]
        dense_rpn_results_list = [rpn_results_list[i][~sparse_mask_list[i]] for i in range(len(rpn_results_list))]

        sparse_batch_gt_instances_3d = [batch_gt_instance.clone() for batch_gt_instance in batch_gt_instances_3d]
        dense_batch_gt_instances_3d = [batch_gt_instance.clone() for batch_gt_instance in batch_gt_instances_3d]
        sparse_sample_results = self._assign_and_sample(sparse_rpn_results_list,
                                                 sparse_batch_gt_instances_3d)
        dense_sample_results = self._assign_and_sample(dense_rpn_results_list,
                                                    dense_batch_gt_instances_3d)

        if self.with_bbox or self.with_sparsity_bbox_head:
            # sparse branch
            bbox_results_list = self._bbox_forward_train(
                "sparse",
                semantic_results['seg_preds'],
                feats_dict['fusion_keypoint_features'],
                feats_dict['keypoints'], sparse_sample_results)

            losses_list = [bbox_res['loss_bbox'] for bbox_res in bbox_results_list]
            for stage in range(self.num_stage):
                stage_loss_weight = self.stage_loss_weights[stage]
                for name, value in losses_list[stage].items():
                    losses[f's{stage}.{name}'] = (
                        value * stage_loss_weight * self.sparse_dense_loss_proportion[0] if 'loss' in name else value)


            # dense branch
            bbox_results_list = self._bbox_forward_train(
                "dense",
                semantic_results['seg_preds'],
                feats_dict['fusion_keypoint_features'],
                feats_dict['keypoints'], dense_sample_results)

            losses_list = [bbox_res['loss_bbox'] for bbox_res in bbox_results_list]
            for stage in range(self.num_stage):
                stage_loss_weight = self.stage_loss_weights[stage]
                for name, value in losses_list[stage].items():
                    losses[f's{stage}.{name}'] = (
                        value * stage_loss_weight * self.sparse_dense_loss_proportion[1] if 'loss' in name else value)

            losses.update(sparse_count)
        return losses
    def dynamic_median_divide(self,keypoints, rpn_results_list):
        bbox_list = [res.bboxes_3d.tensor for res in rpn_results_list]
        keypoints_list = [keypoints[keypoints[..., 0] == i][:, 1:] for i in range(int(keypoints[..., 0].max().item()) + 1)]
        points_count_list =self.count_points_in_bbox(keypoints_list, bbox_list)

        mdedian_list = [points_count.median() for points_count in points_count_list]
        # 以中位数为阈值，划分为稀疏和密集
        assert len(mdedian_list) == len(points_count_list)
        sparse_mask_list = []
        for i in range(len(points_count_list)):
            sparse_mask_list.append(points_count_list[i] < mdedian_list[i])

        # sparse_mask_list = [points_count < 0.22713856399059296 for points_count in points_count_list]

        for i in range(len(sparse_mask_list)):
            if sparse_mask_list[i].all():
                sparse_mask_list[i][points_count_list[i].argmax()] = False
            if (~sparse_mask_list[i]).all():
                sparse_mask_list[i][points_count_list[i].argmin()] = True

            assert not all(sparse_mask_list[i]), 'All sparse_mask_list is True'

        return sparse_mask_list

    def clswise_dynamic_median_divide(self,keypoints, rpn_results_list):
        # 将rpn_results_list中的每个rpn_result的cls_preds,scores_3d,bboxes_3d,labels_3d都按照label排序(包括label本身)
        for i in range(len(rpn_results_list)):
            label_indice = rpn_results_list[i]['labels_3d'].argsort()
            for element in ["cls_preds", "scores_3d", "bboxes_3d", "labels_3d"]:
                rpn_results_list[i][element] = rpn_results_list[i][element][label_indice]

        for i in range(len(rpn_results_list)):
            # 确定每个proposal的label是经过sort的
            assert (rpn_results_list[i]['labels_3d'] == rpn_results_list[i].labels_3d.sort()[0]).all()

        bbox_list = [res.bboxes_3d.tensor for res in rpn_results_list]

        keypoints_list = [keypoints[keypoints[..., 0] == i][:, 1:] for i in range(int(keypoints[..., 0].max().item()) + 1)]
        points_count_list =self.count_points_in_bbox(keypoints_list, bbox_list)

        # 按照proposal的label将points_count_list分组
        points_count_list_grouped = []
        for i in range(len(points_count_list)):
            # 以proposal的label为分组依据,作为dict的key,value是对应的label的所有proposal的points_count，为一个list
            points_count_list_grouped.append(
                dict(zip(torch.unique(rpn_results_list[i]['labels_3d']), [points_count_list[i][rpn_results_list[i]['labels_3d'] == label] for label in torch.unique(rpn_results_list[i]['labels_3d'])]))
            )

        # 按照proposal的label将points_count_list分组后，计算每个label的中位数
        clswise_median_list = []
        for i in range(len(points_count_list_grouped)):
            clswise_median_list.append(
                dict(zip(points_count_list_grouped[i].keys(), [points_count.median() for points_count in points_count_list_grouped[i].values()]))
            )

        sparse_mask_list = []
        for i in range(len(points_count_list)):
            sparse_mask_list.append(
                torch.cat([points_count < clswise_median_list[i][label] for label, points_count in points_count_list_grouped[i].items()])
            )

        for i in range(len(sparse_mask_list)):
            if sparse_mask_list[i].all():
                sparse_mask_list[i][points_count_list[i].argmax()] = False
            if (~sparse_mask_list[i]).all():
                sparse_mask_list[i][points_count_list[i].argmin()] = True

            assert not all(sparse_mask_list[i]), 'All sparse_mask_list is True'

        return sparse_mask_list


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

        assert self.with_bbox or self.with_sparsity_bbox_head, 'Bbox head must be implemented.'
        assert self.with_semantic, 'Semantic head must be implemented.'

        batch_input_metas = [
            data_samples.metainfo for data_samples in batch_data_samples
        ]

        semantic_results = self.semantic_head(feats_dict['keypoint_features'])
        point_features = feats_dict[
            'fusion_keypoint_features'] * semantic_results[
                'seg_preds'].sigmoid().max(
                    dim=-1, keepdim=True).values

        keypoints = feats_dict['keypoints']

        assert self.divide_method in ["dynamic_median", "clswise_dynamic_median"]
        if self.divide_method == "dynamic_median":
            sparse_mask_list = self.dynamic_median_divide(keypoints, rpn_results_list)
        elif self.divide_method == "clswise_dynamic_median":
            sparse_mask_list = self.clswise_dynamic_median_divide(keypoints, rpn_results_list)

        sparse_rpn_results_list = [rpn_results_list[i][sparse_mask_list[i]] for i in range(len(rpn_results_list))]
        dense_rpn_results_list = [rpn_results_list[i][~sparse_mask_list[i]] for i in range(len(rpn_results_list))]


        # <<< 构造1 stage的分数 <<<
        # sparse_roi_scores = torch.cat([
        #     res['scores_3d'].unsqueeze(1) for res in sparse_rpn_results_list
        # ],dim=1)
        # sparse_roi_scores = torch.log(sparse_roi_scores / (1 - sparse_roi_scores))

        # dense_roi_scores = torch.cat([
        #     res['scores_3d'].unsqueeze(1) for res in dense_rpn_results_list
        # ],dim=1)
        # dense_roi_scores = torch.log(dense_roi_scores / (1 - dense_roi_scores))
        # >>> 构造1 stage的分数 >>>


        sparse_rois = bbox3d2roi(
            [res['bboxes_3d'].tensor for res in sparse_rpn_results_list])
        sparse_labels_3d = [res['labels_3d'] for res in sparse_rpn_results_list]

        sparse_bbox_results_list=[]
        sparse_rois_list=[]
        sparse_rois_list.append(sparse_rois)
        for stage in range(self.num_stage):
            # 新的bbox存放在bbox_results中
            if stage == self.num_stage - 1:
                bbox_results = self._bbox_forward(True,"sparse",stage,point_features, feats_dict['keypoints'], sparse_rois_list[stage])
            else:
                bbox_results = self._bbox_forward(False, "sparse", stage, point_features, feats_dict['keypoints'],
                                                  sparse_rois_list[stage])
            sparse_bbox_results_list.append(bbox_results)

            bbox_result=self.sparse_bbox_head[stage].decode_from_rois(sparse_rois_list[stage],bbox_results['bbox_reg'])
            sparse_rois_list.append(bbox3d2roi(bbox_result))

        sparse_result_stage = 1
        sparse_results_list = self.sparse_bbox_head[-sparse_result_stage].get_results(sparse_rois_list[-(sparse_result_stage+1)],
                                                  sparse_bbox_results_list[-sparse_result_stage]['bbox_scores'],
                                                #   sparse_roi_scores,
                                                  sparse_bbox_results_list[-sparse_result_stage]['bbox_reg'],
                                                  sparse_labels_3d, batch_input_metas,
                                                  self.test_cfg)

        dense_rois = bbox3d2roi(
            [res['bboxes_3d'].tensor for res in dense_rpn_results_list])
        dense_labels_3d = [res['labels_3d'] for res in dense_rpn_results_list]

        dense_bbox_results_list=[]
        dense_rois_list=[]
        dense_rois_list.append(dense_rois)
        for stage in range(self.num_stage):
            # 新的bbox存放在bbox_results中
            if stage == self.num_stage - 1:
                bbox_results = self._bbox_forward(True,"dense",stage,point_features, feats_dict['keypoints'], dense_rois_list[stage])
            else:
                bbox_results = self._bbox_forward(False, "dense", stage, point_features, feats_dict['keypoints'],
                                                  dense_rois_list[stage])
            dense_bbox_results_list.append(bbox_results)

            bbox_result=self.dense_bbox_head[stage].decode_from_rois(dense_rois_list[stage],bbox_results['bbox_reg'])
            dense_rois_list.append(bbox3d2roi(bbox_result))

        dense_result_stage = 1
        dense_results_list = self.dense_bbox_head[-dense_result_stage].get_results(dense_rois_list[-(dense_result_stage+1)],
                                                    dense_bbox_results_list[-dense_result_stage]['bbox_scores'],
                                                    # dense_roi_scores,
                                                    dense_bbox_results_list[-dense_result_stage]['bbox_reg'],
                                                    dense_labels_3d, batch_input_metas,
                                                    self.test_cfg)
        results_list = []
        for i in range(len(rpn_results_list)):
            results_list.append(InstanceData.cat([sparse_results_list[i], dense_results_list[i]]))

        return results_list

    def _bbox_forward_train(self, branch,seg_preds: torch.Tensor,
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

        keypoint_features = keypoint_features * seg_preds.sigmoid().max(
            dim=-1, keepdim=True).values

        bbox_results_list=[]
        rois_list=[]
        rois_list.append(bbox3d2roi([res.bboxes for res in sampling_results]))


        for stage in range(self.num_stage):
            if stage == self.num_stage -1:
                # 新的bbox存放在bbox_results中
                bbox_results = self._bbox_forward(True,branch,stage,keypoint_features, keypoints,rois_list[stage])
                if branch == "sparse":
                    bbox_targets = self.sparse_bbox_head[stage].get_targets(sampling_results, self.train_cfg)
                    loss_bbox = self.sparse_bbox_head[stage].loss(
                        True,
                        bbox_results['bbox_scores'],
                        bbox_results['bbox_reg'],
                        rois_list[stage],
                        *bbox_targets
                    )
                    loss_bbox = {f'sparse_{key}': value for key, value in loss_bbox.items()}
                elif branch == "dense":
                    bbox_targets = self.dense_bbox_head[stage].get_targets(sampling_results, self.train_cfg)
                    loss_bbox = self.dense_bbox_head[stage].loss(
                        True,
                        bbox_results['bbox_scores'],
                        bbox_results['bbox_reg'],
                        rois_list[stage],
                        *bbox_targets
                    )
                    loss_bbox = {f'dense_{key}': value for key, value in loss_bbox.items()}
            else:
                # 新的bbox存放在bbox_results中
                bbox_results = self._bbox_forward(False,branch,stage,keypoint_features, keypoints,rois_list[stage])
                if branch == "sparse":
                    bbox_targets = self.sparse_bbox_head[stage].get_targets(sampling_results, self.train_cfg)
                    loss_bbox = self.sparse_bbox_head[stage].loss(
                        False,
                        None,
                        bbox_results['bbox_reg'],
                        rois_list[stage],
                        *bbox_targets
                    )
                    loss_bbox = {f'sparse_{key}': value for key, value in loss_bbox.items()}
                elif branch == "dense":
                    bbox_targets = self.dense_bbox_head[stage].get_targets(sampling_results, self.train_cfg)
                    loss_bbox = self.dense_bbox_head[stage].loss(
                        False,
                        None,
                        bbox_results['bbox_reg'],
                        rois_list[stage],
                        *bbox_targets
                    )
                    loss_bbox = {f'dense_{key}': value for key, value in loss_bbox.items()}

            bbox_results.update(loss_bbox=loss_bbox)
            bbox_results_list.append(bbox_results)
            if branch == "sparse":
                bbox_result=self.sparse_bbox_head[stage].decode_from_rois(rois_list[stage],bbox_results['bbox_reg'])
            elif branch == "dense":
                bbox_result=self.dense_bbox_head[stage].decode_from_rois(rois_list[stage],bbox_results['bbox_reg'])
            # bbox_result=self.bbox_head[stage].decode_from_rois(rois_list[stage],bbox_results['bbox_reg'])
            rois_list.append(bbox3d2roi(bbox_result))

        return bbox_results_list


    def _bbox_forward(self,if_cls_pred ,branch,stage ,keypoint_features: torch.Tensor,
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
        if if_cls_pred:
            if branch == "sparse":
                pooled_keypoint_features = self.sparse_bbox_roi_extractor[stage](
                    keypoint_features, keypoints[..., 1:], keypoints[..., 0].int(),
                    rois)
                bbox_score, bbox_reg = self.sparse_bbox_head[stage](pooled_keypoint_features)
            elif branch == "dense":
                pooled_keypoint_features = self.dense_bbox_roi_extractor[stage](
                    keypoint_features, keypoints[..., 1:], keypoints[..., 0].int(),
                    rois)
                bbox_score, bbox_reg = self.dense_bbox_head[stage](pooled_keypoint_features)
            bbox_results = dict(bbox_scores=bbox_score, bbox_reg=bbox_reg)
        else:
            if branch == "sparse":
                pooled_keypoint_features = self.sparse_bbox_roi_extractor[stage](
                    keypoint_features, keypoints[..., 1:], keypoints[..., 0].int(),
                    rois)
                bbox_reg = self.sparse_bbox_head[stage](pooled_keypoint_features)
            elif branch == "dense":
                pooled_keypoint_features = self.dense_bbox_roi_extractor[stage](
                    keypoint_features, keypoints[..., 1:], keypoints[..., 0].int(),
                    rois)
                bbox_reg = self.dense_bbox_head[stage](pooled_keypoint_features)
            bbox_results = dict(bbox_reg=bbox_reg)
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
        """Count the number of points in each bounding box.

        Args:
            points (torch.Tensor): The list of input points.
            bboxes (torch.Tensor): The list of input bounding boxes.

        Returns:
            torch.Tensor: The list of number of points in each bounding box.
        """
        counts_list = []
        for points, bboxes in zip(points_list, bboxes_list):
            counts = self._ball_query_density(points, bboxes)
            counts_list.append(counts)
        return counts_list

    def _ball_query_density(self,points, bboxes):
        # Points: [N, 3]
        # Bboxes: [M, 7] where 7 = (x, y, z, dx, dy, dz, yaw)

        # Extract bbox center and dimensions
        bbox_center = bboxes[:, :3]  # [M, 3]
        bbox_dims = bboxes[:, 3:6]   # [M, 3]

        # Calculate the diagonal length for each bbox and the corresponding radius
        diagonal_lengths = torch.norm(bbox_dims, dim=1)  # [M]
        radii = diagonal_lengths / 2                     # [M]

        # Compute the distance between each point and each bbox center
        diff = points[:, None, :] - bbox_center[None, :, :]  # [N, M, 3]
        distances = torch.norm(diff, dim=2)                 # [N, M]

        # Check which points are within the ball's radius
        within_ball = distances <= radii  # [N, M]

        # Count the number of points within each ball
        point_counts = within_ball.sum(dim=0).float()  # [M]
        # Compute the volume of each ball (4/3 * pi * r^3)
        volumes = (4/3) * math.pi * (radii ** 3)  # [M]

        # Calculate the density (points count / volume)
        densities = point_counts / volumes  # [M]

        return densities

