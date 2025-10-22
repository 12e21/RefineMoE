# Copyright (c) OpenMMLab. All rights reserved.
from mmdet.models.roi_heads.bbox_heads import (BBoxHead, ConvFCBBoxHead,
                                               DoubleConvFCBBoxHead,
                                               Shared2FCBBoxHead,
                                               Shared4Conv1FCBBoxHead)

from .h3d_bbox_head import H3DBboxHead
from .parta2_bbox_head import PartA2BboxHead
from .point_rcnn_bbox_head import PointRCNNBboxHead
from .pv_rcnn_bbox_head import PVRCNNBBoxHead
from .decouple_refine_bbox_head import DecoupleRefineBBoxHead
from .light_sparsity_bbox_head import LightSparsityBBoxHead
from .soft_sparsity_bbox_head import SoftSparsityBBoxHead
from .multi_stage_multi_branch_bbox_head import MultiStageMultiBranchBBoxHead
from .task_decouple_all_bbox_head import TaskDecoupleAllBBoxHead
from .sample_decouple_separate_bbox_head import SampleDecoupleSeparateBBoxHead

__all__ = [
    'BBoxHead', 'ConvFCBBoxHead', 'Shared2FCBBoxHead',
    'Shared4Conv1FCBBoxHead', 'DoubleConvFCBBoxHead', 'PartA2BboxHead',
    'H3DBboxHead', 'PointRCNNBboxHead', 'PVRCNNBBoxHead', 'DecoupleRefineBBoxHead',
    'LightSparsityBBoxHead', 'SoftSparsityBBoxHead', 'MultiStageMultiBranchBBoxHead',
    'TaskDecoupleAllBBoxHead', 'SampleDecoupleSeparateBBoxHead'
]
