# Copyright (c) OpenMMLab. All rights reserved.
import tempfile
from os import path as osp
from typing import Dict, List, Optional, Sequence, Tuple, Union

import mmengine
import numpy as np
import torch
from mmengine import load
from mmengine.evaluator import BaseMetric
from mmengine.logging import MMLogger, print_log

from mmdet3d.evaluation import kitti_eval
from mmdet3d.registry import METRICS
from mmdet3d.structures import (Box3DMode, CameraInstance3DBoxes,
                                LiDARInstance3DBoxes, points_cam2img)
import json


@METRICS.register_module()
class KittiMetric(BaseMetric):
    """Kitti evaluation metric.

    Args:
        ann_file (str): Annotation file path.
        metric (str or List[str]): Metrics to be evaluated. Defaults to 'bbox'.
        pcd_limit_range (List[float]): The range of point cloud used to filter
            invalid predicted boxes. Defaults to [0, -40, -3, 70.4, 40, 0.0].
        prefix (str, optional): The prefix that will be added in the metric
            names to disambiguate homonymous metrics of different evaluators.
            If prefix is not provided in the argument, self.default_prefix will
            be used instead. Defaults to None.
        pklfile_prefix (str, optional): The prefix of pkl files, including the
            file path and the prefix of filename, e.g., "a/b/prefix". If not
            specified, a temp file will be created. Defaults to None.
        default_cam_key (str): The default camera for lidar to camera
            conversion. By default, KITTI: 'CAM2', Waymo: 'CAM_FRONT'.
            Defaults to 'CAM2'.
        format_only (bool): Format the output results without perform
            evaluation. It is useful when you want to format the result to a
            specific format and submit it to the test server.
            Defaults to False.
        submission_prefix (str, optional): The prefix of submission data. If
            not specified, the submission data will not be generated.
            Defaults to None.
        collect_device (str): Device name used for collecting results from
            different ranks during distributed training. Must be 'cpu' or
            'gpu'. Defaults to 'cpu'.
        backend_args (dict, optional): Arguments to instantiate the
            corresponding backend. Defaults to None.
    """

    def __init__(self,
                 ann_file: str,
                 metric: Union[str, List[str]] = 'bbox',
                 pcd_limit_range: List[float] = [0, -40, -3, 70.4, 40, 0.0],
                 prefix: Optional[str] = None,
                 pklfile_prefix: Optional[str] = None,
                 default_cam_key: str = 'CAM2',
                 format_only: bool = False,
                 submission_prefix: Optional[str] = None,
                 collect_device: str = 'cpu',
                 backend_args: Optional[dict] = None) -> None:
        self.default_prefix = 'Kitti metric'
        super(KittiMetric, self).__init__(
            collect_device=collect_device, prefix=prefix)
        self.pcd_limit_range = pcd_limit_range
        self.ann_file = ann_file
        self.pklfile_prefix = pklfile_prefix
        self.format_only = format_only
        if self.format_only:
            assert submission_prefix is not None, 'submission_prefix must be '
            'not None when format_only is True, otherwise the result files '
            'will be saved to a temp directory which will be cleaned up at '
            'the end.'

        self.submission_prefix = submission_prefix
        self.default_cam_key = default_cam_key
        self.backend_args = backend_args

        allowed_metrics = ['bbox', 'img_bbox', 'mAP', 'LET_mAP']
        self.metrics = metric if isinstance(metric, list) else [metric]
        for metric in self.metrics:
            if metric not in allowed_metrics:
                raise KeyError("metric should be one of 'bbox', 'img_bbox', "
                               f'but got {metric}.')

    def convert_annos_to_kitti_annos(self, data_infos: dict) -> List[dict]:
        """Convert loading annotations to Kitti annotations.

        Args:
            data_infos (dict): Data infos including metainfo and annotations
                loaded from ann_file.

        Returns:
            List[dict]: List of Kitti annotations.
        """
        data_annos = data_infos['data_list']
        if not self.format_only:
            cat2label = data_infos['metainfo']['categories']
            label2cat = dict((v, k) for (k, v) in cat2label.items())
            assert 'instances' in data_annos[0]
            for i, annos in enumerate(data_annos):
                if len(annos['instances']) == 0:
                    kitti_annos = {
                        'name': np.array([]),
                        'truncated': np.array([]),
                        'occluded': np.array([]),
                        'alpha': np.array([]),
                        'bbox': np.zeros([0, 4]),
                        'dimensions': np.zeros([0, 3]),
                        'location': np.zeros([0, 3]),
                        'rotation_y': np.array([]),
                        'score': np.array([]),
                    }
                else:
                    kitti_annos = {
                        'name': [],
                        'truncated': [],
                        'occluded': [],
                        'alpha': [],
                        'bbox': [],
                        'location': [],
                        'dimensions': [],
                        'rotation_y': [],
                        'score': []
                    }
                    for instance in annos['instances']:
                        label = instance['bbox_label']
                        kitti_annos['name'].append(label2cat[label])
                        kitti_annos['truncated'].append(instance['truncated'])
                        kitti_annos['occluded'].append(instance['occluded'])
                        kitti_annos['alpha'].append(instance['alpha'])
                        kitti_annos['bbox'].append(instance['bbox'])
                        kitti_annos['location'].append(instance['bbox_3d'][:3])
                        kitti_annos['dimensions'].append(
                            instance['bbox_3d'][3:6])
                        kitti_annos['rotation_y'].append(
                            instance['bbox_3d'][6])
                        kitti_annos['score'].append(instance['score'])
                    for name in kitti_annos:
                        kitti_annos[name] = np.array(kitti_annos[name])
                data_annos[i]['kitti_annos'] = kitti_annos
        return data_annos

    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        """Process one batch of data samples and predictions.

        The processed results should be stored in ``self.results``, which will
        be used to compute the metrics when all batches have been processed.

        Args:
            data_batch (dict): A batch of data from the dataloader.
            data_samples (Sequence[dict]): A batch of outputs from the model.
        """

        for data_sample in data_samples:
            result = dict()
            pred_3d = data_sample['pred_instances_3d']
            pred_2d = data_sample['pred_instances']
            for attr_name in pred_3d:
                pred_3d[attr_name] = pred_3d[attr_name].to('cpu')
            result['pred_instances_3d'] = pred_3d
            for attr_name in pred_2d:
                pred_2d[attr_name] = pred_2d[attr_name].to('cpu')
            result['pred_instances'] = pred_2d
            sample_idx = data_sample['sample_idx']
            result['sample_idx'] = sample_idx
            self.results.append(result)

    @staticmethod
    def transform_annos_to_pred_coords(bboxes: List[List[float]]) -> List[List[float]]:
        """
        将多个 anno 检测框从原始坐标系（如 KITTI）转换为 pred 坐标系（如 Open3D 或 LiDAR）
        
        假设转换规则为：
        - [x, y, z] → [z, -x, -y]
        - [dx, dy, dz] 保持不变（如果方向是以 box 为准）
        - yaw → -(-yaw - π/2)
        """
        transformed = []
        for bbox in bboxes:
            if len(bbox) != 7:
                continue  # 忽略格式错误的框
            
            x, y, z, dx, dy, dz, yaw = bbox
            
            # 坐标轴变换
            new_x = z
            new_y = -x
            new_z = -y
            
            # yaw 角度转换（从绕Y轴转换为绕Z轴）
            new_yaw = -(-yaw - np.pi / 2)
            
            # 顺序调整为 [x, y, z, dx, dy, dz, yaw]
            transformed.append([new_x, new_y, new_z, dx, dz, dy, new_yaw])
        
        return transformed
    def save_results_for_vis(self, results:List[dict], save_json_path):
        save_list = []
        for result in results:
            sample_id = result['sample_idx']
            pcd_path = self.data_infos[sample_id]['lidar_points']['lidar_path']
            gt_anno = self.data_infos[sample_id]['kitti_annos']

            anno = np.concatenate((gt_anno['location'],gt_anno['dimensions'],np.expand_dims(gt_anno['rotation_y'],axis=1)),axis=1)
            car_anno = anno[np.where(gt_anno['name'] == 'Car')].tolist()
            car_anno = KittiMetric.transform_annos_to_pred_coords(car_anno)

            pred = result['pred_instances_3d']['bboxes_3d'].tensor.numpy()
            pred_label = result['pred_instances_3d']['labels_3d'].numpy()
            car_pred = pred[np.where(pred_label == 2)].tolist()

            cur_sample = {
                            'sample_id':sample_id,
                            'pcd_path': pcd_path,
                            'anno': car_anno,
                            'pred': car_pred
                          }
            save_list.append(cur_sample)
        
        with open(save_json_path, "w", encoding="utf-8") as file:
            json.dump(save_list, file, indent=4, ensure_ascii=False)
        
        print("save result for visualization finished.")
        

    def compute_metrics(self, results: List[dict]) -> Dict[str, float]:
        """Compute the metrics from processed results.

        Args:
            results (List[dict]): The processed results of the whole dataset.

        Returns:
            Dict[str, float]: The computed metrics. The keys are the names of
            the metrics, and the values are corresponding results.
        """
        logger: MMLogger = MMLogger.get_current_instance()
        self.classes = self.dataset_meta['classes']

        # load annotations
        pkl_infos = load(self.ann_file, backend_args=self.backend_args)
        self.data_infos = self.convert_annos_to_kitti_annos(pkl_infos)
        result_dict, tmp_dir = self.format_results(
            results,
            pklfile_prefix=self.pklfile_prefix,
            submission_prefix=self.submission_prefix,
            classes=self.classes)

        metric_dict = {}

        if self.format_only:
            logger.info(
                f'results are saved in {osp.dirname(self.submission_prefix)}')
            return metric_dict

        gt_annos = [
            self.data_infos[result['sample_idx']]['kitti_annos']
            for result in results
        ]
        self.save_results_for_vis(results,save_json_path="results_for_vis/result.json")
        
        for metric in self.metrics:
            ap_dict = self.kitti_evaluate(
                result_dict,
                gt_annos,
                metric=metric,
                logger=logger,
                classes=self.classes)
            for result in ap_dict:
                metric_dict[result] = ap_dict[result]

        if tmp_dir is not None:
            tmp_dir.cleanup()
        return metric_dict

    def kitti_evaluate(self,
                       results_dict: dict,
                       gt_annos: List[dict],
                       metric: Optional[str] = None,
                       classes: Optional[List[str]] = None,
                       logger: Optional[MMLogger] = None) -> Dict[str, float]:
        """Evaluation in KITTI protocol.

        Args:
            results_dict (dict): Formatted results of the dataset.
            gt_annos (List[dict]): Contain gt information of each sample.
            metric (str, optional): Metrics to be evaluated. Defaults to None.
            classes (List[str], optional): A list of class name.
                Defaults to None.
            logger (MMLogger, optional): Logger used for printing related
                information during evaluation. Defaults to None.

        Returns:
            Dict[str, float]: Results of each evaluation metric.
        """
        ap_dict = dict()
        for name in results_dict:
            if name == 'pred_instances' or metric == 'img_bbox':
                eval_types = ['bbox']
            else:
                eval_types = ['bbox', 'bev', '3d']
            ap_result_str, ap_dict_ = kitti_eval(
                gt_annos, results_dict[name], classes, eval_types=eval_types)
            for ap_type, ap in ap_dict_.items():
                ap_dict[f'{name}/{ap_type}'] = float(f'{ap:.4f}')

            print_log(f'Results of {name}:\n' + ap_result_str, logger=logger)
        return ap_dict

    def format_results(
        self,
        results: List[dict],
        pklfile_prefix: Optional[str] = None,
        submission_prefix: Optional[str] = None,
        classes: Optional[List[str]] = None
    ) -> Tuple[dict, Union[tempfile.TemporaryDirectory, None]]:
        """Format the results to pkl file.

        Args:
            results (List[dict]): Testing results of the dataset.
            pklfile_prefix (str, optional): The prefix of pkl files. It
                includes the file path and the prefix of filename, e.g.,
                "a/b/prefix". If not specified, a temp file will be created.
                Defaults to None.
            submission_prefix (str, optional): The prefix of submitted files.
                It includes the file path and the prefix of filename, e.g.,
                "a/b/prefix". If not specified, a temp file will be created.
                Defaults to None.
            classes (List[str], optional): A list of class name.
                Defaults to None.

        Returns:
            tuple: (result_dict, tmp_dir), result_dict is a dict containing the
            formatted result, tmp_dir is the temporal directory created for
            saving json files when jsonfile_prefix is not specified.
        """
        import pickle
        with open('pred.pkl', 'wb') as f:
            pickle.dump(results, f)

        if pklfile_prefix is None:
            tmp_dir = tempfile.TemporaryDirectory()
            pklfile_prefix = osp.join(tmp_dir.name, 'results')
        else:
            tmp_dir = None
        result_dict = dict()
        sample_idx_list = [result['sample_idx'] for result in results]
        for name in results[0]:
            if submission_prefix is not None:
                submission_prefix_ = osp.join(submission_prefix, name)
            else:
                submission_prefix_ = None
            if pklfile_prefix is not None:
                pklfile_prefix_ = osp.join(pklfile_prefix, name) + '.pkl'
            else:
                pklfile_prefix_ = None
            if 'pred_instances' in name and '3d' in name and name[
                    0] != '_' and results[0][name]:
                net_outputs = [result[name] for result in results]
                result_list_ = self.bbox2result_kitti(net_outputs,
                                                      sample_idx_list, classes,
                                                      pklfile_prefix_,
                                                      submission_prefix_)
                result_dict[name] = result_list_
            elif name == 'pred_instances' and name[0] != '_' and results[0][
                    name]:
                net_outputs = [result[name] for result in results]
                result_list_ = self.bbox2result_kitti2d(
                    net_outputs, sample_idx_list, classes, pklfile_prefix_,
                    submission_prefix_)
                result_dict[name] = result_list_
        return result_dict, tmp_dir

    def bbox2result_kitti(
            self,
            net_outputs: List[dict],
            sample_idx_list: List[int],
            class_names: List[str],
            pklfile_prefix: Optional[str] = None,
            submission_prefix: Optional[str] = None) -> List[dict]:
        """Convert 3D detection results to kitti format for evaluation and test
        submission.

        Args:
            net_outputs (List[dict]): List of dict storing the inferenced
                bounding boxes and scores.
            sample_idx_list (List[int]): List of input sample idx.
            class_names (List[str]): A list of class names.
            pklfile_prefix (str, optional): The prefix of pkl file.
                Defaults to None.
            submission_prefix (str, optional): The prefix of submission file.
                Defaults to None.

        Returns:
            List[dict]: A list of dictionaries with the kitti format.
        """
        assert len(net_outputs) == len(self.data_infos), \
            'invalid list length of network outputs'
        if submission_prefix is not None:
            mmengine.mkdir_or_exist(submission_prefix)

        det_annos = []
        print('\nConverting 3D prediction to KITTI format')
        for idx, pred_dicts in enumerate(
                mmengine.track_iter_progress(net_outputs)):
            sample_idx = sample_idx_list[idx]
            info = self.data_infos[sample_idx]
            # Here default used 'CAM2' to compute metric. If you want to
            # use another camera, please modify it.
            image_shape = (info['images'][self.default_cam_key]['height'],
                           info['images'][self.default_cam_key]['width'])
            box_dict = self.convert_valid_bboxes(pred_dicts, info)
            anno = {
                'name': [],
                'truncated': [],
                'occluded': [],
                'alpha': [],
                'bbox': [],
                'dimensions': [],
                'location': [],
                'rotation_y': [],
                'score': []
            }
            if len(box_dict['bbox']) > 0:
                box_2d_preds = box_dict['bbox']
                box_preds = box_dict['box3d_camera']
                scores = box_dict['scores']
                box_preds_lidar = box_dict['box3d_lidar']
                label_preds = box_dict['label_preds']
                pred_box_type_3d = box_dict['pred_box_type_3d']

                for box, box_lidar, bbox, score, label in zip(
                        box_preds, box_preds_lidar, box_2d_preds, scores,
                        label_preds):
                    bbox[2:] = np.minimum(bbox[2:], image_shape[::-1])
                    bbox[:2] = np.maximum(bbox[:2], [0, 0])
                    anno['name'].append(class_names[int(label)])
                    anno['truncated'].append(0.0)
                    anno['occluded'].append(0)
                    if pred_box_type_3d == CameraInstance3DBoxes:
                        anno['alpha'].append(-np.arctan2(box[0], box[2]) +
                                             box[6])
                    elif pred_box_type_3d == LiDARInstance3DBoxes:
                        anno['alpha'].append(
                            -np.arctan2(-box_lidar[1], box_lidar[0]) + box[6])
                    anno['bbox'].append(bbox)
                    anno['dimensions'].append(box[3:6])
                    anno['location'].append(box[:3])
                    anno['rotation_y'].append(box[6])
                    anno['score'].append(score)

                anno = {k: np.stack(v) for k, v in anno.items()}
            else:
                anno = {
                    'name': np.array([]),
                    'truncated': np.array([]),
                    'occluded': np.array([]),
                    'alpha': np.array([]),
                    'bbox': np.zeros([0, 4]),
                    'dimensions': np.zeros([0, 3]),
                    'location': np.zeros([0, 3]),
                    'rotation_y': np.array([]),
                    'score': np.array([]),
                }

            if submission_prefix is not None:
                curr_file = f'{submission_prefix}/{sample_idx:06d}.txt'
                with open(curr_file, 'w') as f:
                    bbox = anno['bbox']
                    loc = anno['location']
                    dims = anno['dimensions']  # lhw -> hwl

                    for idx in range(len(bbox)):
                        print(
                            '{} -1 -1 {:.4f} {:.4f} {:.4f} {:.4f} '
                            '{:.4f} {:.4f} {:.4f} '
                            '{:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f}'.format(
                                anno['name'][idx], anno['alpha'][idx],
                                bbox[idx][0], bbox[idx][1], bbox[idx][2],
                                bbox[idx][3], dims[idx][1], dims[idx][2],
                                dims[idx][0], loc[idx][0], loc[idx][1],
                                loc[idx][2], anno['rotation_y'][idx],
                                anno['score'][idx]),
                            file=f)

            anno['sample_idx'] = np.array(
                [sample_idx] * len(anno['score']), dtype=np.int64)

            det_annos.append(anno)

        if pklfile_prefix is not None:
            if not pklfile_prefix.endswith(('.pkl', '.pickle')):
                out = f'{pklfile_prefix}.pkl'
            else:
                out = pklfile_prefix
            mmengine.dump(det_annos, out)
            print(f'Result is saved to {out}.')

        return det_annos

    def bbox2result_kitti2d(
            self,
            net_outputs: List[dict],
            sample_idx_list: List[int],
            class_names: List[str],
            pklfile_prefix: Optional[str] = None,
            submission_prefix: Optional[str] = None) -> List[dict]:
        """Convert 2D detection results to kitti format for evaluation and test
        submission.

        Args:
            net_outputs (List[dict]): List of dict storing the inferenced
                bounding boxes and scores.
            sample_idx_list (List[int]): List of input sample idx.
            class_names (List[str]): A list of class names.
            pklfile_prefix (str, optional): The prefix of pkl file.
                Defaults to None.
            submission_prefix (str, optional): The prefix of submission file.
                Defaults to None.

        Returns:
            List[dict]: A list of dictionaries with the kitti format.
        """
        assert len(net_outputs) == len(self.data_infos), \
            'invalid list length of network outputs'
        det_annos = []
        print('\nConverting 2D prediction to KITTI format')
        for i, bboxes_per_sample in enumerate(
                mmengine.track_iter_progress(net_outputs)):
            anno = dict(
                name=[],
                truncated=[],
                occluded=[],
                alpha=[],
                bbox=[],
                dimensions=[],
                location=[],
                rotation_y=[],
                score=[])
            sample_idx = sample_idx_list[i]

            num_example = 0
            bbox = bboxes_per_sample['bboxes']
            for i in range(bbox.shape[0]):
                anno['name'].append(class_names[int(
                    bboxes_per_sample['labels'][i])])
                anno['truncated'].append(0.0)
                anno['occluded'].append(0)
                anno['alpha'].append(0.0)
                anno['bbox'].append(bbox[i, :4])
                # set dimensions (height, width, length) to zero
                anno['dimensions'].append(
                    np.zeros(shape=[3], dtype=np.float32))
                # set the 3D translation to (-1000, -1000, -1000)
                anno['location'].append(
                    np.ones(shape=[3], dtype=np.float32) * (-1000.0))
                anno['rotation_y'].append(0.0)
                anno['score'].append(bboxes_per_sample['scores'][i])
                num_example += 1

            if num_example == 0:
                anno = dict(
                    name=np.array([]),
                    truncated=np.array([]),
                    occluded=np.array([]),
                    alpha=np.array([]),
                    bbox=np.zeros([0, 4]),
                    dimensions=np.zeros([0, 3]),
                    location=np.zeros([0, 3]),
                    rotation_y=np.array([]),
                    score=np.array([]),
                )
            else:
                anno = {k: np.stack(v) for k, v in anno.items()}

            anno['sample_idx'] = np.array(
                [sample_idx] * num_example, dtype=np.int64)
            det_annos.append(anno)

        if pklfile_prefix is not None:
            if not pklfile_prefix.endswith(('.pkl', '.pickle')):
                out = f'{pklfile_prefix}.pkl'
            else:
                out = pklfile_prefix
            mmengine.dump(det_annos, out)
            print(f'Result is saved to {out}.')

        if submission_prefix is not None:
            # save file in submission format
            mmengine.mkdir_or_exist(submission_prefix)
            print(f'Saving KITTI submission to {submission_prefix}')
            for i, anno in enumerate(det_annos):
                sample_idx = sample_idx_list[i]
                cur_det_file = f'{submission_prefix}/{sample_idx:06d}.txt'
                with open(cur_det_file, 'w') as f:
                    bbox = anno['bbox']
                    loc = anno['location']
                    dims = anno['dimensions'][::-1]  # lhw -> hwl
                    for idx in range(len(bbox)):
                        print(
                            '{} -1 -1 {:4f} {:4f} {:4f} {:4f} {:4f} {:4f} '
                            '{:4f} {:4f} {:4f} {:4f} {:4f} {:4f} {:4f}'.format(
                                anno['name'][idx],
                                anno['alpha'][idx],
                                *bbox[idx],  # 4 float
                                *dims[idx],  # 3 float
                                *loc[idx],  # 3 float
                                anno['rotation_y'][idx],
                                anno['score'][idx]),
                            file=f,
                        )
            print(f'Result is saved to {submission_prefix}')

        return det_annos

    def convert_valid_bboxes(self, box_dict: dict, info: dict) -> dict:
        """Convert the predicted boxes into valid ones.

        Args:
            box_dict (dict): Box dictionaries to be converted.

                - bboxes_3d (:obj:`BaseInstance3DBoxes`): 3D bounding boxes.
                - scores_3d (Tensor): Scores of boxes.
                - labels_3d (Tensor): Class labels of boxes.
            info (dict): Data info.

        Returns:
            dict: Valid predicted boxes.

            - bbox (np.ndarray): 2D bounding boxes.
            - box3d_camera (np.ndarray): 3D bounding boxes in
              camera coordinate.
            - box3d_lidar (np.ndarray): 3D bounding boxes in
              LiDAR coordinate.
            - scores (np.ndarray): Scores of boxes.
            - label_preds (np.ndarray): Class label predictions.
            - sample_idx (int): Sample index.
        """
        # TODO: refactor this function
        box_preds = box_dict['bboxes_3d']
        scores = box_dict['scores_3d']
        labels = box_dict['labels_3d']
        sample_idx = info['sample_idx']
        box_preds.limit_yaw(offset=0.5, period=np.pi * 2)

        if len(box_preds) == 0:
            return dict(
                bbox=np.zeros([0, 4]),
                box3d_camera=np.zeros([0, 7]),
                box3d_lidar=np.zeros([0, 7]),
                scores=np.zeros([0]),
                label_preds=np.zeros([0, 4]),
                sample_idx=sample_idx)
        # Here default used 'CAM2' to compute metric. If you want to
        # use another camera, please modify it.
        lidar2cam = np.array(
            info['images'][self.default_cam_key]['lidar2cam']).astype(
                np.float32)
        P2 = np.array(info['images'][self.default_cam_key]['cam2img']).astype(
            np.float32)
        img_shape = (info['images'][self.default_cam_key]['height'],
                     info['images'][self.default_cam_key]['width'])
        P2 = box_preds.tensor.new_tensor(P2)

        if isinstance(box_preds, LiDARInstance3DBoxes):
            box_preds_camera = box_preds.convert_to(Box3DMode.CAM, lidar2cam)
            box_preds_lidar = box_preds
        elif isinstance(box_preds, CameraInstance3DBoxes):
            box_preds_camera = box_preds
            box_preds_lidar = box_preds.convert_to(Box3DMode.LIDAR,
                                                   np.linalg.inv(lidar2cam))

        box_corners = box_preds_camera.corners
        box_corners_in_image = points_cam2img(box_corners, P2)
        # box_corners_in_image: [N, 8, 2]
        minxy = torch.min(box_corners_in_image, dim=1)[0]
        maxxy = torch.max(box_corners_in_image, dim=1)[0]
        box_2d_preds = torch.cat([minxy, maxxy], dim=1)
        # Post-processing
        # check box_preds_camera
        image_shape = box_preds.tensor.new_tensor(img_shape)
        valid_cam_inds = ((box_2d_preds[:, 0] < image_shape[1]) &
                          (box_2d_preds[:, 1] < image_shape[0]) &
                          (box_2d_preds[:, 2] > 0) & (box_2d_preds[:, 3] > 0))
        # check box_preds_lidar
        if isinstance(box_preds, LiDARInstance3DBoxes):
            limit_range = box_preds.tensor.new_tensor(self.pcd_limit_range)
            valid_pcd_inds = ((box_preds_lidar.center > limit_range[:3]) &
                              (box_preds_lidar.center < limit_range[3:]))
            valid_inds = valid_cam_inds & valid_pcd_inds.all(-1)
        else:
            valid_inds = valid_cam_inds

        if valid_inds.sum() > 0:
            return dict(
                bbox=box_2d_preds[valid_inds, :].numpy(),
                pred_box_type_3d=type(box_preds),
                box3d_camera=box_preds_camera[valid_inds].numpy(),
                box3d_lidar=box_preds_lidar[valid_inds].numpy(),
                scores=scores[valid_inds].numpy(),
                label_preds=labels[valid_inds].numpy(),
                sample_idx=sample_idx)
        else:
            return dict(
                bbox=np.zeros([0, 4]),
                pred_box_type_3d=type(box_preds),
                box3d_camera=np.zeros([0, 7]),
                box3d_lidar=np.zeros([0, 7]),
                scores=np.zeros([0]),
                label_preds=np.zeros([0]),
                sample_idx=sample_idx)
