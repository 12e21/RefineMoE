# Copyright (c) OpenMMLab. All rights reserved.
import gc
import io as sysio

import numba
import numpy as np

@numba.jit(nopython=True)
def image_box_overlap(boxes, query_boxes, criterion=-1):
    N = boxes.shape[0]
    K = query_boxes.shape[0]
    overlaps = np.zeros((N, K), dtype=boxes.dtype)
    for k in range(K):
        qbox_area = ((query_boxes[k, 2] - query_boxes[k, 0]) *
                     (query_boxes[k, 3] - query_boxes[k, 1]))
        for n in range(N):
            iw = (
                min(boxes[n, 2], query_boxes[k, 2]) -
                max(boxes[n, 0], query_boxes[k, 0]))
            if iw > 0:
                ih = (
                    min(boxes[n, 3], query_boxes[k, 3]) -
                    max(boxes[n, 1], query_boxes[k, 1]))
                if ih > 0:
                    if criterion == -1:
                        ua = ((boxes[n, 2] - boxes[n, 0]) *
                              (boxes[n, 3] - boxes[n, 1]) + qbox_area -
                              iw * ih)
                    elif criterion == 0:
                        ua = ((boxes[n, 2] - boxes[n, 0]) *
                              (boxes[n, 3] - boxes[n, 1]))
                    elif criterion == 1:
                        ua = qbox_area
                    else:
                        ua = 1.0
                    overlaps[n, k] = iw * ih / ua
    return overlaps

@numba.jit
def get_thresholds(scores: np.ndarray, num_gt, num_sample_pts=41):
    scores.sort()
    scores = scores[::-1]
    current_recall = 0
    thresholds = []
    for i, score in enumerate(scores):
        l_recall = (i + 1) / num_gt
        if i < (len(scores) - 1):
            r_recall = (i + 2) / num_gt
        else:
            r_recall = l_recall
        if (((r_recall - current_recall) < (current_recall - l_recall))
                and (i < (len(scores) - 1))):
            continue
        # recall = l_recall
        thresholds.append(score)
        current_recall += 1 / (num_sample_pts - 1.0)
    return thresholds


def clean_data_by_pts(gt_anno, dt_anno, current_class, pts_range):
    """
    Clean data based on points in bbox range instead of difficulty.
    
    Args:
        gt_anno: Ground truth annotations
        dt_anno: Detection annotations
        current_class: Current class index (0: car, 1: pedestrian, 2: cyclist)
        pts_range: [min_pts, max_pts] range for filtering
    """
    CLASS_NAMES = ['car', 'pedestrian', 'cyclist']
    dc_bboxes, ignored_gt, ignored_dt = [], [], []
    current_cls_name = CLASS_NAMES[current_class].lower()
    num_gt = len(gt_anno['name'])
    num_dt = len(dt_anno['name'])
    num_valid_gt = 0
    
    for i in range(num_gt):
        bbox = gt_anno['bbox'][i]
        gt_name = gt_anno['name'][i].lower()
        num_pts = gt_anno['num_lidar_pts'][i] if 'num_lidar_pts' in gt_anno else 0
        
        valid_class = -1
        if (gt_name == current_cls_name):
            valid_class = 1
        elif (current_cls_name == 'Pedestrian'.lower()
              and 'Person_sitting'.lower() == gt_name):
            valid_class = 0
        elif (current_cls_name == 'Car'.lower() and 'Van'.lower() == gt_name):
            valid_class = 0
        else:
            valid_class = -1
        
        # Filter by points in bbox range instead of difficulty
        ignore = False
        if num_pts < pts_range[0] or num_pts > pts_range[1]:
            ignore = True
        
        if valid_class == 1 and not ignore:
            ignored_gt.append(0)
            num_valid_gt += 1
        elif (valid_class == 0 or (ignore and (valid_class == 1))):
            ignored_gt.append(1)
        else:
            ignored_gt.append(-1)
    
    # Handle DontCare boxes
    for i in range(num_gt):
        if gt_anno['name'][i] == 'DontCare':
            dc_bboxes.append(gt_anno['bbox'][i])
    
    # Process detections (no filtering by points for detections)
    for i in range(num_dt):
        if (dt_anno['name'][i].lower() == current_cls_name):
            valid_class = 1
        else:
            valid_class = -1
        
        if valid_class == 1:
            ignored_dt.append(0)
        else:
            ignored_dt.append(-1)

    return num_valid_gt, ignored_gt, ignored_dt, dc_bboxes


def bev_box_overlap(boxes, qboxes, criterion=-1):
    from .rotate_iou import rotate_iou_gpu_eval
    riou = rotate_iou_gpu_eval(boxes, qboxes, criterion)
    return riou


@numba.jit(nopython=True, parallel=True)
def d3_box_overlap_kernel(boxes, qboxes, rinc, criterion=-1):
    # ONLY support overlap in CAMERA, not lidar.
    N, K = boxes.shape[0], qboxes.shape[0]
    for i in numba.prange(N):
        for j in numba.prange(K):
            if rinc[i, j] > 0:
                iw = (
                    min(boxes[i, 1], qboxes[j, 1]) -
                    max(boxes[i, 1] - boxes[i, 4],
                        qboxes[j, 1] - qboxes[j, 4]))

                if iw > 0:
                    area1 = boxes[i, 3] * boxes[i, 4] * boxes[i, 5]
                    area2 = qboxes[j, 3] * qboxes[j, 4] * qboxes[j, 5]
                    inc = iw * rinc[i, j]
                    if criterion == -1:
                        ua = (area1 + area2 - inc)
                    elif criterion == 0:
                        ua = area1
                    elif criterion == 1:
                        ua = area2
                    else:
                        ua = inc
                    rinc[i, j] = inc / ua
                else:
                    rinc[i, j] = 0.0


def d3_box_overlap(boxes, qboxes, criterion=-1):
    from .rotate_iou import rotate_iou_gpu_eval
    rinc = rotate_iou_gpu_eval(boxes[:, [0, 2, 3, 5, 6]],
                               qboxes[:, [0, 2, 3, 5, 6]], 2)
    d3_box_overlap_kernel(boxes, qboxes, rinc, criterion)
    return rinc


@numba.jit(nopython=True)
def compute_statistics_jit(overlaps,
                           gt_datas,
                           dt_datas,
                           ignored_gt,
                           ignored_det,
                           dc_bboxes,
                           metric,
                           min_overlap,
                           thresh=0,
                           compute_fp=False,
                           compute_aos=False):

    det_size = dt_datas.shape[0]
    gt_size = gt_datas.shape[0]
    dt_scores = dt_datas[:, -1]
    dt_alphas = dt_datas[:, 4]
    gt_alphas = gt_datas[:, 4]
    dt_bboxes = dt_datas[:, :4]

    assigned_detection = [False] * det_size
    ignored_threshold = [False] * det_size
    if compute_fp:
        for i in range(det_size):
            if (dt_scores[i] < thresh):
                ignored_threshold[i] = True
    NO_DETECTION = -10000000
    tp, fp, fn, similarity = 0, 0, 0, 0
    thresholds = np.zeros((gt_size, ))
    thresh_idx = 0
    delta = np.zeros((gt_size, ))
    delta_idx = 0
    for i in range(gt_size):
        if ignored_gt[i] == -1:
            continue
        det_idx = -1
        valid_detection = NO_DETECTION
        max_overlap = 0
        assigned_ignored_det = False

        for j in range(det_size):
            if (ignored_det[j] == -1):
                continue
            if (assigned_detection[j]):
                continue
            if (ignored_threshold[j]):
                continue
            overlap = overlaps[j, i]
            dt_score = dt_scores[j]
            if (not compute_fp and (overlap > min_overlap)
                    and dt_score > valid_detection):
                det_idx = j
                valid_detection = dt_score
            elif (compute_fp and (overlap > min_overlap)
                  and (overlap > max_overlap or assigned_ignored_det)
                  and ignored_det[j] == 0):
                max_overlap = overlap
                det_idx = j
                valid_detection = 1
                assigned_ignored_det = False
            elif (compute_fp and (overlap > min_overlap)
                  and (valid_detection == NO_DETECTION)
                  and ignored_det[j] == 1):
                det_idx = j
                valid_detection = 1
                assigned_ignored_det = True

        if (valid_detection == NO_DETECTION) and ignored_gt[i] == 0:
            fn += 1
        elif ((valid_detection != NO_DETECTION)
              and (ignored_gt[i] == 1 or ignored_det[det_idx] == 1)):
            assigned_detection[det_idx] = True
        elif valid_detection != NO_DETECTION:
            tp += 1
            thresholds[thresh_idx] = dt_scores[det_idx]
            thresh_idx += 1
            if compute_aos:
                delta[delta_idx] = gt_alphas[i] - dt_alphas[det_idx]
                delta_idx += 1

            assigned_detection[det_idx] = True
    if compute_fp:
        for i in range(det_size):
            if (not (assigned_detection[i] or ignored_det[i] == -1
                     or ignored_det[i] == 1 or ignored_threshold[i])):
                fp += 1
        nstuff = 0
        if metric == 0:
            overlaps_dt_dc = image_box_overlap(dt_bboxes, dc_bboxes, 0)
            for i in range(dc_bboxes.shape[0]):
                for j in range(det_size):
                    if (assigned_detection[j]):
                        continue
                    if (ignored_det[j] == -1 or ignored_det[j] == 1):
                        continue
                    if (ignored_threshold[j]):
                        continue
                    if overlaps_dt_dc[j, i] > min_overlap:
                        assigned_detection[j] = True
                        nstuff += 1
        fp -= nstuff
        if compute_aos:
            tmp = np.zeros((fp + delta_idx, ))
            for i in range(delta_idx):
                tmp[i + fp] = (1.0 + np.cos(delta[i])) / 2.0
            if tp > 0 or fp > 0:
                similarity = np.sum(tmp)
            else:
                similarity = -1
    return tp, fp, fn, similarity, thresholds[:thresh_idx]


def get_split_parts(num, num_part):
    if num % num_part == 0:
        same_part = num // num_part
        return [same_part] * num_part
    else:
        same_part = num // (num_part - 1)
        remain_num = num % (num_part - 1)
        return [same_part] * (num_part - 1) + [remain_num]


@numba.jit(nopython=True)
def fused_compute_statistics(overlaps,
                             pr,
                             gt_nums,
                             dt_nums,
                             dc_nums,
                             gt_datas,
                             dt_datas,
                             dontcares,
                             ignored_gts,
                             ignored_dets,
                             metric,
                             min_overlap,
                             thresholds,
                             compute_aos=False):
    gt_num = 0
    dt_num = 0
    dc_num = 0
    for i in range(gt_nums.shape[0]):
        for t, thresh in enumerate(thresholds):
            overlap = overlaps[dt_num:dt_num + dt_nums[i],
                               gt_num:gt_num + gt_nums[i]]

            gt_data = gt_datas[gt_num:gt_num + gt_nums[i]]
            dt_data = dt_datas[dt_num:dt_num + dt_nums[i]]
            ignored_gt = ignored_gts[gt_num:gt_num + gt_nums[i]]
            ignored_det = ignored_dets[dt_num:dt_num + dt_nums[i]]
            dontcare = dontcares[dc_num:dc_num + dc_nums[i]]
            tp, fp, fn, similarity, _ = compute_statistics_jit(
                overlap,
                gt_data,
                dt_data,
                ignored_gt,
                ignored_det,
                dontcare,
                metric,
                min_overlap=min_overlap,
                thresh=thresh,
                compute_fp=True,
                compute_aos=compute_aos)
            pr[t, 0] += tp
            pr[t, 1] += fp
            pr[t, 2] += fn
            if similarity != -1:
                pr[t, 3] += similarity
        gt_num += gt_nums[i]
        dt_num += dt_nums[i]
        dc_num += dc_nums[i]


def calculate_iou_partly(dt_annos, gt_annos, metric, num_parts=50):
    """Fast iou algorithm for 3D metric only."""
    assert len(dt_annos) == len(gt_annos)
    total_gt_num = np.stack([len(a['name']) for a in gt_annos], 0)
    total_dt_num = np.stack([len(a['name']) for a in dt_annos], 0)
    num_examples = len(dt_annos)
    split_parts = get_split_parts(num_examples, num_parts)
    parted_overlaps = []
    example_idx = 0

    for num_part in split_parts:
        dt_annos_part = dt_annos[example_idx:example_idx + num_part]
        gt_annos_part = gt_annos[example_idx:example_idx + num_part]
        
        # Only support 3D metric (metric == 2)
        if metric == 2:
            loc = np.concatenate([a['location'] for a in dt_annos_part], 0)
            dims = np.concatenate([a['dimensions'] for a in dt_annos_part], 0)
            rots = np.concatenate([a['rotation_y'] for a in dt_annos_part], 0)
            dt_boxes = np.concatenate([loc, dims, rots[..., np.newaxis]],
                                      axis=1)
            loc = np.concatenate([a['location'] for a in gt_annos_part], 0)
            dims = np.concatenate([a['dimensions'] for a in gt_annos_part], 0)
            rots = np.concatenate([a['rotation_y'] for a in gt_annos_part], 0)
            gt_boxes = np.concatenate([loc, dims, rots[..., np.newaxis]],
                                      axis=1)
            overlap_part = d3_box_overlap(dt_boxes,
                                          gt_boxes).astype(np.float64)
        else:
            raise ValueError('Only 3D metric is supported')
        parted_overlaps.append(overlap_part)
        example_idx += num_part
    
    overlaps = []
    example_idx = 0
    for j, num_part in enumerate(split_parts):
        gt_num_idx, dt_num_idx = 0, 0
        for i in range(num_part):
            gt_box_num = total_gt_num[example_idx + i]
            dt_box_num = total_dt_num[example_idx + i]
            overlaps.append(
                parted_overlaps[j][dt_num_idx:dt_num_idx + dt_box_num,
                                   gt_num_idx:gt_num_idx + gt_box_num])
            gt_num_idx += gt_box_num
            dt_num_idx += dt_box_num
        example_idx += num_part

    return overlaps, parted_overlaps, total_dt_num, total_gt_num


def _prepare_data_by_pts(gt_annos, dt_annos, current_class, pts_range):
    """Prepare data filtered by points in bbox range."""
    gt_datas_list = []
    dt_datas_list = []
    total_dc_num = []
    ignored_gts, ignored_dets, dontcares = [], [], []
    total_num_valid_gt = 0
    for i in range(len(gt_annos)):
        rets = clean_data_by_pts(gt_annos[i], dt_annos[i], current_class, pts_range)
        num_valid_gt, ignored_gt, ignored_det, dc_bboxes = rets
        ignored_gts.append(np.array(ignored_gt, dtype=np.int64))
        ignored_dets.append(np.array(ignored_det, dtype=np.int64))
        if len(dc_bboxes) == 0:
            dc_bboxes = np.zeros((0, 4)).astype(np.float64)
        else:
            dc_bboxes = np.stack(dc_bboxes, 0).astype(np.float64)
        total_dc_num.append(dc_bboxes.shape[0])
        dontcares.append(dc_bboxes)
        total_num_valid_gt += num_valid_gt
        gt_datas = np.concatenate(
            [gt_annos[i]['bbox'], gt_annos[i]['alpha'][..., np.newaxis]], 1)
        dt_datas = np.concatenate([
            dt_annos[i]['bbox'], dt_annos[i]['alpha'][..., np.newaxis],
            dt_annos[i]['score'][..., np.newaxis]
        ], 1)
        gt_datas_list.append(gt_datas)
        dt_datas_list.append(dt_datas)
    total_dc_num = np.stack(total_dc_num, axis=0)
    return (gt_datas_list, dt_datas_list, ignored_gts, ignored_dets, dontcares,
            total_dc_num, total_num_valid_gt)


def eval_class_3d_ap40_by_pts(gt_annos,
                              dt_annos,
                              current_classes,
                              pts_ranges,
                              min_overlaps,
                              num_parts=200):
    """
    Evaluate 3D AP40 for different classes and points ranges.
    
    Args:
        gt_annos: Ground truth annotations
        dt_annos: Detection annotations  
        current_classes: List of class indices to evaluate
        pts_ranges: List of [min_pts, max_pts] ranges
        min_overlaps: Overlap thresholds [num_overlap, metric, class]
        num_parts: Number of parts for parallel processing
    """
    assert len(gt_annos) == len(dt_annos)
    num_examples = len(gt_annos)
    if num_examples < num_parts:
        num_parts = num_examples
    split_parts = get_split_parts(num_examples, num_parts)

    # Only calculate 3D IoU (metric = 2)
    metric = 2
    rets = calculate_iou_partly(dt_annos, gt_annos, metric, num_parts)
    overlaps, parted_overlaps, total_dt_num, total_gt_num = rets

    N_SAMPLE_PTS = 41
    num_minoverlap = len(min_overlaps)
    num_class = len(current_classes)
    num_pts_range = len(pts_ranges)
    
    # Only store precision for AP40 calculation
    precision = np.zeros([num_class, num_pts_range, num_minoverlap, N_SAMPLE_PTS])
    
    for m, current_class in enumerate(current_classes):
        for idx_r, pts_range in enumerate(pts_ranges):
            rets = _prepare_data_by_pts(gt_annos, dt_annos, current_class, pts_range)
            (gt_datas_list, dt_datas_list, ignored_gts, ignored_dets,
             dontcares, total_dc_num, total_num_valid_gt) = rets
            for k, min_overlap in enumerate(min_overlaps[:, metric, m]):
                thresholdss = []
                for i in range(len(gt_annos)):
                    rets = compute_statistics_jit(
                        overlaps[i],
                        gt_datas_list[i],
                        dt_datas_list[i],
                        ignored_gts[i],
                        ignored_dets[i],
                        dontcares[i],
                        metric,
                        min_overlap=min_overlap,
                        thresh=0.0,
                        compute_fp=False)
                    tp, fp, fn, similarity, thresholds = rets
                    thresholdss += thresholds.tolist()
                thresholdss = np.array(thresholdss)
                thresholds = get_thresholds(thresholdss, total_num_valid_gt)
                thresholds = np.array(thresholds)
                pr = np.zeros([len(thresholds), 4])
                idx = 0
                for j, num_part in enumerate(split_parts):
                    gt_datas_part = np.concatenate(
                        gt_datas_list[idx:idx + num_part], 0)
                    dt_datas_part = np.concatenate(
                        dt_datas_list[idx:idx + num_part], 0)
                    dc_datas_part = np.concatenate(
                        dontcares[idx:idx + num_part], 0)
                    ignored_dets_part = np.concatenate(
                        ignored_dets[idx:idx + num_part], 0)
                    ignored_gts_part = np.concatenate(
                        ignored_gts[idx:idx + num_part], 0)
                    fused_compute_statistics(
                        parted_overlaps[j],
                        pr,
                        total_gt_num[idx:idx + num_part],
                        total_dt_num[idx:idx + num_part],
                        total_dc_num[idx:idx + num_part],
                        gt_datas_part,
                        dt_datas_part,
                        dc_datas_part,
                        ignored_gts_part,
                        ignored_dets_part,
                        metric,
                        min_overlap=min_overlap,
                        thresholds=thresholds,
                        compute_aos=False)
                    idx += num_part
                for i in range(len(thresholds)):
                    precision[m, idx_r, k, i] = pr[i, 0] / (pr[i, 0] + pr[i, 1])
                for i in range(len(thresholds)):
                    precision[m, idx_r, k, i] = np.max(
                        precision[m, idx_r, k, i:], axis=-1)
    
    ret_dict = {'precision': precision}

    # Clean temp variables
    del overlaps
    del parted_overlaps
    gc.collect()
    return ret_dict


def get_mAP40(prec):
    """Calculate mAP40 (average precision over 40 recall points)."""
    sums = 0
    for i in range(1, prec.shape[-1]):
        sums = sums + prec[..., i]
    return sums / 40 * 100


def print_str(value, *arg, sstream=None):
    if sstream is None:
        sstream = sysio.StringIO()
    sstream.truncate(0)
    sstream.seek(0)
    print(value, *arg, file=sstream)
    return sstream.getvalue()


def pts_eval(gt_annos,
             dt_annos,
             current_classes,
             pts_ranges=None):
    """
    KITTI 3D AP40 evaluation grouped by points in bbox.
    
    Args:
        gt_annos: Ground truth annotations (must contain 'num_lidar_pts' key)
        dt_annos: Detection annotations
        current_classes: Classes to evaluate (e.g., ['Car', 'Pedestrian', 'Cyclist'])
        pts_ranges: List of [min_pts, max_pts] ranges. If None, use default ranges.
        
    Returns:
        tuple: (result_string, result_dict)
    """
    # Default points ranges if not provided
    if pts_ranges is None:
        # pts_ranges = [
        #     [0, 10],      # Very sparse
        #     [11, 50],     # Sparse  
        #     [51, 200],    # Medium
        #     [201, 1000],  # Dense
        #     [1001, 10000] # Very dense
        # ]
        # pts_ranges = [
        #     [0, 100],     
        #     [101, 200],     # Sparse  
        #     [201, 1000],  # Dense
        #     [1001, 10000] # Very dense
        # ]
        # pts_ranges = [
        #     [0,10],
        #     [11,10000]
        # ]
        pts_ranges = [
            [0,10000],
        ]
    
    # Use overlap_0_7 thresholds as specified
    overlap_0_7 = np.array([[0.7, 0.5, 0.5, 0.7, 0.5], 
                            [0.7, 0.5, 0.5, 0.7, 0.5],
                            [0.7, 0.5, 0.5, 0.7, 0.5]])
    min_overlaps = overlap_0_7[np.newaxis, ...]  # Shape: [1, 3, 5]
    
    class_to_name = {
        0: 'Car',
        1: 'Pedestrian', 
        2: 'Cyclist',
        3: 'Van',
        4: 'Person_sitting',
    }
    name_to_class = {v: n for n, v in class_to_name.items()}
    
    if not isinstance(current_classes, (list, tuple)):
        current_classes = [current_classes]
    current_classes_int = []
    for curcls in current_classes:
        if isinstance(curcls, str):
            current_classes_int.append(name_to_class[curcls])
        else:
            current_classes_int.append(curcls)
    current_classes = current_classes_int
    min_overlaps = min_overlaps[:, :, current_classes]
    
    # Evaluate 3D AP40
    ret = eval_class_3d_ap40_by_pts(
        gt_annos, dt_annos, current_classes, pts_ranges, min_overlaps)
    
    mAP40_3d = get_mAP40(ret['precision'])
    
    ret_dict = {}
    result = ''
    
    # Format results
    result += '\n----------- 3D AP40 Results by Points in BBox ------------\n\n'
    pts_range_names = [f'{r[0]}-{r[1]}pts' for r in pts_ranges]
    
    for j, curcls in enumerate(current_classes):
        curcls_name = class_to_name[curcls]
        result += f'{curcls_name} 3D AP40@0.7 IoU:\n'
        
        for idx_r, pts_range_name in enumerate(pts_range_names):
            result += f'  {pts_range_name}: {mAP40_3d[j, idx_r, 0]:.4f}\n'
            ret_dict[f'KITTI/{curcls_name}_3D_AP40_{pts_range_name}'] = mAP40_3d[j, idx_r, 0]
    
    # Calculate overall results if multiple classes
    if len(current_classes) > 1:
        result += '\nOverall 3D AP40@0.7 IoU:\n'
        mAP40_3d_overall = mAP40_3d.mean(axis=0)
        for idx_r, pts_range_name in enumerate(pts_range_names):
            result += f'  {pts_range_name}: {mAP40_3d_overall[idx_r, 0]:.4f}\n'
            ret_dict[f'KITTI/Overall_3D_AP40_{pts_range_name}'] = mAP40_3d_overall[idx_r, 0]
    
    return result, ret_dict