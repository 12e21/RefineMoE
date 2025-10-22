import numpy as np
from shapely.geometry import Polygon

def get_class_boxes(anno, cls_name):
    """
    提取某类别的所有 box 和置信度。
    返回: list of dict {center, size, yaw, score}
    """
    res = []
    for i, name in enumerate(anno['name']):
        if name != cls_name:
            continue
        box = {
            'center': np.array([anno['location'][i][0], anno['location'][i][1]]),
            'size': np.array([anno['dimensions'][i][0], anno['dimensions'][i][1]]),  # 长宽 (l, w)
            'yaw': anno['rotation_y'][i],
            'score': anno['score'][i] if 'score' in anno else 1.0
        }
        res.append(box)
    return res

def box_to_polygon(box):
    """
    将 box 转为 shapely Polygon (2D)
    """
    l, w = box['size']
    x_c, y_c = box['center']
    yaw = box['yaw']
    corners = np.array([
        [ l/2,  w/2],
        [ l/2, -w/2],
        [-l/2, -w/2],
        [-l/2,  w/2]
    ])
    c, s = np.cos(yaw), np.sin(yaw)
    rot = np.array([[c, -s], [s, c]])
    corners = (rot @ corners.T).T
    corners += np.array([x_c, y_c])
    return Polygon(corners)

def compute_ate(gt, dt):
    return np.linalg.norm(gt['center'] - dt['center'])

def compute_ase(gt, dt):
    # 对齐中心和角度后算IOU
    aligned_dt = dt.copy()
    aligned_dt['center'] = gt['center']
    aligned_dt['yaw'] = gt['yaw']
    poly_gt = box_to_polygon(gt)
    poly_dt = box_to_polygon(aligned_dt)
    iou = poly_gt.intersection(poly_dt).area / poly_gt.union(poly_dt).area
    return 1.0 - iou

def compute_aoe(gt, dt, cls_name):
    diff = np.abs((dt['yaw'] - gt['yaw'] + np.pi) % (2*np.pi) - np.pi)
    if cls_name == 'barrier':
        diff = min(diff, np.abs(diff - np.pi))
    return diff

def match_boxes(gt_boxes, dt_boxes, dist_thresh=2.0):
    matches = []
    used_gt = set()
    used_dt = set()
    
    for dt_idx, dt in enumerate(dt_boxes):
        best_gt_idx = -1
        best_dist = dist_thresh
        for gt_idx, gt in enumerate(gt_boxes):
            if gt_idx in used_gt:
                continue
            dist = compute_ate(gt, dt)
            if dist < best_dist:
                best_dist = dist
                best_gt_idx = gt_idx
        if best_gt_idx >= 0:
            matches.append((gt_boxes[best_gt_idx], dt, dt['score']))
            used_gt.add(best_gt_idx)
            used_dt.add(dt_idx)
    return matches

def eval_tp_errors(gt_annos, dt_annos, current_classes):
    results = {}
    # 表头：Class 宽 12，数值列宽 8
    lines = ["{:<12} {:>8} {:>8} {:>8}".format("Class", "ATE", "ASE", "AOE")]

    for cls in current_classes:
        tp_ates = []
        tp_ases = []
        tp_aoes = []
        scores = []
        num_gt_total = 0

        for gt, dt in zip(gt_annos, dt_annos):
            gt_boxes = get_class_boxes(gt, cls)
            dt_boxes = get_class_boxes(dt, cls)
            num_gt_total += len(gt_boxes)
            matches = match_boxes(gt_boxes, dt_boxes)
            for gt_box, dt_box, score in matches:
                tp_ates.append(compute_ate(gt_box, dt_box))
                tp_ases.append(compute_ase(gt_box, dt_box))
                if cls == 'cone':
                    tp_aoes.append(np.nan)
                else:
                    tp_aoes.append(compute_aoe(gt_box, dt_box, cls))
                scores.append(score)

        if len(tp_ates) == 0 or num_gt_total == 0:
            ate_val = 1.0
            ase_val = 1.0
            aoe_val = 1.0 if cls != 'cone' else np.nan
        else:
            scores = np.array(scores)
            tp_ates = np.array(tp_ates)
            tp_ases = np.array(tp_ases)
            tp_aoes = np.array(tp_aoes)

            order = np.argsort(-scores)
            tp_ates = tp_ates[order]
            tp_ases = tp_ases[order]
            tp_aoes = tp_aoes[order]

            recall = np.arange(1, len(tp_ates) + 1) / num_gt_total
            mask = recall >= 0.1

            if np.any(mask):
                ate_val = np.nanmean(tp_ates[mask])
                ase_val = np.nanmean(tp_ases[mask])
                aoe_val = np.nan if cls == 'cone' else np.nanmean(tp_aoes[mask])
            else:
                ate_val = 1.0
                ase_val = 1.0
                aoe_val = 1.0 if cls != 'cone' else np.nan

        # 保存到字典
        results[f"{cls}_ATE"] = float(ate_val)
        results[f"{cls}_ASE"] = float(ase_val)
        results[f"{cls}_AOE"] = float(aoe_val) if not np.isnan(aoe_val) else np.nan

        # 格式化字符串行
        aoe_str = f"{aoe_val:.3f}" if not np.isnan(aoe_val) else "NaN"
        line = "{:<12} {:>8.3f} {:>8.3f} {:>8}".format(cls, ate_val, ase_val, aoe_str)
        lines.append(line)

    output_str = "\n".join(lines)
    return output_str, results

