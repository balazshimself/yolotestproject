from collections import Counter
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from enum import Enum

"""
Terv:
    - először szűkítés 1 class-ra
    - classonként IoU számolása
    - 

"""


# class syntax
class Result(Enum):
    """
    [1.] |2.|
    |3.| [4.]
    """
    TRUE_POSITIVE = 1
    FALSE_NEGATIVE = 2
    FALSE_POSITIVE = 3
    TRUE_NEGATIVE = 4


def calculate_tp_fp_fn(pred_boxes, gt_boxes, iou_threshold, cls):
    pred_boxes_cls = [box for box in pred_boxes if box[0] == cls]
    gt_boxes_cls = [box for box in gt_boxes if box[0] == cls]

    row_ind, col_ind, ious = matched_boxes(pred_boxes_cls, gt_boxes_cls)

    tp = 0
    fp = 0
    fn = 0

    matched_gt_boxes = set()
    for i, j, iou in zip(row_ind, col_ind, ious):
        if iou >= iou_threshold:
            tp += 1
            matched_gt_boxes.add(j)
        else:
            fp += 1

    fn = len(gt_boxes_cls) - len(matched_gt_boxes)
    fp += len(pred_boxes_cls) - tp

    return tp, fp, fn


def precision(pred_boxes, gt_boxes, iou_threshold=0.5):
    classes = get_classes(pred_boxes, gt_boxes)
    precisions = []
    for cls in classes:
        tp, fp, fn = calculate_tp_fp_fn(pred_boxes, gt_boxes, iou_threshold, cls)
        precisions.append(tp / (tp + fp) if tp + fp > 0 else 0)
    return precisions


def recall(pred_boxes, gt_boxes, iou_threshold=0.5):
    classes = get_classes(pred_boxes, gt_boxes)
    recalls = []
    for cls in classes:
        tp, fp, fn = calculate_tp_fp_fn(pred_boxes, gt_boxes, iou_threshold, cls)
        recalls.append(tp / (tp + fn) if tp + fn > 0 else 0)
    return recalls


def iou2(boxA, boxB):
    """
    Calculate the Intersection over Union (IOU) between two bounding boxes.
    """
    interAreaLengthX = min(boxA[2], boxB[2]) - max(boxA[0], boxB[0]) + 1
    interAreaLengthY = min(boxA[3], boxB[3]) - max(boxA[1], boxB[1]) + 1

    interArea = float(max(0, interAreaLengthX) * max(0, interAreaLengthY))
    boxAArea = float(boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = float(boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    return interArea / float(boxAArea + boxBArea - interArea)


def calculate_padded_iou_matrix(pred_boxes, gt_boxes):
    side_length = max(len(pred_boxes), len(gt_boxes))
    iou_matrix = np.zeros((side_length, side_length))

    for i in range(side_length):
        for j in range(side_length):
            if (i >= len(pred_boxes)) or (j >= len(gt_boxes)):
                iou_matrix[i, j] = 0.0
            else:
                iou_matrix[i, j] = iou2(pred_boxes[i], gt_boxes[j])

    return iou_matrix


def matched_boxes(pred_boxes, gt_boxes):
    iou_matrix = calculate_padded_iou_matrix(pred_boxes, gt_boxes)
    row_ind, col_ind = linear_sum_assignment(-iou_matrix)
    return row_ind, col_ind, iou_matrix[row_ind, col_ind]


def get_classes(pred_boxes, gt_boxes):
    pred_classes = set(box[0] for box in pred_boxes)
    gt_classes = set(box[0] for box in gt_boxes)
    return pred_classes.union(gt_classes)


def calculate_ap(recalls, precisions):
    """
    Calculate the Average Precision (AP) given recall and precision values.
    Uses the 11-point interpolation method.
    """
    recalls = np.concatenate(([0.0], recalls, [1.0]))
    precisions = np.concatenate(([0.0], precisions, [0.0]))

    # Ensure precision is non-decreasing
    for i in range(len(precisions) - 1, 0, -1):
        precisions[i - 1] = np.maximum(precisions[i - 1], precisions[i])

    indices = np.where(recalls[1:] != recalls[:-1])[0]
    ap = np.sum((recalls[indices + 1] - recalls[indices]) * precisions[indices + 1])
    return ap


def calculate_precision_recall(pred_boxes, gt_boxes, iou_thresholds):
    all_precisions = {}
    all_recalls = {}
    for cls in get_classes(pred_boxes, gt_boxes):
        precisions = []
        recalls = []
        for iou_threshold in iou_thresholds:
            tp, fp, fn = calculate_tp_fp_fn(pred_boxes, gt_boxes, iou_threshold, cls)
            precisions.append(tp / (tp + fp) if tp + fp > 0 else 0)
            recalls.append(tp / (tp + fn) if tp + fn > 0 else 0)
        all_precisions[cls] = precisions
        all_recalls[cls] = recalls
    return all_precisions, all_recalls


def mean_average_precision(pred_boxes, gt_boxes, iou_thresholds=np.linspace(0.5, 0.95, 10)):
    precisions, recalls = calculate_precision_recall(pred_boxes, gt_boxes, iou_thresholds)
    aps = []
    for cls in precisions.keys():
        ap = calculate_ap(np.array(recalls[cls]), np.array(precisions[cls]))
        aps.append(ap)
    return np.mean(aps)