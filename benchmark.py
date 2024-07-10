# benchmark(model="yolov8n.pt", data="coco8.yaml", imgsz=640, half=False)

from ultralytics import YOLO
import numpy as np
import os
import cv2
from scipy.optimize import linear_sum_assignment

IMAGES = []
LABELS = []


def load_images_and_labels(folder):
    image_folder = os.path.join(folder, 'images/')
    label_folder = os.path.join(folder, 'labels/')

    for filename in os.listdir(image_folder):
        without_extension = filename.split('.')[0]
        img_path = os.path.join(image_folder, without_extension + '.jpg')
        label_path = os.path.join(label_folder, without_extension + '.txt')

        img = cv2.imread(img_path)
        if img is not None:
            IMAGES.append(img)

        labels = np.loadtxt(label_path)
        if len(labels.shape) == 1:
            labels = labels[np.newaxis, :]
        LABELS.append(labels)


def iou(boxA, boxB):
    """
    Calculate the Intersection over Union (IOU) between two bounding boxes.
    """
    interAreaLengthX = min(boxA[3], boxB[3]) - max(boxA[1], boxB[1]) + 1
    interAreaLengthY = min(boxA[4], boxB[4]) - max(boxA[2], boxB[2]) + 1

    interArea = (max(0, interAreaLengthX) * max(0, interAreaLengthY))
    boxAArea = (boxA[3] - boxA[1] + 1) * (boxA[4] - boxA[2] + 1)
    boxBArea = (boxB[3] - boxB[1] + 1) * (boxB[4] - boxB[2] + 1)

    return interArea / float(boxAArea + boxBArea - interArea)


def calculate_iou_matrix(pred_boxes, gt_boxes):
    num_preds = len(pred_boxes)
    num_gts = len(gt_boxes)
    iou_matrix = np.zeros((num_preds, num_gts))

    for i, pred in enumerate(pred_boxes):
        for j, gt in enumerate(gt_boxes):
            iou_matrix[i, j] = iou(pred, gt)

    return iou_matrix


def match_boxes(pred_boxes, gt_boxes):
    iou_matrix = calculate_iou_matrix(pred_boxes, gt_boxes)
    row_ind, col_ind = linear_sum_assignment(-iou_matrix)
    return row_ind, col_ind, iou_matrix[row_ind, col_ind]





load_images_and_labels('val_data_small/')
model = YOLO('yolov5n.pt')
results = model(IMAGES)

RESULT_BOXES = []
for result in results:
    result_boxes = np.array([[box.cls, *box.xywhn[0]] for box in result.boxes])
    RESULT_BOXES.append(result_boxes)

matched_pred_indices, matched_gt_indices, matched_ious = match_boxes(RESULT_BOXES[0], LABELS[0])
mean_iou = np.mean(matched_ious)
print("Mean IoU:", mean_iou)
