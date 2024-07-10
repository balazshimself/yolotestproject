import benchmark2

pred_boxes = [
    [0, 1, 1, 2, 2, 0.9],  # [class, x1, y1, x2, y2, confidence]
    [0, 0.5, 0.5, 1, 1, 0.8],  # class 0
    [1, 1, 1, 2, 2, 0.95]  # class 1
]
gt_boxes = [
    [0, 1, 1, 2, 2],  # class 0
    [1, 1, 1, 2, 2]  # class 1
]

print("mAP:", benchmark2.mean_average_precision(pred_boxes[0:4], gt_boxes[0:4]))
