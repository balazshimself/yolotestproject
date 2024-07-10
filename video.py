# This is a sample Python script.
import torch
from torch.distributed._spmd.api import Override
# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

from ultralytics import YOLO
import cv2
import numpy as np

# Loading yolov5
model1 = YOLO('yolov5n.pt')
model2 = YOLO('yolov8n.pt')
model3 = YOLO("yolov10n.pt")

color1 = (0, 255, 0)  # Green for model1
color2 = (255, 0, 0)  # Blue for model2
color3 = (0, 0, 255)  # Red for model3


def draw_boxes(input_frame, boxes, color):
    for box in boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])  # Adjust indexing based on your structure
        confidence = np.maximum(float(box.conf[0]), 0.3)
        opacity = confidence  # Using confidence as the opacity level

        # Create a separate overlay for each box
        box_overlay = input_frame.copy()

        cv2.rectangle(box_overlay, (x1, y1), (x2, y2), color, 6)
        label = f"{int(box.cls[0])} {round(float(box.conf[0]), 3)}"  # Adjust to match your label format
        cv2.putText(box_overlay, label, (x1, y1 - 10), cv2.FONT_HERSHEY_DUPLEX, 2, color, 3)
        cv2.addWeighted(box_overlay, opacity, input_frame, 1 - opacity, 0, input_frame)

    return input_frame


video_path = './test.mp4'
cap = cv2.VideoCapture(video_path)

# Get video properties
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter.fourcc(*'mp4v')
out = cv2.VideoWriter('v5v8v10compare.mp4', fourcc, fps, (frame_width, frame_height))

video_not_over = True

while video_not_over:
    video_not_over, frame = cap.read()

    if video_not_over:

        results1 = model1.track(frame, persist=True)
        results2 = model2.track(frame, persist=True)
        results3 = model3.track(frame, persist=True)

        frame_with_boxes = draw_boxes(frame.copy(), results1[0].boxes, color1)
        frame_with_boxes = draw_boxes(frame_with_boxes, results2[0].boxes, color2)
        frame_with_boxes = draw_boxes(frame_with_boxes, results3[0].boxes, color3)

        out.write(frame_with_boxes)

        # visualize
        imS = cv2.resize(frame_with_boxes, (960, 540))
        cv2.imshow('frame', imS)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

cap.release()
out.release()
cv2.destroyAllWindows()
