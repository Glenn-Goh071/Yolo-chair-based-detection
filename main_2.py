import torch
from pathlib import Path
from models.common import DetectMultiBackend
from utils.general import non_max_suppression, scale_boxes
from utils.plots import colors, plot_images
from utils.augmentations import letterbox
import cv2
import numpy as np

# Load model
model_path = 'yolo_test.pt'  # Path to your segmentation model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = DetectMultiBackend(model_path, device=device, dnn=False)

print("Model loaded successfully")
# Assign category_dict as model names
category_dict = {
    0: 'chair', 1: 'person',
}
model.names = category_dict

# Define categories to display and confidence threshold
display_categories = {0, 1}  # Only 'person' and 'chair'
confidence_threshold = 0.05

# Initialize video capture
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Failed to open video capture")
    exit()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break  # Exit if no frame is captured

    img0 = frame.copy()
    img = letterbox(frame, (640, 640))[0]
    img = img.transpose((2, 0, 1))[::-1]
    img = np.ascontiguousarray(img)

    # Run inference
    img_tensor = torch.from_numpy(img).to(device).float() / 255.0
    if len(img_tensor.shape) == 3:
        img_tensor = img_tensor[None]  # Expand for batch dim

    # Perform prediction
    pred = model(img_tensor, augment=False, visualize=False)

    # Apply non-max suppression
    pred = non_max_suppression(pred, confidence_threshold, 0.45, agnostic=False, max_det=1000)

    for i, det in enumerate(pred):
        if len(det):
            det[:, :4] = scale_boxes(img_tensor.shape[2:], det[:, :4], img0.shape).round()

            chair_detections = []
            other_detections = []

            for *xyxy, conf, cls in det[:, :6]:
                c = int(cls)
                if c == 0:  # Chair category
                    chair_detections.append((*xyxy, conf, cls))
                elif c in display_categories:
                    other_detections.append((*xyxy, conf, cls))

            # Process chair detections to retain only the highest confidence instance
            if chair_detections:
                chair_detections.sort(key=lambda x: x[4], reverse=True)  # Sort by confidence
                chair_detections = [chair_detections[0]]  # Keep only the highest confidence chair

            # Combine filtered detections
            final_detections = chair_detections + other_detections

            for *xyxy, conf, cls in final_detections:
                label = f'{category_dict[int(cls)]} {conf:.2f}'
                xyxy = [int(x) for x in xyxy]
                cv2.rectangle(img0, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), colors(int(cls)), 2)
                cv2.putText(img0, label, (xyxy[0], xyxy[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors(int(cls)), 2)

    cv2.imshow('YOLOv5 Segmentation - Video Feed', img0)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

