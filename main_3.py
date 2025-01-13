import cv2
import numpy as np
import onnxruntime as ort
import torch  # Import torch for tensor operations
from utils.general import non_max_suppression, scale_boxes
from utils.plots import colors
from utils.augmentations import letterbox

# Load ONNX model
model_path = 'yolov5s.onnx'  # Path to your ONNX model
session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])

print("ONNX model loaded successfully")

# Assign category_dict as model names
category_dict = {
    0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'chair', 5: 'bus',
    6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light', 10: 'fire hydrant',
    11: 'stop sign', 12: 'parking meter', 13: 'bench', 14: 'bird', 15: 'cat',
    16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow', 20: 'elephant', 21: 'bear',
    22: 'zebra', 23: 'giraffe', 24: 'backpack', 25: 'umbrella', 26: 'handbag',
    27: 'tie', 28: 'suitcase', 29: 'frisbee', 30: 'skis', 31: 'snowboard',
    32: 'sports ball', 33: 'kite', 34: 'baseball bat', 35: 'baseball glove',
    36: 'skateboard', 37: 'surfboard', 38: 'tennis racket', 39: 'bottle',
    40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon', 45: 'bowl',
    46: 'banana', 47: 'apple', 48: 'sandwich', 49: 'orange', 50: 'broccoli',
    51: 'carrot', 52: 'hot dog', 53: 'pizza', 54: 'donut', 55: 'cake',
    56: 'chair', 57: 'couch', 58: 'potted plant', 59: 'bed', 60: 'dining table',
    61: 'toilet', 62: 'tv', 63: 'laptop', 64: 'mouse', 65: 'remote', 66: 'keyboard',
    67: 'cell phone', 68: 'microwave', 69: 'oven', 70: 'toaster', 71: 'sink',
    72: 'refrigerator', 73: 'book', 74: 'clock', 75: 'vase', 76: 'scissors',
    77: 'teddy bear', 78: 'hair drier', 79: 'toothbrush',
    80: 'office chair', 81: 'recliner', 82: 'gaming chair',  # Added chair types
    83: 'chair legs'
}

# Define categories to display and confidence threshold
display_categories = {0, 4}  # Only 'person' and 'chair'
confidence_threshold = 0.5  # Increased to reduce false positives

# Initialize video capture
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Failed to open video capture")
    exit()

# Retrieve model's expected input shape dynamically
input_shape = session.get_inputs()[0].shape[2:]  # Get height and width
print(f"Model input shape: {input_shape}")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break  # Exit if no frame is captured

    img0 = frame.copy()
    # Resize the image to match the model's expected input shape
    img = letterbox(frame, input_shape, stride=32, auto=False)[0]
    img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW and BGR to RGB
    img = np.ascontiguousarray(img, dtype=np.float32) / 255.0  # Normalize

    # Add batch dimension
    img_tensor = np.expand_dims(img, axis=0)

    # Run inference
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: img_tensor})

    # Convert the ONNX model output to a PyTorch tensor
    predictions = torch.tensor(outputs[0])  # Convert numpy array to PyTorch tensor

    # Apply non-max suppression
    pred = non_max_suppression(predictions, confidence_threshold, 0.45, agnostic=False, max_det=1000)

    for i, det in enumerate(pred):
        if len(det):
            det[:, :4] = scale_boxes(img_tensor.shape[2:], det[:, :4], img0.shape).round()
            
            for *xyxy, conf, cls in det[:, :6]:
                c = int(cls)
                if c in display_categories and conf >= confidence_threshold:
                    label = f'{category_dict[c]} {conf:.2f}'
                    
                    xyxy = [int(x) for x in xyxy]
                    cv2.rectangle(img0, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), colors(c), 2)
                    cv2.putText(img0, label, (xyxy[0], xyxy[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors(c), 2)

    cv2.imshow('YOLOv5 ONNX - Video Feed', img0)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

