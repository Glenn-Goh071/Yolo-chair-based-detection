from ultralytics import YOLOv5
import cv2
import torch
import time

MODEL_PATH = 'yolov5n.pt'  # Path to your YOLOv5 model weights

# Load the YOLOv5 model with AutoShape already enabled by default
model = torch.hub.load('ultralytics/yolov5', 'custom', path=MODEL_PATH, force_reload=True)

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

# Initialize video capture
cap = cv2.VideoCapture(0)

# Set desired resolution
desired_width = 640
desired_height = 360
cap.set(cv2.CAP_PROP_FRAME_WIDTH, desired_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, desired_height)

if not cap.isOpened():
    print("Error: Camera not accessible.")
    exit()

# Specify which object types to display and set a confidence threshold
ALLOWED_CLASSES = {0,4}  # Example: 0 corresponds to 'person', 4 to 'chair'
CONFIDENCE_THRESHOLD = 0.2  # Only display detections with confidence > 0.2

# Create a dictionary to track stable detections
object_tracker = {}

# Start time for FPS calculation
prev_time = time.time()

while True:
    ret, frame = cap.read()
    
    if not ret:
        print("Error: Failed to capture image.")
        break
    
    # Measure the start time of the frame
    start_time = time.time()

    # Perform inference using the AutoShape feature
    results = model(frame)  # No need to preprocess the image manually
    annotated_frame = frame.copy()

    # Clear old detections from tracker
    for obj_id in list(object_tracker.keys()):
        object_tracker[obj_id]['frame_count'] -= 1
        if object_tracker[obj_id]['frame_count'] <= 0:
            del object_tracker[obj_id]

    # Process detections
    for result in results.xyxy[0]:  # Iterate through detections
        x1, y1, x2, y2, confidence, class_id = result

        if int(class_id) in ALLOWED_CLASSES and confidence > CONFIDENCE_THRESHOLD:
            label = f"{category_dict[int(class_id)]} {confidence:.2f}"

            # Use object id based on coordinates to track stability
            obj_id = (int(x1), int(y1), int(x2), int(y2))

            # Update tracker or add new detection
            if obj_id not in object_tracker:
                object_tracker[obj_id] = {'frame_count': 3}  # Set a frame count for stability
            else:
                object_tracker[obj_id]['frame_count'] = 3  # Reset count for stable detection

            # Draw bounding box and label
            cv2.rectangle(annotated_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(annotated_frame, label, (int(x1) + 2, int(y1) + 17), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Measure end time and calculate FPS
    end_time = time.time()
    fps = 1 / (end_time - start_time)
    
    # Display FPS on the frame
    cv2.putText(annotated_frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow('Live Feed', annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


