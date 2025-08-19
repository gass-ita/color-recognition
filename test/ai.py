import torch
import cv2

# Load YOLOv5 (local cloned version)
model = torch.load('yolov5s.pt', map_location='cpu')  # or 'cuda' if you have a GPU!
model.eval()

# For inference with the ultralytics repo model object...
import sys
sys.path.append('.')
from models.common import DetectMultiBackend
from utils.datasets import letterbox
from utils.general import non_max_suppression, scale_coords
from utils.augmentations import letterbox

# Alternatively, use torch.hub as before:
# model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

PHONE_CLASS_ID = 67  # COCO cell phone class

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Prepare image (this matches the YOLOv5 letterbox/dataset format)
    img = letterbox(frame, stride=32, auto=True)[0]
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3xHxW
    img = np.ascontiguousarray(img)
    import numpy as np
    img = torch.from_numpy(img).float()
    img /= 255.0  # 0 - 1
    img = img.unsqueeze(0)  # add batch dim

    with torch.no_grad():
        pred = model(img)

    # NMS (requires same device as model)
    pred = non_max_suppression(pred, conf_thres=0.35)[0]

    detected_phones = 0
    if pred is not None and len(pred):
        pred[:, :4] = scale_coords(img.shape[2:], pred[:, :4], frame.shape).round()
        for *xyxy, conf, cls in pred.tolist():
            # Only plot cell phone (class 67)
            if int(cls) == PHONE_CLASS_ID:
                detected_phones += 1
                x1, y1, x2, y2 = map(int, xyxy)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 3)
                cx = (x1 + x2)//2
                cy = (y1 + y2)//2
                cv2.putText(frame, f"Phone {detected_phones} ({conf:.2f})", (x1, y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
                cv2.circle(frame, (cx, cy), 6, (0,0,255), -1)

    cv2.putText(frame, f"Phones detected: {detected_phones}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

    cv2.imshow('Phone Detector (YOLOv5+PyTorch)', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()