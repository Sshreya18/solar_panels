# models/hotspots.py
import numpy as np
from ultralytics import YOLO
import cv2
from config import MODEL_PATHS

class HotspotModel:
    def __init__(self):
        self.model = YOLO(MODEL_PATHS)
        self.class_to_severity = {
            0: 0.0,   # Healthy
            1: 0.2,   # Mild
            2: 0.3,   # Moderate
            3: 0.4,   # Moderate+
            4: 0.6,   # Severe
            5: 0.7,   # Severe+
            6: 0.85,  # Very severe
            7: 1.0    # Worst case
        }

    def predict(self, image_path: str, img_size: int = 640) -> float:
        
        print("[INFO] Loading Hotspot model...")
        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(f"Image not found: {image_path}")
        img_h, img_w = img.shape[:2]

        # Run inference silently
        pred = self.model.predict(image_path, imgsz=img_size, verbose=False)[0]

        # No detections case
        if len(pred.boxes) == 0:
            return 0.0

        classes = pred.boxes.cls.cpu().numpy().astype(int)
        confs = pred.boxes.conf.cpu().numpy()
        boxes = pred.boxes.xyxy.cpu().numpy()

        per_box_severity = []
        for cls, conf, box in zip(classes, confs, boxes):
            x1, y1, x2, y2 = box
            area_factor = ((x2 - x1) * (y2 - y1)) / (img_w * img_h)
            severity = self.class_to_severity.get(cls, 0.0) * conf * (1 + area_factor)
            per_box_severity.append(severity)

        image_severity = float(np.mean(per_box_severity))
        return round(image_severity, 3)

