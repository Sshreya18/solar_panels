import cv2
import numpy as np
from roboflow import Roboflow
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv("ROBOFLOW_API_KEY")

rf = Roboflow(api_key=api_key)
project = rf.workspace().project("solar_panels_cracks-l0hsf")
crack_model = project.version(9).model

def compute_crack_severity(image_path, max_expected_cracks=10, w_a=0.33, w_c=0.33, w_l=0.33):
    print("[INFO] Loading cracks model...")
    result = crack_model.predict(image_path, confidence=40)
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not load image: {image_path}")

    img_area = img.shape[0] * img.shape[1]
    height, width = img.shape[:2]
    center_x, center_y = width / 2, height / 2
    predictions = result.json()["predictions"]

    mask_total = np.zeros(img.shape[:2], dtype=np.uint8)
    location_scores = []

    for pred in predictions:
        if "points" in pred:
            pts = np.array([[int(p['x']), int(p['y'])] for p in pred['points']], np.int32)
            pts = pts.reshape((-1, 1, 2))
            cv2.fillPoly(mask_total, [pts], 255)

            M = cv2.moments(pts)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                dist_center = np.sqrt((cx - center_x)**2 + (cy - center_y)**2)
                max_dist = np.sqrt(center_x**2 + center_y**2)
                norm_dist = 1 - (dist_center / max_dist)
                location_scores.append(norm_dist)

    area_factor = np.sum(mask_total > 0) / img_area * 100
    count_factor = min(len(predictions), max_expected_cracks) / max_expected_cracks * 100
    location_factor = np.mean(location_scores) * 100 if location_scores else 0

    consolidated_score = (w_a * area_factor) + (w_c * count_factor) + (w_l * location_factor)
    return round(float(consolidated_score), 2)
