import cv2
import numpy as np
from roboflow import Roboflow
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv("ROBOFLOW_API_KEY")

rf = Roboflow(api_key=api_key)
project = rf.workspace().project("discoloration-m3gvs")
discoloration_model = project.version(19).model

def compute_discoloration_severity(image_path):

    print("[INFO] Loading discoloration model...")
    result = discoloration_model.predict(image_path, confidence=40)

    # Load original image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not load image: {image_path}")

    img_area = img.shape[0] * img.shape[1]
    predictions = result.json()["predictions"]

    # Create a blank mask
    mask_total = np.zeros(img.shape[:2], dtype=np.uint8)

    # Fill polygons
    for pred in predictions:
        if "points" in pred:
            pts = np.array([[int(p['x']), int(p['y'])] for p in pred['points']], np.int32)
            pts = pts.reshape((-1, 1, 2))
            cv2.fillPoly(mask_total, [pts], 255)

    # Compute severity
    discolored_area = np.sum(mask_total > 0)
    severity_score = (discolored_area / img_area) * 100

    return round(float(severity_score), 2)

