import os
import cv2
import numpy as np
from glob import glob
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import matplotlib

IMAGE_DIR = "data/test/images"
LABEL_DIR = "data/test/labels"
MAX_DELTA_T = 50  # max ΔT for normalization

def read_yolo_annotations(label_file, img_width=640, img_height=640):
    """Read YOLO annotations and convert to bounding boxes."""
    annotations = []
    try:
        with open(label_file, 'r') as f:
            for line in f:
                parts = list(map(float, line.strip().split()))
                if len(parts) == 5:
                    class_id = int(parts[0])
                    cx, cy, w_norm, h_norm = parts[1:]
                    cx, cy = int(cx * img_width), int(cy * img_height)
                    w, h = int(w_norm * img_width), int(h_norm * img_height)
                    x_min = max(0, cx - w // 2)
                    y_min = max(0, cy - h // 2)
                    x_max = min(img_width, cx + w // 2)
                    y_max = min(img_height, cy + h // 2)
                    annotations.append({"class_id": class_id, "bbox": (x_min, y_min, x_max, y_max)})
    except FileNotFoundError:
        pass
    return annotations

def calculate_relative_delta_t(image_path, annotations):
    """Compute ΔT for anomaly regions using grayscale intensity."""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return []

    results = []
    anomaly_mask = np.zeros_like(img, dtype=bool)
    for ann in annotations:
        x_min, y_min, x_max, y_max = ann["bbox"]
        anomaly_mask[y_min:y_max, x_min:x_max] = True

    reference_pixels = img[~anomaly_mask]
    T_reference_avg = np.mean(reference_pixels) if len(reference_pixels) > 0 else np.mean(img)

    for ann in annotations:
        x_min, y_min, x_max, y_max = ann["bbox"]
        anomaly_region = img[y_min:y_max, x_min:x_max]
        if anomaly_region.size == 0:
            continue
        T_anomaly_max = np.max(anomaly_region)
        delta_t = T_anomaly_max - T_reference_avg
        results.append({
            "class_id": ann["class_id"],
            "bbox": ann["bbox"],
            "Relative_Delta_T_Score": delta_t
        })
    return results

def get_pid_severity(image_path):
    """
    Compute severity score for a thermal image based on PID + ΔT.
    Returns a float between 0 and 1.
    """
    print("[INFO] PID model...")
    base = os.path.basename(image_path).replace(".jpg", "")
    label_file = os.path.join(LABEL_DIR, base + ".txt")

    # Load annotations
    annotations = read_yolo_annotations(label_file)
    if not annotations:
        return 0.0  # No PID anomalies

    # Compute ΔT scores
    delta_results = calculate_relative_delta_t(image_path, annotations)
    if not delta_results:
        return 0.0

    # Compute cluster info
    centers = np.array([((x1+x2)//2, (y1+y2)//2) for r in delta_results for (x1,y1,x2,y2) in [r["bbox"]]])
    clustering = DBSCAN(eps=60, min_samples=1).fit(centers)
    labels = clustering.labels_

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    total_hotspots = len(centers)
    largest_cluster = max([np.sum(labels == i) for i in range(n_clusters)], default=0)
    pid_risk_score = largest_cluster / total_hotspots if total_hotspots > 0 else 0

    # Max ΔT
    max_delta = max([r["Relative_Delta_T_Score"] for r in delta_results])
    normalized_delta = min(max_delta / MAX_DELTA_T, 1.0)

    # Severity Score: weighted sum of ΔT + PID + cluster spread
    w_delta, w_pid, w_cluster = 0.5, 0.4, 0.1
    severity_score = (w_delta * normalized_delta +
                      w_pid * pid_risk_score +
                      w_cluster * min(n_clusters / total_hotspots, 1.0))

    # Return as 0-1 float
    return min(max(severity_score, 0.0), 1.0)
