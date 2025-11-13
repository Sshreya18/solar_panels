# models/dust.py
import json
import os
from roboflow import Roboflow
from dotenv import load_dotenv

# Load .env file
load_dotenv()
api_key = os.getenv("ROBOFLOW_API_KEY")

rf = Roboflow(api_key=api_key)
project = rf.workspace("solarpanel-nermr").project("dust-7egvc")
version = project.version(1)   
model = version.model

def compute_dust_severity(image_path: str) -> float:
    try:
        print("[INFO] Loading dust model..")
        response = model.predict(image_path).json()
        preds_outer = response.get("predictions", [])
        if not preds_outer:
            print("[DustModel] No predictions found.")
            return 0.0
        
        preds_inner = preds_outer[0].get("predictions", [])
        if not preds_inner:
            print("[DustModel] No nested predictions.")
            return 0.0

        pred = preds_inner[0]
        label = pred.get("class", "").lower()
        confidence = float(pred.get("confidence", 0))

        if label not in ["dust", "clean"]:
            print(f"[DustModel] Unknown label: {label}")
            return confidence * 50  # neutral fallback

        severity = confidence * 100 if label == "dust" else confidence * 0
        return round(severity, 2)

    except Exception as e:
        print("[DustModel] Inference Error:", e)
        return 0.0