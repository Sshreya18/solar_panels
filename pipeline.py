# pipeline.py
import json 
from models.cracks import compute_crack_severity
from models.discoloration import compute_discoloration_severity
from models.pid import get_pid_severity
from models.hotspots import HotspotModel
from utils.severity_utils import combine_severity_scores
from utils.image_utils import detect_image_type
from models.dust import compute_dust_severity

hotspot_model = HotspotModel()  # Hotspots wreights are loaded here

def run_pipeline(image_path: str, image_type: str = None) -> dict:
    print("started the pipeline")
    if image_type is None:
        image_type = detect_image_type(image_path)

    image_type = image_type.strip().lower()  # sanitize

    severity_scores = {}
    
    if image_type.lower() == "rgb":
        print("RGB image is being processed")
        severity_scores['crack'] = compute_crack_severity(image_path)
        severity_scores['discoloration'] = compute_discoloration_severity(image_path)
        severity_scores['dust'] = compute_dust_severity(image_path)

    elif image_type.lower() == "thermal":
        print("Thermal image is being processed")
        severity_scores['PID'] = get_pid_severity(image_path)
        severity_scores['hotspot'] = hotspot_model.predict(image_path)

    # Combine all severities into a single final score
    final_score = combine_severity_scores(severity_scores)


    return {
        "image_path": image_path,
        "image_type": image_type,
        "model_scores": severity_scores,
        "final_severity": round(final_score, 3)
    }

if __name__ == "__main__":
    test_image = "sample_tests/bad_3.jpg            "
    result = run_pipeline(test_image, image_type="rgb")
    print("Pipeline Output:\n", json.dumps(result, indent=4))

