import cv2

def detect_image_type(image_path: str) -> str:

    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    if len(img.shape) == 2 or (len(img.shape) == 3 and img.shape[2] == 1):
        return "thermal"
    else:
        return "rgb"
