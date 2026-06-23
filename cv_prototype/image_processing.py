import numpy as np
from PIL import Image

CV2_AVAILABLE = True
CV2_IMPORT_ERROR = ""
try:
    import cv2
except Exception as exc:
    cv2 = None
    CV2_AVAILABLE = False
    CV2_IMPORT_ERROR = str(exc)


def cv_prototype_available() -> bool:
    return CV2_AVAILABLE


def cv_prototype_error_message() -> str:
    return CV2_IMPORT_ERROR


def process_uploaded_image(image: Image.Image):
    rgb = np.array(image.convert("RGB"))
    if not CV2_AVAILABLE:
        return rgb, rgb, rgb

    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 75, 180)

    gray_rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    edges_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)

    return rgb, gray_rgb, edges_rgb
