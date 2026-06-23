import cv2
import numpy as np
from PIL import Image


def process_uploaded_image(image: Image.Image):
    rgb = np.array(image.convert("RGB"))
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 75, 180)

    gray_rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    edges_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)

    return rgb, gray_rgb, edges_rgb
