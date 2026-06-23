from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image

CV2_AVAILABLE = True
CV2_IMPORT_ERROR = ""
try:
    import cv2
except ImportError as exc:
    cv2 = None
    CV2_AVAILABLE = False
    CV2_IMPORT_ERROR = str(exc)


@dataclass
class DetectionResult:
    label: str
    confidence: float
    bbox: Tuple[int, int, int, int]


@dataclass
class DamageAssessment:
    risk_score: int
    possible_issue: str
    recommendation: str


YOLO_AVAILABLE = False
try:
    from ultralytics import YOLO

    YOLO_AVAILABLE = True
except Exception:
    YOLO = None


def _draw_bbox(image: np.ndarray, bbox: Tuple[int, int, int, int], label: str, confidence: float) -> np.ndarray:
    if not CV2_AVAILABLE:
        return image.copy()

    x1, y1, x2, y2 = bbox
    output = image.copy()
    cv2.rectangle(output, (x1, y1), (x2, y2), (22, 101, 169), 2)
    text = f"{label} ({confidence:.2f})"
    cv2.putText(output, text, (x1, max(20, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (22, 101, 169), 2)
    return output


def detect_with_yolo(rgb_image: np.ndarray) -> Tuple[np.ndarray, List[DetectionResult], bool]:
    """Run YOLO detection when ultralytics is available. Returns annotated image and detections.

    NOTE: This is a prototype using a general pre-trained model unless a custom damage model is provided.
    """
    annotated = rgb_image.copy()
    detections: List[DetectionResult] = []

    if not YOLO_AVAILABLE or not CV2_AVAILABLE:
        return annotated, detections, False

    try:
        model = YOLO("yolov8n.pt")
        results = model.predict(rgb_image, verbose=False)
        if not results:
            return annotated, detections, True

        result = results[0]
        names = result.names
        boxes = result.boxes
        if boxes is None:
            return annotated, detections, True

        for box in boxes:
            cls_id = int(box.cls.item())
            conf = float(box.conf.item())
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int).tolist()
            label = names.get(cls_id, str(cls_id)) if isinstance(names, dict) else str(cls_id)
            detections.append(DetectionResult(label=label, confidence=conf, bbox=(x1, y1, x2, y2)))
            annotated = _draw_bbox(annotated, (x1, y1, x2, y2), label, conf)

        return annotated, detections, True
    except Exception:
        return rgb_image.copy(), [], False


def compute_damage_risk(rgb_image: np.ndarray) -> Tuple[np.ndarray, DamageAssessment]:
    if not CV2_AVAILABLE:
        return rgb_image.copy(), DamageAssessment(
            risk_score=0,
            possible_issue="Computer Vision prototype is unavailable in this deployment environment.",
            recommendation="OpenCV could not be loaded in this runtime.",
        )

    bgr = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    # Feature 1: edge density
    edges = cv2.Canny(gray, 75, 180)
    edge_density = float(np.mean(edges > 0))

    # Feature 2: contrast
    contrast = float(np.std(gray))

    # Feature 3: blur indicator (lower laplacian var -> blurrier)
    lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    # Feature 4: contour irregularity proxy
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour_count = len(contours)

    raw_score = (
        min(40, edge_density * 180)
        + min(25, contrast / 3.2)
        + min(20, max(0.0, 25 - lap_var / 20))
        + min(20, contour_count / 10)
    )
    risk_score = int(max(5, min(100, raw_score * 1.2)))

    if risk_score >= 70:
        possible_issue = "Surface damage or panel irregularity detected"
        recommendation = "Manual inspection recommended"
    elif risk_score >= 45:
        possible_issue = "Moderate visual irregularities detected"
        recommendation = "Schedule a visual workshop inspection"
    else:
        possible_issue = "No major visible irregularity in prototype scan"
        recommendation = "Continue standard monitoring"

    edges_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
    assessment = DamageAssessment(
        risk_score=risk_score,
        possible_issue=possible_issue,
        recommendation=recommendation,
    )
    return edges_rgb, assessment


def run_damage_detection(image: Image.Image):
    rgb = np.array(image.convert("RGB"))
    if not CV2_AVAILABLE:
        assessment = DamageAssessment(
            risk_score=0,
            possible_issue="Computer Vision damage detection is unavailable in this deployment environment.",
            recommendation="OpenCV could not be loaded in this runtime.",
        )
        return rgb, rgb, rgb, [], False, assessment

    yolo_image, detections, yolo_used = detect_with_yolo(rgb)
    processed_image, assessment = compute_damage_risk(rgb)
    return rgb, processed_image, yolo_image, detections, yolo_used, assessment
