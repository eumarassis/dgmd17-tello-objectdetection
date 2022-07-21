### dgmd17-tello-objectdetection - YOLOObjectDetector ###

import numpy as np
from typing import List, Dict, Tuple, Optional
from ai.object_detector import ObjectDetector

class YOLOObjectDetector (ObjectDetector):
    """YOLO Object Detector Class"""

    def __init__(self):
        pass

    def detect_people (self, image: np.ndarray) -> np.ndarray:
        """Detect people and return bounding boxes of all people in the given image"""
        return np.ones((10, 10))

    def measure_distance (self, image: np.ndarray, bounding_boxes: np.ndarray)-> np.ndarray:
        """Return distance between bounding Boxes of all people in the given image"""
        pass

    def draw_bounding_boxes (self,  image: np.ndarray, bounding_boxes: np.ndarray, color: str, border: int  ) -> np.ndarray:
        """Draw bounding boxes in the given image"""
        return image

