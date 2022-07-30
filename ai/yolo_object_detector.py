### dgmd17-tello-objectdetection - YOLOObjectDetector ###

import numpy as np
from typing import List, Dict, Tuple, Optional
from ai.object_detector import ObjectDetector
import torch
from PIL import Image

class YOLOObjectDetector (ObjectDetector):
    """YOLO Object Detector Class"""

    def __init__(self):
        self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # or yolov5n - yolov5x6, custom

    def detect_people (self, image: Image,  previous_image : Image = None) -> np.ndarray:
        """Detect people and return bounding boxes of all people in the given image"""
        results = self.model(image)
        
        return results


    def measure_distance (self, image: Image, bounding_boxes: np.ndarray)-> np.ndarray:
        """Return distance between bounding Boxes of all people in the given image"""
        pass

    def draw_bounding_boxes (self,  image: Image, bounding_boxes: np.ndarray, color: Tuple = (0, 179, 60), border: int = 4,  previous_image : Image = None ) -> np.ndarray:
        """Draw bounding boxes in the given image"""
        try:        
            bounding_boxes.render()
            return Image.fromarray( bounding_boxes.imgs[0] )
        except:
            print ("error drawing boundind boxes. YOLO.")
