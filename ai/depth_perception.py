### dgmd17-tello-objectdetection - YOLOObjectDetector ###

import numpy as np
from typing import List, Dict, Tuple, Optional
from ai.object_detector import ObjectDetector
import torch
from PIL import Image
import os
import torch
from PIL import Image
import cv2
import numpy as np
from manydepth  import manydepth
from ai.yolo_object_detector import YOLOObjectDetector

class DepthPerceptionObjectDetector (ObjectDetector):
    """Depth Perception"""

    def __init__(self):

        self.model_name = 'depth_perception'
        self.object_detection_model = YOLOObjectDetector()
        self.model = manydepth(intrinsics_json_path=os.path.join("assets", "test_sequence_intrinsics.json"))

    def detect_people (self, image: Image,  previous_image : Image = None) -> np.ndarray:
        """Detect people and return bounding boxes of all people in the given image"""
        return self.object_detection_model.model(image)


    def measure_distance (self, image: Image, bounding_boxes: np.ndarray)-> np.ndarray:
        """Return distance between bounding Boxes of all people in the given image"""
        pass

    def draw_bounding_boxes (self,  image: Image, bounding_boxes: np.ndarray, color: Tuple = (0, 179, 60), border: int = 4,  previous_image : Image = None ) -> np.ndarray:
        """Draw bounding boxes in the given image"""
        try:        

            result = self.model.eval(np.array(image), np.array(previous_image))
            return Image.fromarray( result )
        except:
            print ("error drawing boundind boxes. YOLO.")
