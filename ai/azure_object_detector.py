### dgmd17-tello-objectdetection - AzureObjectDetector ###

import numpy as np
from typing import List, Set, Dict, Tuple, Optional
from ai.object_detector import ObjectDetector
from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from azure.cognitiveservices.vision.computervision.models import OperationStatusCodes
from azure.cognitiveservices.vision.computervision.models import VisualFeatureTypes
from msrest.authentication import CognitiveServicesCredentials
from PIL import Image
import cv2
from io import BytesIO
import base64


class AzureObjectDetector (ObjectDetector):
    """Azure Object Detector using YOLO"""

    def __init__(self):
        # This key will serve all examples in this document.
        self.KEY = "5d8006b1d51745c09051886b1b33ba8e"

        # This endpoint will be used in all examples in this quickstart.
        self.ENDPOINT = "https://harvard-dgmd-computervision.cognitiveservices.azure.com"

        # Create an authenticated FaceClient.

        self.computervision_client = ComputerVisionClient(self.ENDPOINT, CognitiveServicesCredentials(self.KEY))


    def detect_people (self, image: Image,  previous_image : Image = None) -> np.ndarray:
        """Detect people and return bounding boxes of all people in the given image"""

        #Convert Image object into Buffer Stream
        buff = BytesIO()
        image.save(buff, format="JPEG")
        img_str = base64.b64encode(buff.getvalue())

        #Azure requires buffer stream to be on a base 64 enconding
        buff = BytesIO(base64.b64decode(img_str))

        #Call Azure Cloud APIs to detect people
        tags_result_remote = self.computervision_client.detect_objects_in_stream(buff)

        bounding_boxes = list()

        #Navigate through all objects detected and only select people
        for x in tags_result_remote.objects:
            if x.object_property == 'person':
                bounding_boxes.append((x.rectangle, x.confidence))

        #Return bounding boxes numpy array
        return np.array(bounding_boxes)

    

    def measure_distance (self, image: Image, bounding_boxes: np.ndarray)-> Image:
        """Return distance between bounding Boxes of all people in the given image"""
        pass

    def draw_bounding_boxes (self,  image: Image, bounding_boxes: np.ndarray, color: Tuple = (0, 179, 60), border: int = 4, previous_image : Image = None ) -> np.ndarray:
        """Draw bounding boxes in the given image"""
        try:
            img_array = np.array(image)

            for item in bounding_boxes:
                box = item[0]

                img_array = cv2.rectangle(img_array,(box.x, box.y),(box.x + box.w,box.y + box.h ),color, border )

                img_array = cv2.putText(img_array, "face", (box.x,box.y-10), cv2.FONT_HERSHEY_COMPLEX, fontScale=1, color=color, thickness=1)
            return Image.fromarray(img_array)
        except:
            print ("error drawing boundind boxes. Azure.")
