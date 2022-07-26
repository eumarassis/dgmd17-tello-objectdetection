### dgmd17-tello-objectdetection - App ###
from djitellopy import Tello
from ai.yolo_object_detector import YOLOObjectDetector
import torch
from PIL import Image
import cv2
import torch

def main():

    #Initialize Tello Control Object
    tello = Tello() 

    im1 = None
    
# Images
    f = 'zidane.jpg'
    torch.hub.download_url_to_file('https://ultralytics.com/images/' + f, f)  # download 2 images
    im1 = Image.open('zidane.jpg')  # PIL image


    #Initialize Object Detector
    detector = YOLOObjectDetector()

    boundings = detector.detect_people(im1)

    print(boundings.pandas().imgs[0].shape)
    print(boundings.pandas().xyxy[0])

    draw_bouding_boxes = detector.draw_bounding_boxes(im1, boundings, "Green", 1 )


if __name__ == "__main__":
    main()

