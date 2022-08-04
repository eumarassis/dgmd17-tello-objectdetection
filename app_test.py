### dgmd17-tello-objectdetection - App ###
import time
import cv2
import torch
import numpy as np
import torch

from djitellopy import Tello
from ai.azure_object_detector import AzureObjectDetector
from ai.depth_perception import DepthPerceptionObjectDetector
from ai.yolo_face_detector import YOLOFaceDetector
from ai.yolo_object_detector import YOLOObjectDetector
from PIL import Image

def main():

    #Initialize Tello Control Object
    tello = Tello() 

    im1 = None

    detection_threshold = 0.7
    
    # Images

    f = "img1.jpg"
    torch.hub.download_url_to_file('https://ultralytics.com/images/zidane.jpg', f)  # download 2 images
    # #torch.hub.download_url_to_file('https://pjreddie.com/media/image/Screen_Shot_2018-03-24_at_10.48.42_PM.png', f)  # download 2 images
    im1 = Image.open('img1.jpg')  # PIL image

    # detector = AzureObjectDetector()

    # bounding_boxes = detector.detect_people(im1)

    # img2 = Image.fromarray(detector.draw_bounding_boxes( np.array(im1), bounding_boxes))

    # img2.show()

    # return 

    #Initialize Object Detector
    objectDetectors = [YOLOObjectDetector(), AzureObjectDetector(), DepthPerceptionObjectDetector()]
    # detector = YOLOObjectDetector()

    filename = time.strftime("%Y%m%d-%H%M%S")
    for detector in objectDetectors:
        model_name = detector.__module__.split('.')[1]
        detected_people = detector.detect_people(im1)
        image = detector.draw_bounding_boxes(im1, detected_people, previous_image=im1)
        image.save('test_output/' + model_name + '_' + filename + '.jpg')

        if model_name == 'yolo_object_detector':
            interested_class = [0]
            img_shape = detected_people.pandas().imgs[0].shape
            img_xcenter = img_shape[1]/2
            img_ycenter = img_shape[0]/2
            df_xywh = detected_people.pandas().xywh[0]
            df_persons_xywh = df_xywh[(df_xywh['class'].isin(interested_class)) & (df_xywh['confidence'] > detection_threshold)]
            
            print(df_persons_xywh)
            print(detected_people.pandas().imgs[0].shape)
            print(detected_people.pandas().xyxy[0])

            person_idx = 0
            if not df_persons_xywh.empty:
                if df_persons_xywh['xcenter'][person_idx] > img_xcenter:
                    # Need to update distance 
                    # self.update_distance()
                    # for now moving 50 cm, need to calcualte the cm using pixel
                    print('move left')# move_left()
                elif df_persons_xywh['xcenter'][person_idx] < img_xcenter:
                    print('move right')# move_right()
                elif df_persons_xywh['ycenter'][person_idx] > img_ycenter:
                    print('move up')# move_up()
                elif df_persons_xywh['ycenter'][person_idx] < img_ycenter:
                    print('move down')# move_down()

            draw_bouding_boxes = detector.draw_bounding_boxes(im1, detected_people, "Green", 1 )


if __name__ == "__main__":
    main()

