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

    interested_class = [0]
    img_shape = boundings.pandas().imgs[0].shape
    img_xcenter = img_shape[1]/2
    img_ycenter = img_shape[0]/2
    df_xywh = boundings.pandas().xywh[0]
    df_persons_xywh = df_xywh[df_xywh['class'].isin(interested_class)]
    df_xywhn = boundings.pandas().xywhn[0]
    df_persons_xywhn = df_xywhn[df_xywhn['class'].isin(interested_class)]
    df_xyxy = boundings.pandas().xyxy[0]
    df_persons_xyxy = df_xyxy[df_xyxy['class'].isin(interested_class)]
    df_xyxyn = boundings.pandas().xyxyn[0]
    df_persons_xyxyn = df_xyxyn[df_xyxyn['class'].isin(interested_class)]

    print(boundings.pandas().imgs[0].shape)
    print(boundings.pandas().xyxy[0])

    person_idx = 0
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

    draw_bouding_boxes = detector.draw_bounding_boxes(im1, boundings, "Green", 1 )


if __name__ == "__main__":
    main()

