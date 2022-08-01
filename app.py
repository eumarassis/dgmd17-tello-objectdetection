### dgmd17-tello-objectdetection - App ###
from tello_ui import TelloControlUI
from djitellopy import Tello
from ai.yolo_object_detector import YOLOObjectDetector
from ai.yolo_face_detector import YOLOFaceDetector
from ai.azure_object_detector import AzureObjectDetector
from ai.depth_perception import DepthPerceptionObjectDetector


def main():

    #Initialize Tello Control Object
    tello = Tello() 

    #Initialize List of Object Detection Model Classes
    list_detector = [ \
        ("YOLO: Face Detection Model", YOLOFaceDetector()), 
        ("YOLO: Real-Time Object Detection", YOLOObjectDetector()), \
        ("Self-Supervised Multi-Frame Monocular Depth", DepthPerceptionObjectDetector()), \
        ("Azure Cognitive Service - People Detector", AzureObjectDetector())]

    #Initialize UI Class
    tello_control_ui = TelloControlUI(tello, list_detector)
    
	#Build UI
    tello_control_ui.build_ui()

    #Start the Tkinter mainloop thread
    tello_control_ui.root.mainloop() 


if __name__ == "__main__":
    main()

