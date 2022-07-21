### dgmd17-tello-objectdetection - App ###
from tello_ui import TelloControlUI
from djitellopy import Tello
from ai.yolo_object_detector import YOLOObjectDetector

def main():

    #Initialize Tello Control Object
    tello = Tello() 

    #Initialize Object Detector
    detector = YOLOObjectDetector()

    tello_control_ui = TelloControlUI(tello, detector)
    
	#Build UI
    tello_control_ui.build_ui()

    #Start the Tkinter mainloop thread
    tello_control_ui.root.mainloop() 


if __name__ == "__main__":
    main()

