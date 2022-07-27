### dgmd17-tello-objectdetection - TelloUI ###

from tkinter import *
from tkinter import ttk
from PIL import ImageTk, Image
from djitellopy import Tello
import threading
import datetime
import cv2
import os
import time
import platform

import numpy as np
from typing import List, Set, Dict, Tuple, Optional
from ai.object_detector import ObjectDetector

class TelloControlUI:
    """Tello Control User Interface Class"""

    def __init__(self,tello : Tello, object_detector : ObjectDetector ):
        print("INFO: Screen Class Initialized")

        #Initialize UI Componenets
        self.root = Tk()
        self.frm = ttk.Frame(self.root, padding=10)
        self.frm.grid()
        self.root.wm_title("Tello Control App")

        #Set Variables
        self.tello = tello

        self.object_detector = object_detector

        #Initialize Thread
        self.is_connected = False
        self.is_flying = False
        self.is_streaming = False
        self.stopEvent = None
        self.thread = None

        #Initialize image
        self.image_panel = None

        #Set default move distance, degree and threshold
        self.distance = 20
        self.degree = 30
        self.detection_threshold = 0.8

        #Subscribe to Window Close Event
        self.root.wm_protocol("WM_DELETE_WINDOW", self.on_close)

    
    def connect_handler (self):
        """Handle click to Connect. Connect to Tello Using Tello API."""
        
        if self.is_connected == True:
            return
            
        self.tello.connect()

        ttk.Label(self.frm, text="Battery: {}%".format( self.tello.get_battery())).grid(column=6, row=0)

        self.is_connected = True

        self.btn_connect['state'] = "disabled"
        self.btn_takeoff['state'] = "normal"
        self.btn_land['state'] = "normal"
        self.btn_streamon['state'] = "normal"
        self.btn_streamoff['state'] = "normal"
        

    def takeoff_handler (self):
        """Handle click to Take off. Take off Using Tello API."""
        
        if self.is_connected == False:
            return

        self.tello.takeoff()

        self.is_flying = True

    def land_handler (self):
        """Handle click to Land. Land Using Tello API."""

        if self.is_connected == False:
            return

        self.tello.land()

        self.is_flying = False

    def start_streaming_handler (self):
        """Handle click to Start streaming. Create thread to start capturing Tello feed."""

        if self.is_connected == False or self.is_streaming == True:
            return

        self.is_streaming = True
        self.tello.streamon()
        self.thread = threading.Thread(target=self.video_capture_thread, args=())
        self.thread.start()


    def stop_streaming_handler (self):
        """Handle click to Stop streaming. Stop Thread."""

        if self.is_connected == False or self.is_streaming == False:
            return

        self.tello.streamoff()   
        self.is_streaming = False


    def update_distance(self, event):
        """Handle click to update distance."""
        self.distance = int (event) 

    def update_degree(self, event):
        """Handle click to update distance."""
        self.degree = int (event)

    def on_keypress_a(self, event):
        """Handle click to move drone using Tello API."""

        if self.is_connected == False or self.is_flying == False:
            return

        self.tello.move_up(self.distance)

    def on_keypress_z(self, event):
        """Handle click to move drone using Tello API.""" 

        if self.is_connected == False or self.is_flying == False:
            return

        self.tello.move_down(self.distance)

    def on_keypress_s(self, event):
        """Handle click to rotate drone using Tello API."""

        if self.is_connected == False or self.is_flying == False:
            return

        self.tello.rotate_counter_clockwise(self.degree)

    def on_keypress_d( self, event):
        """Handle click to rotate drone using Tello API."""

        if self.is_connected == False or self.is_flying == False:
            return

        self.tello.rotate_clockwise(self.degree)

    def on_keypress_up(self,  event):
        """Handle click to move drone using Tello API."""

        if self.is_connected == False or self.is_flying == False:
            return

        self.tello.move_forward(self.distance)

    def on_keypress_down(self, event):
        """Handle click to move drone using Tello API."""

        if self.is_connected == False or self.is_flying == False:
            return

        self.tello.move_back(self.distance)

    def on_keypress_left(self, event):
        """Handle click to move drone using Tello API."""

        if self.is_connected == False or self.is_flying == False:
            return

        self.tello.move_left(self.distance)

    def on_keypress_right(self, event):
        """Handle click to move drone using Tello API."""

        if self.is_connected == False or self.is_flying == False:
            return

        self.tello.move_right(self.distance)

    def build_ui(self):
        """Build drone control UI using tkinter objects."""        
        self.btn_connect = ttk.Button(self.frm, text="Connect", command=self.connect_handler)
        self.btn_connect.grid(row=0, column=1)
        self.btn_takeoff = ttk.Button(self.frm, text="Take Off", command=self.takeoff_handler, state = DISABLED)
        self.btn_takeoff.grid(row=0, column=2)
        self.btn_land = ttk.Button(self.frm, text="Land", command=self.land_handler, state = DISABLED)
        self.btn_land.grid(row=0, column=3)    
        self.btn_streamon = ttk.Button(self.frm, text="Start Streaming", command=self.start_streaming_handler, state = DISABLED)
        self.btn_streamon.grid(row=0, column=4 )    
        self.btn_streamoff = ttk.Button(self.frm, text="Stop Streaming", command=self.stop_streaming_handler, state = DISABLED)
        self.btn_streamoff.grid(row=0, column=5 )    

        ttk.Label(self.frm, text=
                            'A - Move Tello Up\n'
                            'Z - Move Tello Down\n'
                            'S - Rotate Tello Counter-Clockwise\n'
                            'D - Rotate Tello Clockwise').grid(row=2, column=1 )  

        ttk.Label(self.frm, text=
                            'Arrow Up - Move Tello Forward\n'
                            'Arrow Down - Move Tello Backward\n'
                            'Arrow Left - Move Tello Left\n'
                            'Arrow Right - Move Tello Right').grid(row=2, column=2 )  


        Scale(self.frm, from_=20, to=50, tickinterval=10, label='Distance(cm)', command=self.update_distance).grid( row=2, column=4)

        Scale(self.frm, from_=30, to=360, tickinterval=10, label='Degree', command=self.update_degree).grid(row=2, column=5)


        self.root.bind('<KeyPress-a>', self.on_keypress_a)
        self.root.bind('<KeyPress-z>', self.on_keypress_z)
        self.root.bind('<KeyPress-s>', self.on_keypress_s)
        self.root.bind('<KeyPress-d>', self.on_keypress_d)
        self.root.bind('<KeyPress-Up>', self.on_keypress_up)
        self.root.bind('<KeyPress-Down>', self.on_keypress_down)
        self.root.bind('<KeyPress-Left>', self.on_keypress_left)
        self.root.bind('<KeyPress-Right>', self.on_keypress_right)
        
        image = Image.open("tello-drone.jpg")
        photo = ImageTk.PhotoImage(image)

        self.image_panel = Label(self.root, image = photo, width=600, height=400)
        self.image_panel.image = photo
        self.image_panel.grid(row=4)    

    def video_capture_thread(self):
        """
        Handle Tkinter Thread to capture drone video stream.
        """
        try:
            #Sleep to allow screen update
            time.sleep(0.5)
            
            while self.tello.stream_on == True:                
                system = platform.system()

                # read frame from tello stream
                frame = self.tello.get_frame_read().frame
                if frame is None or frame.size == 0:
                    continue 
            
                # Convert the format from frame to image         
                image = Image.fromarray(frame)

                #Call Object Detector 
                detected_people = self.object_detector.detect_people(image) 

                #Draw People Bounding Boxes
                image = self.object_detector.draw_bounding_boxes(image, detected_people, "Green", 1)


                #Update Image UI Component with image captured
                if system =="Windows" or system =="Linux":                
                    self.update_GUI_image(image)
                else: # Work around for MacOS
                    thread_tmp = threading.Thread(target=self.update_GUI_image,args=(image,))
                    thread_tmp.start()
                    time.sleep(0.03)

                #Initialize Thread to Move Drone To keep people at the center
                thread_movement = threading.Thread(target=self.move_drone_thread,args=(detected_people,))
                thread_movement.start()
                                                                                                
        except:
            print("[INFO] RuntimeError on i")
            raise


    def move_drone_thread(self, detected_people):

        try:

            if self.is_flying == False: #cant move the drone if not flying
                return 

            person_idx = 0
            interested_class = [0]
            img_shape = detected_people.pandas().imgs[0].shape
            img_xcenter = img_shape[1]/2
            img_ycenter = img_shape[0]/2
            df_xywh = detected_people.pandas().xywh[0]
            df_persons_xywh = df_xywh[(df_xywh['class'].isin(interested_class)) & (df_xywh['confidence'] > self.detection_threshold)]
            
            if not df_persons_xywh.empty:
                time.sleep(0.1)
                if df_persons_xywh['xcenter'][person_idx] > img_xcenter:
                    # Need to update distance 
                    # self.update_distance()
                    # for now moving 50 cm, need to calcualte the cm using pixel
                    self.tello.rotate_clockwise(30)
                    
                    #self.tello.move_right(50)
                elif df_persons_xywh['xcenter'][person_idx] < img_xcenter:

                    self.tello.rotate_counter_clockwise(30)
                        
                elif df_persons_xywh['ycenter'][person_idx] > img_ycenter:
                    print('move up')
                    self.tello.move_up(50)
                elif df_persons_xywh['ycenter'][person_idx] < img_ycenter:
                    print('move down')
                    self.tello.move_down(50)
        except:
            print("[INFO] Unable to move drone to center people.")                    

    def update_GUI_image(self, image):
        """
        Show Image on the GUI
        """ 
        try:
            image = ImageTk.PhotoImage(image)
            self.image_panel = Label(self.root, image = image, width=600, height=400)
            self.image_panel.image = image
            self.image_panel.grid(row=4) 
        except:
            print("[INFO] Error updating frame.")

    def on_close(self):
        """
        Handle GUI Terminating event
        """
        print("[INFO] closing...")
        del self.tello
        del self.object_detector
        self.root.quit()
