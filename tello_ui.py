### dgmd17-tello-objectdetection - TelloUI ###

from multiprocessing.dummy import Array
from tkinter import *
from tkinter import ttk
from tkinter.scrolledtext import ScrolledText
from turtle import width
from PIL import ImageTk, Image
from djitellopy import Tello
import threading
import datetime
import cv2
import os
import time
import platform
import csv
import numpy as np
from typing import List, Set, Dict, Tuple, Optional

from torch import initial_seed
from ai.azure_object_detector import AzureObjectDetector
from ai.depth_perception import DepthPerceptionObjectDetector
from ai.object_detector import ObjectDetector
from ai.yolo_face_detector import YOLOFaceDetector
from ai.yolo_object_detector import YOLOObjectDetector

class TelloControlUI:
    """Tello Control User Interface Class"""

    def __init__(self,tello : Tello, list_object_detector : Array ):
        print("INFO: Screen Class Initialized")

        #Initialize UI Componenets
        self.root = Tk()
        self.frm = ttk.Frame(self.root, padding=20)
        self.frm.grid()
        self.root.wm_title("Tello Control App")

        #Set Variables
        self.tello = tello

        self.list_object_detector = list_object_detector
        self.object_detector = list_object_detector[0][1]

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
        self.last_move = None
        self.last_frame = None
        # dept perception model
        # taotal intensity to back
        self.move_back_threshold = 270000000
        # taotal intensity to front
        self.move_front_threshold = 200000000

        #Subscribe to Window Close Event
        self.root.wm_protocol("WM_DELETE_WINDOW", self.on_close)

        self.log_combined_model_telemetry = False

        #Create Logging file
        self.telemetry_file = open('assets/telemetry.csv', 'a', newline='')

        self.telemetry_writer = csv.writer(self.telemetry_file)

    
    def connect_handler (self):
        """Handle click to Connect. Connect to Tello Using Tello API."""
        
        if self.is_connected == True:
            return
            
        self.tello.connect()

        self.log_ui_msg("Connected successfully. Battery: {}%".format( self.tello.get_battery()))

        self.is_connected = True

        #self.btn_connect['state'] = "disabled"
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

    def update_detector(self, event):
        """Update the object detector based on the selection"""

        self.object_detector = [x[1] for x in self.list_object_detector if x[0] == event ][0]

    def log_ui_msg(self, message, new_line = True):
        """Log message in the UI"""
        self.text_log.insert('1.0', message + "\n" if new_line == True else message )

    def handle_telemetry_checkbox(self):
        self.log_combined_model_telemetry = True if self.chk_telemetry.get() == 1 else False

    def build_ui(self):
        """Build drone control UI using tkinter objects."""        


        #Drone Control Buttons
        self.btn_connect = ttk.Button(self.frm, text="Connect", command=self.connect_handler)
        self.btn_connect.grid(row=1, column=1)
        self.btn_takeoff = ttk.Button(self.frm, text="Take Off", command=self.takeoff_handler, state = DISABLED)
        self.btn_takeoff.grid(row=1, column=2)
        self.btn_land = ttk.Button(self.frm, text="Land", command=self.land_handler, state = DISABLED)
        self.btn_land.grid(row=1, column=3)    
        self.btn_streamon = ttk.Button(self.frm, text="Start Streaming", command=self.start_streaming_handler, state = DISABLED)
        self.btn_streamon.grid(row=1, column=4 )    
        self.btn_streamoff = ttk.Button(self.frm, text="Stop Streaming", command=self.stop_streaming_handler, state = DISABLED)
        self.btn_streamoff.grid(row=1, column=5 )    

        #Drone Movements Labels
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

        #Bind Keyboard key to event handlers
        self.root.bind('<KeyPress-a>', self.on_keypress_a)
        self.root.bind('<KeyPress-z>', self.on_keypress_z)
        self.root.bind('<KeyPress-s>', self.on_keypress_s)
        self.root.bind('<KeyPress-d>', self.on_keypress_d)
        self.root.bind('<KeyPress-Up>', self.on_keypress_up)
        self.root.bind('<KeyPress-Down>', self.on_keypress_down)
        self.root.bind('<KeyPress-Left>', self.on_keypress_left)
        self.root.bind('<KeyPress-Right>', self.on_keypress_right)
        
        #Build List of Object Detection Classess
        OPTIONS = [x[0] for x in self.list_object_detector]

        variable = StringVar(self.root)
        variable.set(OPTIONS[0]) # default value

        OptionMenu(self.frm, variable, *OPTIONS, command=self.update_detector).grid(row=3, column=2)

        self.chk_telemetry = IntVar()
        ttk.Checkbutton(self.frm, text='Log Combined Model Telemetry',variable=self.chk_telemetry, onvalue=1, offvalue=0, command=self.handle_telemetry_checkbox).grid(row=3, column=4) 

        #Create Logging Section
        self.text_log = ScrolledText(self.root, width=80,  height=3)
        self.text_log.grid(row=4)

        #Show Image
        image = Image.open("tello-drone.jpg")
        photo = ImageTk.PhotoImage(image)
        self.image_panel = Label(self.root, image = photo, width=600, height=400)
        self.image_panel.image = photo
        self.image_panel.grid(row=5) 

        self.log_ui_msg("=== INFO: Screen Class Initialized === ", False)

    def video_capture_thread(self):
        """
        Handle Tkinter Thread to capture drone video stream.
        """
        try:
            while self.tello.stream_on == True:                
                system = platform.system()

                # Read frame from tello stream
                frame = self.tello.get_frame_read().frame
                if frame is None or frame.size == 0:
                    continue

            
                # Convert the format from frame to image         
                image = Image.fromarray(frame)

                if self.last_frame != image:
                    self.last_frame = image


                #Capture Telemetry
                self.log_all_models_telemetry(image, previous_image=self.last_frame)

                #Call Object Detector Class
                detected_people = self.object_detector.detect_people(image, previous_image=self.last_frame) 

                #Draw People Bounding Boxes
                image = self.object_detector.draw_bounding_boxes(image, detected_people, previous_image=self.last_frame)

                #Update Image UI Component with image captured
                if system =="Windows" or system =="Linux":                
                    self.update_GUI_image(image)
                else: # Work around for MacOS
                    thread_tmp = threading.Thread(target=self.update_GUI_image,args=(image,))
                    thread_tmp.start()
                    time.sleep(0.03)
                
                #Initialize Thread to Move Drone To keep people at the center
                thread_movement = threading.Thread(target=self.move_drone_thread,args=(detected_people,image))
                thread_movement.start()

                
                                                                                                
        except:
            print("[INFO] RuntimeError on i")
            raise

    def log_all_models_telemetry(self, image, previous_image):
        """
        Log Model Telemetry and Comparison to CSV File
        Save one row for each model output with the following schema:
        log timestamp, model name, confidence, img xcenter, img ycenter, model output xcenter, model output ycenter, height (distance to floor), is drone flying
        """
        try:

            if self.log_combined_model_telemetry == False:
                return
                
            log_time = datetime.datetime.now()
            logs = list()
            img_xcenter = image.width/2
            img_ycenter = image.height/2
            height = self.tello.get_height()


            for item in self.list_object_detector:
                model = item[1]

                result = model.detect_people(image, previous_image=self.last_frame)


                if isinstance(model, DepthPerceptionObjectDetector):

                    depth_image = model.draw_bounding_boxes(image=image, bounding_boxes= result, previous_image=previous_image)
                    depth_array = np.array(depth_image)
                    total_intensity = np.sum(np.sum(np.sum(depth_array, axis = 2),axis =0))

                    logs.append([log_time, "DepthPerceptionObjectDetector", total_intensity, img_xcenter, img_ycenter, np.nan, np.nan, height, self.is_flying ])

                elif isinstance(model, YOLOObjectDetector):
                   
                    interested_class = [0]
                    
                    df_xywh = result.pandas().xywh[0]
                    df_persons_xywh = df_xywh[(df_xywh['class'].isin(interested_class)) & (df_xywh['confidence'] > self.detection_threshold)]

                    xcenter = df_persons_xywh['xcenter'][0] if df_persons_xywh['xcenter'].empty == False else np.nan
                    ycenter = df_persons_xywh['ycenter'][0] if df_persons_xywh['ycenter'].empty == False else np.nan
                    confidence =df_persons_xywh['confidence'][0] if df_persons_xywh['confidence'].empty == False else np.nan
                    
                    logs.append([log_time,"YOLOObjectDetector", confidence, img_xcenter, img_ycenter, xcenter, ycenter, height, self.is_flying ])
                            
                elif isinstance(model, YOLOFaceDetector):

                    xcenter = np.nan
                    ycenter = np.nan
                    confidence = np.nan
                    
                    if len(result) > 0:
                        xcenter = result[0][0][0] / 2
                        ycenter = result[0][0][1] / 2
                        confidence = result[0][1]
                    
                    logs.append([log_time,"YOLOFaceDetector", confidence, img_xcenter, img_ycenter, xcenter, ycenter, height, self.is_flying ])

                elif isinstance(model, AzureObjectDetector):

                    xcenter = np.nan
                    ycenter = np.nan
                    confidence = np.nan
                    
                    if len(result) > 0:
                        xcenter = result[0][0].x / 2
                        ycenter = result[0][0].y / 2
                        confidence = result[0][1]
                    
                    logs.append([log_time,"AzureObjectDetector", confidence, img_xcenter, img_ycenter, xcenter, ycenter, height, self.is_flying ])

            self.telemetry_writer.writerows(logs)
        except Exception as e:
            print("[Error logging telemetry]:", e)                  

    def move_drone_thread(self, detected_people, image):

        try:

            if self.is_flying == False: #cant move the drone if not flying
               return
    
            if self.last_move != None:
                if (datetime.datetime.now() - self.last_move).seconds < 5:
                    return

            # print('[Check Moving]: Trying to move if person is there at', datetime.datetime.now())
            if isinstance(self.object_detector, DepthPerceptionObjectDetector):
                depth_array = np.array(image)
                total_intensity = np.sum(np.sum(np.sum(depth_array, axis = 2),axis =0))
                if total_intensity > self.move_back_threshold:
                    print(' [Moving]: Back by 20cm the intensity is: {0}'.format(total_intensity))
                    self.log_ui_msg(' [Moving]: Back by 20cm the intensity is: {0}'.format(total_intensity), True)
                    self.tello.move_back(20)
                    # self.tello.move_up(20)
                    self.last_move = datetime.datetime.now()
                elif total_intensity < self.move_front_threshold:
                    print(' [Moving]: Front by 10cm the intensity is: {0}'.format(total_intensity))
                    self.log_ui_msg(' [Moving]: Front by 10cm the intensity is: {0}'.format(total_intensity), True)
                    self.tello.move_forward(10)
                    self.last_move = datetime.datetime.now()

            elif isinstance(self.object_detector, YOLOObjectDetector):
                person_idx = 0
                interested_class = [0]
                img_shape = detected_people.pandas().imgs[0].shape
                img_xcenter = img_shape[1]/2
                img_ycenter = img_shape[0]/2
                df_xywh = detected_people.pandas().xywh[0]
                df_persons_xywh = df_xywh[(df_xywh['class'].isin(interested_class)) & (df_xywh['confidence'] > self.detection_threshold)]
                
                if not df_persons_xywh.empty:
                    self.log_ui_msg('[Identified Person]: Identified a person at {}'.format(datetime.datetime.now()))                
                    
                    if df_persons_xywh['xcenter'][person_idx] > img_xcenter:
                        self.log_ui_msg(' [Moving]: Right by rotating clockwise 30', False)
                        self.tello.rotate_clockwise(30)
                        self.last_move = datetime.datetime.now()
                        #self.tello.move_right(50)
                    elif df_persons_xywh['xcenter'][person_idx] < img_xcenter:
                        self.log_ui_msg(' [Moving]: Left by rotating counter clockwise 30', False)
                        self.tello.rotate_counter_clockwise(30)
                        self.last_move = datetime.datetime.now()
                    elif df_persons_xywh['ycenter'][person_idx] > img_ycenter:
                        self.log_ui_msg(' [Moving]: Move up', False)
                        self.tello.move_up(50)
                        self.last_move = datetime.datetime.now()
                    elif df_persons_xywh['ycenter'][person_idx] < img_ycenter:
                        self.log_ui_msg(' [Moving]: Move Down', False)
                        self.tello.move_down(50)
                        self.last_move = datetime.datetime.now()
                        
            elif isinstance(self.object_detector, YOLOFaceDetector):

                img_xcenter = image.width/2
                img_ycenter = image.height/2

                #Find faces that meet the confidence level
                faces_filtered_threshold = [ x[0] for x in detected_people if x[1] > self.detection_threshold ]
                
                if len(faces_filtered_threshold) > 0:
                    self.log_ui_msg('[Identified Face]: Identified a person at {}'.format(datetime.datetime.now()))                

                    #Tuple Structure = x,y,w,h                    
                    face_tuple = faces_filtered_threshold[0]

                    xcenter = face_tuple[0] / 2
                    ycenter = face_tuple[1] / 2

                    if  xcenter > img_xcenter:
                        self.log_ui_msg(' [Moving]: Right by rotating clockwise 30', False)
                        self.tello.rotate_clockwise(30)
                        self.last_move = datetime.datetime.now()
                        #self.tello.move_right(50)
                    elif xcenter < img_xcenter:
                        self.log_ui_msg(' [Moving]: Left by rotating counter clockwise 30', False)
                        self.tello.rotate_counter_clockwise(30)
                        self.last_move = datetime.datetime.now()
                    elif ycenter > img_ycenter:
                        self.log_ui_msg(' [Moving]: Move up', False)
                        self.tello.move_up(50)
                        self.last_move = datetime.datetime.now()
                    elif ycenter < img_ycenter:
                        self.log_ui_msg(' [Moving]: Move Down', False)
                        self.tello.move_down(50)
                        self.last_move = datetime.datetime.now()

            elif isinstance(self.object_detector, AzureObjectDetector):

                img_xcenter = image.width/2
                img_ycenter = image.height/2

                #Find faces that meet the confidence level
                faces_filtered_threshold = [ x[0] for x in detected_people if x[1] > self.detection_threshold ]
                
                if len(faces_filtered_threshold) > 0:
                    self.log_ui_msg('[Identified Face]: Identified a person at {}'.format(datetime.datetime.now()))                

                    #Tuple Structure = x,y,w,h                    
                    face_tuple = faces_filtered_threshold[0]

                    xcenter = face_tuple.x / 2
                    ycenter = face_tuple.y / 2

                    if  xcenter > img_xcenter:
                        self.log_ui_msg(' [Moving]: Right by rotating clockwise 30', False)
                        self.tello.rotate_clockwise(30)
                        self.last_move = datetime.datetime.now()
                        #self.tello.move_right(50)
                    elif xcenter < img_xcenter:
                        self.log_ui_msg(' [Moving]: Left by rotating counter clockwise 30', False)
                        self.tello.rotate_counter_clockwise(30)
                        self.last_move = datetime.datetime.now()
                    elif ycenter > img_ycenter:
                        self.log_ui_msg(' [Moving]: Move up', False)
                        self.tello.move_up(50)
                        self.last_move = datetime.datetime.now()
                    elif ycenter < img_ycenter:
                        self.log_ui_msg(' [Moving]: Move Down', False)
                        self.tello.move_down(50)
                        self.last_move = datetime.datetime.now()
        except Exception as e:
            print("[Error]:", e)                    


    def update_GUI_image(self, image):
        """
        Show Image on the GUI
        """ 
        try:
            image = ImageTk.PhotoImage(image)
            self.image_panel = Label(self.root, image = image, width=600, height=400)
            self.image_panel.image = image
            self.image_panel.grid(row=5) 
        except:
            print("[INFO] Error updating frame.")


    def on_close(self):
        """
        Handle GUI Terminating event
        """
        print("[INFO] closing...")

        del self.telemetry_writer
        del self.telemetry_file
        del self.tello
        del self.object_detector
        self.root.quit()
