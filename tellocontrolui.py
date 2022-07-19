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


#Initialize UI Componenets
root = Tk()
frm = ttk.Frame(root, padding=10)
frm.grid()


root.wm_title("Tello Control App")

#Initialize Tello Componenet
tello = Tello()

#Initialize Thread
stopEvent = None
thread = None

#Initialize image
image_panel = None

#Set default move distance and degree

distance = 20
degree = 30

def connect_handler ():
    tello.connect()

    ttk.Label(frm, text="Battery: {}%".format( tello.get_battery())).grid(column=6, row=0)

def takeoff_handler ():
    tello.takeoff()

def land_handler ():
    tello.land()

def start_streaming_handler ():
    tello.streamon()
    thread = threading.Thread(target=video_capture_thread, args=())
    thread.start()


def stop_streaming_handler ():
    tello.streamoff()   

def update_distance(event):
    distance = int (event) 

def update_degree(event):
    degree = int (event)


def on_keypress_a(event):
    tello.move_up(distance)

def on_keypress_z(event):
    tello.move_down(distance)

def on_keypress_s(event):
    tello.rotate_counter_clockwise(degree)

def on_keypress_d( event):
    tello.rotate_clockwise(degree)

def on_keypress_up( event):
    tello.move_forward(distance)

def on_keypress_down(event):
    tello.move_back(distance)

def on_keypress_left( event):
    tello.move_left(distance)

def on_keypress_right( event):

    tello.move_right(distance)



def build_ui():
    ttk.Button(frm, text="Connect", command=connect_handler).grid(row=0, column=1)
    ttk.Button(frm, text="Take Off", command=takeoff_handler).grid(row=0, column=2)
    ttk.Button(frm, text="Land", command=land_handler).grid(row=0, column=3)    
    ttk.Button(frm, text="Start Streaming", command=start_streaming_handler).grid(row=0, column=4 )    
    ttk.Button(frm, text="Stop Streaming", command=stop_streaming_handler).grid(row=0, column=5 )    

    ttk.Label(frm, text=
                          'A - Move Tello Up\n'
                          'Z - Move Tello Down\n'
                          'S - Rotate Tello Counter-Clockwise\n'
                          'D - Rotate Tello Clockwise').grid(row=2, column=1 )  

    ttk.Label(frm, text=
                          'Arrow Up - Move Tello Forward\n'
                          'Arrow Down - Move Tello Backward\n'
                          'Arrow Left - Move Tello Left\n'
                          'Arrow Right - Move Tello Right').grid(row=2, column=2 )  


    Scale(frm, from_=20, to=50, tickinterval=10, label='Distance(cm)', command=update_distance).grid( row=2, column=4)

    Scale(frm, from_=30, to=360, tickinterval=10, label='Degree', command=update_degree).grid(row=2, column=5)


    root.bind('<KeyPress-a>', on_keypress_a)
    root.bind('<KeyPress-z>', on_keypress_z)
    root.bind('<KeyPress-s>', on_keypress_s)
    root.bind('<KeyPress-d>', on_keypress_d)
    root.bind('<KeyPress-Up>', on_keypress_up)
    root.bind('<KeyPress-Down>', on_keypress_down)
    root.bind('<KeyPress-Left>', on_keypress_left)
    root.bind('<KeyPress-Right>', on_keypress_right)
    
    image = Image.open("tello-drone.jpg")
    photo = ImageTk.PhotoImage(image)

    image_panel = Label(root, image = photo, width=600, height=400)
    image_panel.image = photo
    image_panel.grid(row=4)    

def video_capture_thread():
    """
    The mainloop thread of Tkinter 
    Raises:
        RuntimeError: To get around a RunTime error that Tkinter throws due to threading.
    """
    try:
        # start the thread that get GUI image and drwa skeleton 
        time.sleep(0.5)
        
        while tello.stream_on == True:                
            system = platform.system()


        # read the frame for GUI show
            frame = tello.get_frame_read().frame
            if frame is None or frame.size == 0:
                continue 
        
        # transfer the format from frame to image         
            #image = Image.fromarray(frame)

            image = Image.fromarray(frame)

        #TODO: Add Integration with Yolo - Identify Bounding Boxes
        #TODO: Paint Pictures

        # we found compatibility problem between Tkinter,PIL and Macos,and it will 
        # sometimes result the very long preriod of the "ImageTk.PhotoImage" function,
        # so for Macos,we start a new thread to execute the _updateGUIImage function.
            if system =="Windows" or system =="Linux":                
                update_GUI_image(image)

            else:
                thread_tmp = threading.Thread(target=update_GUI_image,args=(image,))
                thread_tmp.start()
                time.sleep(0.03)                                                            
    except:
        print("[INFO] caught a RuntimeError")
        raise
def update_GUI_image(image):
    """
    Main operation to initial the object of image,and update the GUI panel 
    """ 


    #image = ttk.PhotoImage(image)
    # if the panel none ,we need to initial it
  
    image = ImageTk.PhotoImage(image)
    image_panel = Label(root, image = image, width=800, height=600)
    image_panel.image = image
    image_panel.grid(row=4)    
    # # otherwise, simply update the panel
    # else:
    #     image_panel.configure(image=image)
    #     image_panel.image = image


def on_close():
    """
    set the stop event, cleanup the camera, and allow the rest of

    the quit process to continue
    """
    print("[INFO] closing...")
    #del tello
    root.quit()


root.wm_protocol("WM_DELETE_WINDOW", on_close)

build_ui()

root.mainloop()




