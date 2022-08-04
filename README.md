# dgmd17-tello-objectdetection
Final project by Ryan, Muthu, and Eumar. Object Detection with Yolo, Azure, and Tello Drone

## Step to run the project locally:

The app will be launched even if a drone is not connected to the WIFI.

 1. Clone the repo locally

 2. Create a [Python Enviroment](https://www.tutorialspoint.com/how-to-create-a-virtual-environment-in-python)
    ```
    python -m venv .venv & ./venv/Scripts/activate)
    ```
 3. Install the dependencies using pip
    ```
    pip install -r requirements.txt
    ```
 4. Run python app.py
    ```
    python app.py
    ```

## Project Structure


File | Description
------ | ------
./app.py | Entry point for the application. Instatiates UI and model classes to start the application
./tello_ui.py | Tello Control User Interface (UI) Class. Create the UI using tkinter
./ai folder   | Contains class for each model
-- /object_detector.py   | Base class for object detector
-- /azure_object_detector.py   | Azure object detector class using azure-cognitiveservices-vision-computervision library
-- /yolo_face_detector.py   | YOLO face detector class using custom weights
-- /yolo_object_detector.py   | YOLO object detector class using the ultralytics/yolov5 library
-- /depth_perception.py   | Depth Perception class using manydepth library
./assets folder   | Contains custom weights for YOLO face detection and holds model telemetry data
./EDA   | Python Notebooks for Exploratory Data Analysis
-- /model_analytics.ipynb   | Python Notebook for Exploratory Data Analysis by Ryan
-- /ProjectEDA.ipynb   | Python Notebook for Exploratory Data Analysis by Eumar and Muthu
./requirements.txt | Contains list of required python libraries
