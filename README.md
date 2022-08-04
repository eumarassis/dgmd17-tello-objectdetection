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
 4. If you have the Tello Drone
        * Connect the Tello Drone to the computer via wifi
        * Run python app.py
                ```python app.py```
        * Once the app starts click `connect` button in the app to connect the app with Drone
        * Click `Start Streaming` to start the video feeds
        * Pick the model from the drop down to the one you would like to use
        * Click `Take Off` button to lift the drone and drone should move based on the out put from the model
5. If you do not have the drone you can use our test app
        * Run python app_test.py
                ```python app_test.py```
        * We have inluded sample image for the models, you can change the images with people in here
        * The models output will be saved to output dir
        Cost volumes are commonly used for estimating depths from multiple input views:

#### Model Output location
<p align="center">
  <img src="output/test_output.png" alt="Model outputs saved here" width="400" />
</p>

6. How to use the app [App Video Demo](https://youtu.be/LKzUzrd4MzM)

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
>>>>>>> 5d665b6d812055825c80cb24a01d5d98cee1b651
