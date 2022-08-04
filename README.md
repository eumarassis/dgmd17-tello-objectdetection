# dgmd17-tello-objectdetection
Final project by Ryan, Muthu, and Eumar. Object Detection with Yolo, Azure, and Tello Drone

Step to run the project locally:
 1. Clone the repo locally

 2. (Optional) Create a [Python Enviroment](https://www.tutorialspoint.com/how-to-create-a-virtual-environment-in-python)

 3. Install the dependencies using pip install -r requirements.txt
 
 4. If you have the Tello Drone
        * Connect the Tello Drone to the computer via wifi
        * Run `python app.py`
        * Once the app starts click `connect` button in the app to connect the app with Drone
        * Click `Start Streaming` to start the video feeds
        * Pick the model from the drop down to the one you would like to use
        * Click `Take Off` button to lift the drone and drone should move based on the out put from the model

5. If you do not have the drone you can use our test app
        * Run `python app_test.py`
        * We have inluded sample image for the models, you can change the images with people in here
        * The models output will be saved to output dir
        Cost volumes are commonly used for estimating depths from multiple input views:

#### Model Output location
<p align="center">
  <img src="output/test_output.png" alt="Model outputs saved here" width="400" />
</p>

6. Code Strucutre
        * All the models are in `ai` folder
        * all the model configurations and weights are in `assets` folder
        * Telemetry data used for the EDA is in [assets]('./assets/telemetry_v2.csv')
        * The python notebooks used for the EDA is `ProjectEDA.ipynb`