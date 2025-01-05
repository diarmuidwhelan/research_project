# research_project


Computer Vision System to generate performance data from Gaelic Football Footage

1. Training Libraries
Scripts for training models to use in the overall system. Training visulaisations and files included.
     - Object Detection: YOLO model that detects players, ball, goalkeepers, referees and umpires
     - Player identification: ResNet models that perform player team classification and number recognition.
     - Action Recognition: Model that detects one of 4 events from video footage
2. Resources
Images and supplementary files for research write up

3. Training Data
Sample data outputted from the system for the purposes of refining models and developing new solutions.

4. Tracking Module
POC tracking script that processes videos with trained models extending detection to tracking. (Not included in research or system) 

Main system comprises the videogui opticalflow.py file and the calibrator.py files.
videogui opticalflow.py is run first and subsequently calls the calibrator when a valid event is identified. 
Required to select team colours and load the video into the system. 
