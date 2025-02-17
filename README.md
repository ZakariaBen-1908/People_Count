# People Counting and Tracking System

This project is a people counting and tracking system that uses YOLOv8 for object detection and OpenCV for video processing. It detects and tracks people in a video feed, counts unique individuals within a predefined polygon region, and provides visualizations, including a simulated infrared map.

## Features
People Detection: Uses YOLOv8 to detect people in a video.

### Tracking: Assigns unique IDs to detected people and tracks their movements.

Region of Interest (ROI) Counting: Counts and tracks people who enter a predefined polygon region.

## Visualization:

Displays the original video with bounding boxes, IDs, and the polygon.

Simulates an infrared map for additional visualization.

Unique People Count: Keeps a count of unique individuals who have entered the polygon.

## Requirements

To run this project, you need the following:

Python 3.7 or higher

OpenCV (opencv-python)

Ultralytics (ultralytics)

NumPy (numpy)

You can install the required libraries using pip:

pip install opencv-python ultralytics numpy

## Setup

### Clone the Repository:

git clone https://github.com/ZakariaBen-1908/people-counting-system.git

cd people-counting

Download the YOLOv8 Model:

The YOLOv8 model (yolov8n.pt) is automatically downloaded by the ultralytics library when you run the script for the first time.

## Usage

### Run the Script:

python people_counting.py

### Interact with the Program:

The program will display two windows:

Original Frame: Shows the video with bounding boxes, IDs, and the polygon.

Infrared Map: Simulates an infrared visualization of the video.

Press q to exit the program.

### Customize the Polygon:

Update the polygon_points variable in the script to define your region of interest (ROI). The polygon is defined as a list of [x, y] coordinates.

## Code Overview

### Key Components

### YOLOv8 Model:

The YOLOv8 model (yolov8n.pt) is used for detecting people in the video.

### Polygon Region:

A polygon is defined using four points. The program counts and tracks people who enter this region.

### Tracking:

Unique IDs are assigned to detected people using uuid.uuid4().

The program checks if a person was previously detected by comparing their current position with stored positions.

## Visualization:

Bounding boxes, IDs, and the polygon are drawn on the original frame.

A simulated infrared map is created using OpenCV's color mapping.

## Counting:

The program keeps a count of unique individuals who have entered the polygon.

Simulates an infrared visualization of the video.

## Customization

Change the Video Source:

Update the video_path variable to point to your video file or use a webcam by setting video_path = 0.

## Modify the Polygon:

Adjust the polygon_points variable to define your region of interest.

## Add Additional Classes:

Modify the script to detect and track other objects (e.g., vehicles) by updating the class filter in the detection loop.

## Acknowledgments

YOLOv8: Ultralytics for the YOLOv8 model.

OpenCV: For video processing and visualization.
