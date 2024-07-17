
 # LogoDetectionModel

# Setting Up the Environment and Training the Model

## Setting Up

To build and train the model using Python, we'll leverage its powerful libraries and predefined functions for ease of calculation. Follow these steps to set up your environment:

### Create a Virtual Environment

This helps prevent conflicts with other projects on your system.

#### For Windows:
#### Python
```bash
python -m venv yolov8_env
yolov8_env\Scripts\activate
```
#### For Linux or macOS:

```bash
python3 -m venv yolov8_env
source yolov8_env/bin/activate
```
* Install Required Libraries
Download the ultralytics library, which includes YOLOv8, along with OpenCV for image and video processing.

```bash
pip install ultralytics opencv-python-headless
```
Import Libraries
Import the necessary libraries for model processing and image handling.

#### Python
```bash
from ultralytics import YOLO
import cv2
import json
import os
```
Ensure Proper Path to the Model
**Verify the correct path to your model before running the Python script**.

### Machine Training
Training your model on a GPU is crucial due to its heavy computational demands. Using platforms like Google Colab can provide the necessary resources.

#### Initialize GPU
Run the following command to check if the GPU is available:

#### Python
```bash
!nvidia-smi
```
This command displays your GPU status and utilization.

Install the ultralytics library if you haven't already:
#### Python
```bash
!pip install ultralytics
```
Load and Train Your Model

##### Install Your Dataset
In this example, we'll use a dataset from Roboflow.
#### Python
```bash
!pip install roboflow
```

#### Python
```bash
from roboflow import Roboflow
rf = Roboflow(api_key="YOUR_API_KEY")
project = rf.workspace("WORKSPACE_NAME").project("PROJECT_NAME")
version = project.version(1)
dataset = version.download("yolov8")
```
* Ensure your dataset contains train, test, and valid folders, each with a labels folder. Verify that none of the label files are empty, or you may need to update your dataset.

**Train the Model**
Use the following command to start training your model. This example uses the YOLOv8 medium model and sets the number of epochs to 150 with an image size of 640.
#### Python
```bash
!yolo task=detect mode=train model=yolov8m.pt data=/path/to/your/data.yaml epochs=150 imgsz=640
```
* Task: Specifies that you are doing object detection i.e train , predict , valid.
* Mode: Indicates that you are in training mode.
* Model: Specifies the model you are using (e.g., yolov8m.pt for the medium version).
* Data: Path to your dataset configuration file (data.yaml).
* Epochs: Number of training iterations (higher epochs generally result in a better-trained model).
* Imgsz: The size of the images used for training.

### The more images and higher epochs you use, the better the accuracy of your model.

### Approach Document: [Link](https://docs.google.com/document/d/1KQSaetpCBsrTuot5i2vIA7VLtMClUB-fdLqIiRkPGPE/edit?usp=sharing)
### Final Video : [Link](https://www.loom.com/share/b5877b23b24c466fbdbe1a3109b28fbd?sid=15a783a0-b8bd-4672-b984-5869584b2a73)


