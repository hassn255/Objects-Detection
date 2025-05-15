# Real-Time Object Detection with OpenCV and SSD MobileNet v2

This project performs real-time object detection using a webcam feed and a pre-trained SSD MobileNet v2 model from TensorFlow's model zoo. It uses OpenCV’s DNN module for inference and displays bounding boxes with labels and confidence scores.

---

## Features

- Real-time object detection via webcam
- Uses TensorFlow SSD MobileNet v2 trained on the COCO dataset
- Resizes frames for better performance
- Labels detected objects with confidence scores
- Clean bounding box drawing with label overlays

---

## Requirements

- Python 3.x
- OpenCV (`opencv-python`)
- TensorFlow model files (downloaded from the model zoo):
  - `frozen_inference_graph.pb`
  - `ssd_mobilenet_v2_coco_2018_03_29.pbtxt`

---

## How to Run
Run the script:

``` bash
python modified.py
```
Exit preview by pressing the ESC key.

---

## How It Works
- Captures video from webcam

- Resizes frame to 300x300 (model input size)

- Performs object detection on resized frame

- Scales detection results to fit original frame size

- Displays bounding boxes and class labels with confidence scores