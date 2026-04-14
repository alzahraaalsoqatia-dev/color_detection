# Color Detection using OpenCV 

## Description

This project demonstrates real-time color detection using computer vision techniques. It captures video from a webcam, detects objects based on their color, and highlights them with bounding boxes.

The project includes:

* A basic yellow color detection model (inspired by a tutorial)
* An extended version that detects multiple colors and labels them dynamically

---

## Project Motivation

This project was developed as part of my learning journey in computer vision. I first implemented a basic color detection system by following an online tutorial, then expanded it to build a more advanced and flexible color detection model.

---

## Technologies Used

* Python
* OpenCV
* NumPy
* PIL (Python Imaging Library)

---

## Features

### 1. Basic Color Detection (Tutorial-Based)

* Detects yellow objects in real-time
* Uses HSV color space masking
* Draws bounding boxes around detected objects

### 2. Advanced Color Detection (My Implementation)

* Detects multiple colors dynamically
* Converts HSV values into human-readable color names
* Uses contour detection and filtering
* Applies background subtraction to improve detection
* Displays labeled bounding boxes with detected color names

---

## Requirements

All required libraries are listed in the requirements.txt file:

opencv-python==4.8.1.78
numpy==1.26.0
pillow==10.0.1

---

## How to Run

1. Install dependencies:

```
pip install -r requirements.txt
```

2. Run the basic version:

```
python yellowdetection.py
```

3. Run the advanced version:

```
python colordetection.py
```

---

## Input

* Live webcam feed

---

## Output

* Real-time video with detected objects highlighted and labeled by color

---

## Acknowledgment

The basic yellow color detection implementation was inspired by the following YouTube tutorial:
https://www.youtube.com/watch?v=aFNDh5k3SjU

The advanced color detection system was independently developed by extending the concepts learned from the tutorial.

---

## Future Improvements

* Improve accuracy using machine learning models
* Detect multiple objects simultaneously
* Add a GUI interface
* Optimize performance for real-time applications

---

## Notes

* Press **'q'** to exit the program
* Lighting conditions may affect detection accuracy
