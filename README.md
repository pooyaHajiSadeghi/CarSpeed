# 🚗 **Real-time Vehicle Speed Estimation** 🏎️

This project uses **YOLOv8** for real-time vehicle detection and tracking in video streams, estimating their speed based on their movement. This is ideal for road video analysis and simulating **vehicle speed** based on YOLO-detected data.

---

## 🚀 **Features**:

- **Vehicle Detection**: Uses the YOLOv8 model for real-time vehicle detection.
- **Speed Estimation**: Calculates the speed of detected vehicles based on tracking data and defined region points.
- **Interactive Mouse Input**: Displays mouse coordinates for debugging purposes.
- **Video Output**: Shows detected bounding boxes and vehicle speed in the video stream.

---

## 📦 **Installation Requirements**:

To run this project, you'll need the following packages:

- `opencv-python` for video processing and displaying frames.
- `ultralytics` for YOLOv8 model inference.
- `numpy` for mathematical operations.

You can install all the required packages at once using the `requirements.txt` file.


![Output GIF](output.gif)
