# Pothole Detection with YOLOv8n on Custom Dataset

A pothole detection project using YOLOv8n (nano version) for object detection in images and videos.

## Demo

[![Pothole Detection Demo](https://img.youtube.com/vi/Nw79lFp2yAM/0.jpg)](https://youtu.be/Nw79lFp2yAM)

*Click the image above to watch the demo video*

## Project Overview

This project trains a YOLOv8n model to detect potholes in road videos. The model is trained on a pothole dataset from Kaggle and can process videos to identify and mark potholes with bounding boxes.

## Project Structure

```
├── download_dataset.py      # Script to download the pothole dataset from Kaggle
├── train.py                  # Script to train the YOLO model
├── predict.py                # Script to run inference on videos
├── potholes-dataset/         # Dataset directory (downloaded via script)
│   ├── data.yaml            # Dataset configuration file
│   ├── train/               # Training images and labels
│   └── valid/               # Validation images and labels
├── runs/                     # Training outputs and model weights
└── predicted_sample_video.mp4 # Sample output video with predictions
```

## Installation

**Prerequisites:**
- Python 3.8+
- GPU recommended for training

**Setup:**

1. Clone the repository:
```bash
git clone https://github.com/BlazeWild/POTHOLE_YOLO.git
cd POTHOLE_YOLO
```

2. Install dependencies:
```bash
pip install ultralytics kagglehub opencv-python
```

## Usage

### 1. Download Dataset

```bash
python download_dataset.py
```

Downloads the [Potholes Detection YOLOv8 dataset](https://www.kaggle.com/datasets/anggadwisunarto/potholes-detection-yolov8) from Kaggle.

### 2. Train the Model

```bash
python train.py
```

Trains YOLOv8n model for 100 epochs. Model weights are saved in `runs/detect/train/weights/`.

### 3. Run Inference

```bash
python predict.py
```

Processes a video and outputs `predicted_sample_video.mp4` with detected potholes.

## Model Details

- **Architecture**: YOLOv8n (nano version)
- **Classes**: 1 (pothole)
- **Confidence Threshold**: 0.5
- **Training Epochs**: 100

## Requirements

```
ultralytics
kagglehub
opencv-python
```
