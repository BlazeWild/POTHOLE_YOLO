# Pothole Detection with YOLO

A deep learning project for detecting potholes in images and videos using YOLOv8/YOLO11 object detection model.

## 📋 Project Overview

This project uses the YOLO (You Only Look Once) framework to detect potholes in road images and videos. The model is trained on a custom pothole dataset and can be used for real-time pothole detection in traffic management systems.

## 🚀 Features

- Download and prepare pothole detection dataset from Kaggle
- Train YOLOv8/YOLO11 model on pothole dataset
- Real-time pothole detection in videos
- Bounding box visualization with confidence scores
- Easy-to-use Python scripts

## 📁 Project Structure

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

## 🛠️ Installation

### Prerequisites

- Python 3.8+
- CUDA-enabled GPU (recommended for training)

### Setup

1. Clone this repository:
```bash
git clone https://github.com/BlazeWild/POTHOLE_YOLO.git
cd POTHOLE_YOLO
```

2. Install required packages:
```bash
pip install ultralytics kagglehub opencv-python
```

## 📊 Dataset

The project uses the [Potholes Detection YOLOv8 dataset](https://www.kaggle.com/datasets/anggadwisunarto/potholes-detection-yolov8) from Kaggle.

### Download Dataset

Run the download script to fetch the dataset:
```bash
python download_dataset.py
```

This will download and extract the dataset to the `potholes-dataset/` folder.

### Dataset Details
- **Classes**: 1 (pothole)
- **Training Images**: Located in `train/images/`
- **Validation Images**: Located in `valid/images/`
- **Annotations**: YOLO format (.txt files)

## 🏋️ Training

To train the model:

```bash
python train.py
```

### Training Configuration
- **Model**: YOLOv8n (nano) - lightweight and fast
- **Epochs**: 100
- **Dataset**: Custom pothole dataset

The trained model weights will be saved in `runs/detect/train/weights/`.

## 🎯 Inference

To run detection on a video:

```bash
python predict.py
```

### Inference Details
- **Input**: Sample video from `potholes-dataset/sample_video.mp4`
- **Output**: `predicted_sample_video.mp4` with bounding boxes
- **Threshold**: 0.5 (50% confidence)
- **Model**: Best weights from training (`runs/detect/train/weights/last.pt`)

## 📈 Results

The model outputs:
- Bounding boxes around detected potholes
- Class labels ("POTHOLE")
- Confidence scores
- Processed video with real-time annotations

## 🔧 Configuration

### Model Configuration
You can modify the model architecture by changing the YAML file in `train.py`:
- `yolov8n.yaml` - Nano (fastest, smallest)
- `yolov8s.yaml` - Small
- `yolov8m.yaml` - Medium
- `yolov8l.yaml` - Large
- `yolov8x.yaml` - Extra Large

### Training Parameters
Adjust training parameters in `train.py`:
- `epochs`: Number of training epochs
- `imgsz`: Image size for training
- `batch`: Batch size
- `device`: GPU device ID

## 📝 Requirements

```
ultralytics
kagglehub
opencv-python
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is open-source and available for educational and research purposes.

## 🙏 Acknowledgments

- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) for the YOLO implementation
- [Kaggle Dataset](https://www.kaggle.com/datasets/anggadwisunarto/potholes-detection-yolov8) for the pothole dataset
- OpenCV for image processing capabilities

## 📧 Contact

For questions or suggestions, please open an issue on GitHub.

---

**Made with ❤️ for safer roads**
