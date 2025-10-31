from ultralytics import YOLO

# Use YOLOv8n (nano) - it will auto-download if not present
model = YOLO('yolov8n.yaml')

# use the model
# Fix: use relative path or absolute path for data.yaml
results = model.train(data='potholes-dataset/data.yaml', epochs=100)