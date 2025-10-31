import os
import kagglehub
import shutil


download_path = "potholes-dataset"

# Download dataset using kagglehub
temp_path = kagglehub.dataset_download("anggadwisunarto/potholes-detection-yolov8")

os.makedirs(download_path, exist_ok=True)

# Copy extracted dataset to the desired location
shutil.copytree(temp_path, download_path, dirs_exist_ok=True)

print("✅ Dataset downloaded to:", download_path)
