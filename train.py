from ultralytics import YOLO
from ultralytics import settings
# Update a setting
settings.update({'datasets_dir': '.'})

# Load a pre-trained YOLO model
model = YOLO("yolov8l.pt")

# 'pandaai.yaml' dataset to train the model
model.train(data="pandaai.yaml", epochs=300, batch=8, workers=4, degrees=90.0)
