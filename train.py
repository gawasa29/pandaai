from ultralytics import YOLO

# Load a pre-trained YOLO model
model = YOLO("yolov8l.pt")

# 'pandaai.yaml' dataset to train the model
#  mps is Apple M1 use
model.train(data="pandaai.yaml", epochs=100, batch=32, device='mps')
