from ultralytics import YOLO

# Loading a custom dataset trained model
model = YOLO("yolov8s.pt")

results=model(source=0,show=True,conf=0.6,save=True)