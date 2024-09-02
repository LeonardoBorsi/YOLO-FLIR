from ultralytics import YOLO, YOLOWorld, RTDETR

model = YOLO('yolov8n.pt')
model.train(data='train-config.yaml', device=0, epochs=50, batch=8, verbose=True)