from ultralytics import YOLO, YOLOWorld, RTDETR

model = YOLO('yolov8n.pt')
metrics = model.val(data='test-config.yaml')

print("map50-95: ", metrics.box.map)
print("map50: ", metrics.box.map50)
print("map75: ", metrics.box.map75)
print("maps: ", metrics.box.maps)







