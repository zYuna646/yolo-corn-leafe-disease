from ultralytics import YOLO

# Buat model YOLOv8
model = YOLO('yolov8s.pt')  # Pre-trained model

# Mulai pelatihan
model.train(data='../../data.yaml', epochs=100, batch_size=16)