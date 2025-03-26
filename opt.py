from ultralytics import YOLO

# Load the YOLO model
model_path = "runs/detect/train/weights/best.pt"
model = YOLO(model_path)

# Export the model to ONNX format for better compatibility with edge devices
model.export(format="onnx", simplify=True)
