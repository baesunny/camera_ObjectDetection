from ultralytics import YOLO

# Initialize the model
model = YOLO('yolov8n.pt', task='detect')

# Train the model on your dataset
model.train(data='coco8.yaml', epochs=30, lr0=0.05)

# Save the trained model
model_path = r"Edit path"
model.save(model_path)

model = YOLO(model_path)
