from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from ultralytics import YOLO
from sklearn.model_selection import train_test_split
import numpy as np
import os
import cv2

# Assuming you have your dataset prepared
img_dir = r"C:\Users\soeon\OneDrive\Desktop\External_Univ\BITAmin_13th\Projects\Projects\Projects2\YOLO\train\images\train"  # feature set
label_dir = r"C:\Users\soeon\OneDrive\Desktop\External_Univ\BITAmin_13th\Projects\Projects\Projects2\YOLO\train\labels\train"  # labels

images = [os.path.join(img_dir, f) for f in os.listdir(img_dir) if f.endswith('.jpg')]
labels = [os.path.join(label_dir, f.replace('.jpg', '.txt')) for f in os.listdir(img_dir) if f.endswith('.jpg')]

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

# Initialize the model
model = YOLO('yolov8n.pt', task='detect')

# Train the model on your dataset
model.train(data='coco8.yaml', epochs=30, lr0=0.01)

# Test the model and compute metrics
def evaluate_model(model, X_test, y_test):
    y_true = []
    y_pred = []
    for img_path, label_path in zip(X_test, y_test):
        img = cv2.imread(img_path)
        results = model(img)
        # Extract the actual and predicted labels and add them to y_true and y_pred
        # Aa simplified example, add the actual label to y_true and the predicted label to y_pred
        # In an actual implementation, labeling must be performed for each class
        y_true.append(0)  # Actual Label
        y_pred.append(0)  # Predicted Label

    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='weighted')
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')

    return accuracy, f1, precision, recall

# Get the evaluation metrics
accuracy, f1, precision, recall = evaluate_model(model, X_test, y_test)

# Get the evaluation metrics
print(f"Accuracy: {accuracy:.2f}")
print(f"F1 Score: {f1:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")

# Save the trained model
model_path = r"C:\Users\soeon\runs\detect\trained_model.pt"
model.save(model_path)

model = YOLO(model_path)

