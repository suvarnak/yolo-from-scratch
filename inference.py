import torch
import cv2
import numpy as np
from models.model import MyYolo
from utils.util import non_max_suppression, plot_boxes

# Path to best model and test image
MODEL_PATH = 'runs/train/best.pt'
TEST_IMAGE = 'test2.jpg'  # Change to your test image path

# Load model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = MyYolo(version='n').to(device)
checkpoint = torch.load(MODEL_PATH, map_location=device)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

# Load and preprocess image
img = cv2.imread(TEST_IMAGE)
img_resized = cv2.resize(img, (640, 640))
img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float().unsqueeze(0) / 255.0
img_tensor = img_tensor.to(device)

# Inference
with torch.no_grad():
    preds = model(img_tensor)
    detections = non_max_suppression(preds, confidence_threshold=0.25, iou_threshold=0.45)[0]

# Visualize results
result_img = plot_boxes(img_resized, detections)
cv2.imshow('YOLO Inference', result_img)
cv2.waitKey(0)
cv2.destroyAllWindows()