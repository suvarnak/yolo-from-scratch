import math, os, random, cv2, numpy, torch
import torch.nn as nn
from ultralytics import YOLO

# explore the original yolov8 model

# original yolov8n
model_n=YOLO('yolov8n.pt')
print(f"yolov8-nano: {sum(p.numel() for p in model_n.parameters())/1e6} million parameters")


# model_s=YOLO('yolov8s.pt')
# print(f"yolov8-small: {sum(p.numel() for p in model_s.parameters())/1e6} million parameters")

print(model_n.model)