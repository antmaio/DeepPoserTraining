#External
from PIL import Image
import os
import numpy as np
from ultralytics import YOLO
import torch

def visualize_yolo_results(res, cam:str)->None:
    annotated_frame = res[0].plot()
    im = Image.fromarray(annotated_frame)
    os.makedirs('processed_images', exist_ok=True)
    im.save(os.path.join('processed_images', f"yolo_results_on_{cam}.jpeg"))

def init_yolo(yolo_model:str='yolov8x-pose.pt'):
    # Create yolo model
    model = YOLO(yolo_model)
    model.to(0)
    # Perform object detection on an image using the model
    print('cuda:', torch.cuda.is_available())
    return model


