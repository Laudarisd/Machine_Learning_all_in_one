# src/core/train.py
import os
from ultralytics import YOLO
os.environ['WANDB_MODE'] = 'disabled'

class TrainYolo:
    def __init__(self, 
                 data_yaml_path, model_path, 
                 epochs, 
                 batch_size, 
                 img_size, 
                 device):
        
        self.data_yaml_path = data_yaml_path
        self.model_path = model_path
        self.epochs = epochs
        self.batch_size = batch_size
        self.img_size = img_size
        self.device = device

    def train_model(self):
        try:
            model = YOLO(self.model_path)
            model.train(
                data=self.data_yaml_path,
                epochs=self.epochs,
                batch=self.batch_size,
                imgsz=self.img_size,
                device=self.device
            )
        except Exception as e:
            print(f"An error occurred during training with parameters: {e}")
