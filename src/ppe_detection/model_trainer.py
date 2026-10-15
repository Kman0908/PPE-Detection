import torch
from  pathlib import Path
from ultralytics import YOLO
from src.logger import logging
from src.exception import CustomException
from src.entity.config_entity import ModelTrainerConfig

class ModelTrainer():
    def __init__(self, config: ModelTrainerConfig):
        self.config = config
        self.model = YOLO(config.name)
        self.data_yaml = Path(config.data_root) / 'data.yaml'
        self.epochs = config.epochs
        self.device = 0 if torch.cuda.is_available() else 'cpu'
    
    def train(self):
        try:
            logging.info('Training Started')

            results = self.model.train(
                data = str(self.data_yaml),
                epochs = self.epochs,
                imgsz = 640,
                batch = 32,
                device = self.device,
                name = 'training'
            )
            metrics = results.box

            logging.info(
                "Training completed | "
                f"mAP50: {metrics.map50:.4f} | "
                f"mAP50-95: {metrics.map:.4f} | "
                f"Precision: {metrics.mp:.4f} | "
                f"Recall: {metrics.mr:.4f}"
            )

            return results
            
        except CustomException as e:
            logging.exception('Error occured while training')
            raise CustomException(e)