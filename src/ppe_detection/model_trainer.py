import torch
from ultralytics import YOLO
from src.logger import logging
from src.exception import CustomException
from src.entity.config_entity import ModelTrainerConfig

class ModelTrainer():
    def __init__(self, config: ModelTrainerConfig):
        self.config = config
        self.model = YOLO(config.name)
        self.data_yaml = config.data_root / 'data.yaml'
        self.epochs = config.epochs
        self.save_path = config.checkpoint_dir
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
                project = str(self.save_path),
                name = 'training'
            )
            metrics = results.box


            logging.info(
                "Training completed | "
                f"mAP50: {metrics.get('metrics/mAP50(B)', 0):.4f} | "
                f"mAP50-95: {metrics.get('metrics/mAP50-95(B)', 0):.4f} | "
                f"Precision: {metrics.get('metrics/precision(B)', 0):.4f} | "
                f"Recall: {metrics.get('metrics/recall(B)', 0):.4f}"
            )

            return results
            
        except CustomException as e:
            logging.exception('Error occured while training')
            raise CustomException(e)