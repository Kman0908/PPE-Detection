from pathlib import Path
from ultralytics import YOLO
from src.entity.config_entity import ModelEvaluatorConfig
from src.logger import logging
from src.exception import CustomException

class ModelEvaluator:
    def __init__(self, config: ModelEvaluatorConfig):
        self.config = config
        self.model_path = config.model_path
        self.imgsz = config.imgsz
        self.data_yaml = config.data_root
        self.split = config.split

    def evaluate(self):
        try:
            logging.info('Model Evaluation started')

            if not self.model_path.is_file():
                raise FileNotFoundError(f'Model checkpoint not found: {self.model_path}')
            
            if not self.data_yaml.is_file():
                raise FileNotFoundError(f'Dataset YAML not found at: {self.data_yaml}')
            
            model = YOLO(str(self.model_path))

            results = model.val(
                data = str(self.data_yaml),
                split = self.split,
                imgsz = self.imgsz,
                plots = True,
                verbose = False
            )

            metrics = results.box

            evaluation_result = {
                'precision': float(metrics.mp),
                'recall': float(metrics.mr),
                'mAP50': float(metrics.mAP50),
                'mAP50-95': float(metrics.map)
            }

            logging.info(
                f'Evaluation completed on {self.split} split | ',
                f'Precision: {metrics.mp:.4f} | ',
                f'Recall: {metrics.mr:.4f} | ',
                f'mAP50: {metrics.mAP50:.4f} | ',
                f'mAP50-95: {metrics.map:.4f}'
            )

            for class_id, class_name in results.names.items():
                if class_id < len(metrics.ap50):
                    logging.info(
                        f'Class: {class_name}| ',
                        f'mAP50: {metrics.ap50['class_id']:.4f}| ',
                        f'mAP50-95: {metrics.ap['class_id']:.4f}'
                    )
            return evaluation_result
        except exception as e:
            logging.exception('Error occurred while model evaluation')
            raise CustomException(e)