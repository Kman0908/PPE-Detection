import os
from yaml import safe_load
from src.logger import logging
from src.exception import CustomException
from src.entity.config_entity import ModelTrainerConfig
from src.ppe_detection.model_trainer import ModelTrainer

path = os.path.join(os.getcwd(), 'Data/data.yaml')

with open(path, 'r') as f:
    try:
        _yaml = safe_load(f)
    except Exception as e:
        raise CustomException(e)
config = _yaml['Model_trainer']

mt_conf = ModelTrainerConfig(
    data_root = config['data_root'],
    epochs = config['epoch'],
    name = config['name']
)

if __name__ == '__main__':
    model_trainer_object = ModelTrainer(config = mt_conf)
    results = model_trainer_object.train()
