import os
from yaml import safe_load
from src.logger import logging
from src.exception import CustomException
from src.entity.config_entity import ModelTrainerConfig
from src.ppe_detection.model_trainer import ModelTrainer
from src.ppe_detection.model_evaluator import ModelEvaluator
from src.entity.config_entity import ModelEvaluatorConfig

path = os.path.join(os.getcwd(), 'config/config.yaml')

with open(path, 'r') as f:
    try:
        _yaml = safe_load(f)
    except Exception as e:
        raise CustomException(e)
config = _yaml['Model_Trainer']

mt_conf = ModelTrainerConfig(
    data_root = config['data_root'],
    epochs = config['epoch'],
    name = config['name']
)

eval_conf = _yaml['Model_Evaluator']

me_conf = ModelEvaluatorConfig(
    model_path = eval_conf['model_path'],
    data_root = eval_conf['data_root'],
    imgsz = eval_conf['imgsz'],
    split = eval_conf['split']
)

if __name__ == '__main__':
    model_trainer_object = ModelTrainer(config = mt_conf)
    results = model_trainer_object.train()

    model_evaluator_object = ModelEvaluator(config = me_conf)
    evaluation_result = model_evaluator_object.evaluate()

    logging.info(f'Final Evaluaton Metrics: {evaluation_result}')
    print(f'Evaluation result: {evaluation_result}')
