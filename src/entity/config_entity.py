from pathlib import Path
from dataclasses import dataclass

@dataclass
class ModelTrainerConfig:
    data_root: Path
    epochs: int
    name: str

@dataclass
class ModelEvaluatorConfig:
    model_path: Path
    data_root: Path
    imgsz: int = 768
    split: str = 'test'