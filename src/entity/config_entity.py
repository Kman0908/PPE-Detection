from pathlib import Path
from dataclasses import dataclass

@dataclass
class ModelTrainerConfig:
    checkpoint_dir: Path
    data_root: Path
    epochs: int
    name: str

