from pathlib import Path
import torch
from dataclasses import dataclass
import yaml

@dataclass
class Config:
    #-------- Signal Parameters -------
    SR = 2000
    DURATION_S = 5
    SEQ_LEN = SR * DURATION_S

    DATA_ROOT = Path("./data/dataset_downsampled")
    ECG_DIR = DATA_ROOT / "ECG"
    PCG_DIR = DATA_ROOT / "PCG"
    CHECK_DIR = Path("checkpoints")

    BATCH_SIZE = 32
    SEED = 1420
    PIN_MEMORY = True

    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    @staticmethod
    def load_model_config():
        with open("configs/model_config.yaml", "r") as f:
            return yaml.safe_load(f)
    @staticmethod
    def load_inference_config():
        with open("configs/inference_config.yaml", "r") as f:
            return yaml.safe_load(f)

    