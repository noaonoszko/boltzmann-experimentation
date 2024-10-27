from pydantic_settings import BaseSettings
from datetime import datetime
from pathlib import Path
import torch
from boltzmann_experimentation.literals import GPU


class PerceptronSettings(BaseSettings):
    hidden_size: int = 10000
    output_size: int = 1
    input_size: int = 1


class GeneralSettings(BaseSettings):
    # Distributed learning
    num_miners: int = -1
    num_communication_rounds: int = -1
    compression_factor: int = 100

    # Data
    batch_size: int = -1
    num_workers_dataloader: int = 4  # This will change based on the device

    # Synthetic data
    data_size: int = 10000
    val_data_fraction: float = 0.2

    # Results
    results_dir: Path = Path(__file__).parents[2] / "results"

    # Device
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def set_device(self, gpu: GPU | None = None):
        t = torch.cuda.is_available()
        self.device = torch.device(
            f"cuda:{gpu}" if t and gpu is not None else "cuda" if t else "cpu"
        )

        # Update other settings depending on the device
        self.num_workers_dataloader = 4 if self.device.type == "cuda" else 0

    # Logging
    log_to_wandb: bool = True


perceptron_settings = PerceptronSettings()
general_settings = GeneralSettings()
start_ts = datetime.now()
