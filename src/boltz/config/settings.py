from pydantic_settings import BaseSettings
from pydantic import Field
from datetime import datetime
from pathlib import Path
import torch
from boltz.config.literals import GPU, OPTIMIZER


class PerceptronSettings(BaseSettings):
    hidden_size: int = 10000
    output_size: int = 1
    input_size: int = 1


class GeneralSettings(BaseSettings):
    """
    General settings used for the experiment.

    Note! -1 default values are temporary and are overridden, mostly by CLI
    arguments or their defaults but in some cases elsewhere.
    """

    # Training
    num_epochs: int = -1
    optimizer: OPTIMIZER = "adam"

    # Model
    model_kwargs: dict = Field(default_factory=dict)

    # Distributed learning
    num_miners: int = -1
    num_comrounds: int = -1
    compression_factor: int = 100
    agg_bn_params: bool = True

    # Data
    batch_size_train: int = -1
    batch_size_val: int = -1
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
    wandb_legend_params: list[str] = Field(default_factory=list)

    model_config = {"protected_namespaces": ("settings_",)}


perceptron_settings = PerceptronSettings()
general_settings = GeneralSettings()
start_ts = datetime.now()
