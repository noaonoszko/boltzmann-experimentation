from loguru import logger
import sys
from pathlib import Path
import wandb
from boltzmann_experimentation.config.settings import (
    general_settings as g,
    start_ts,
)
from boltzmann_experimentation.schemas.training_components import TrainingComponents

# Remove the default logger
logger.remove()

# Set up the general logger (stdout)
general_logger = logger.bind(name="general")
general_logger.add(
    sys.stdout,
    level="INFO",
    filter=lambda record: record["extra"].get("name") == "general",
)

# Set up the metrics logger (file logging)
metrics_logger = logger.bind(name="metrics")


def add_file_logger(log_dir: Path) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    metrics_logger.add(
        log_dir / "metrics.log",
        level="INFO",
        rotation="10 MB",
        compression="zip",
        retention=5,
        filter=lambda record: record["extra"].get("name") == "metrics",
    )


def init_wandb_run(
    *,
    run_name: str,
    model_type: str,
    training_components: TrainingComponents,
    start_ts_str: str | None = None,
) -> None:
    wandb.init(
        project="chakana",
        name=run_name,
        group=start_ts_str if start_ts_str is not None else str(start_ts),
        config={
            "model_type": model_type,
            "batch_size": g.batch_size_train,
        }
        | g.model_config
        | training_components.__dict__,
        resume="allow",
    )
