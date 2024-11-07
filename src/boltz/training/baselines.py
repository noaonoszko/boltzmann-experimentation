from typing import Iterator
import torch
from boltz.config.literals import MODEL_TYPE
from boltz.config.settings import general_settings as g
from boltz.factories import ModelFactory
from boltz.schemas.training_components import TrainingComponents
from boltz.utils.logger import (
    general_logger,
    init_wandb_run,
)
from tqdm import tqdm
from tqdm.auto import trange

import wandb


def train_baselines(
    same_model_init_values: list[bool],
    SEED: int,
    log_to_wandb: bool,
    t: TrainingComponents,
    wandb_group: str,
    model_type: MODEL_TYPE,
    infinite_val_loader: Iterator[tuple[torch.Tensor, torch.Tensor]],
    infinite_train_loader: Iterator[tuple[torch.Tensor, torch.Tensor]],
) -> None:
    for same_model_init in tqdm(same_model_init_values):
        if same_model_init:
            torch.manual_seed(SEED)
        model = ModelFactory.create_model(t)
        general_logger.info(f"Model has {model.num_params()/1e6:.0f}M params.")
        if log_to_wandb:
            wandb.finish()
            run_name = "Central"
            init_wandb_run(
                run_name=run_name,
                group=wandb_group,
                model_type=model_type,
                training_components=t,
            )
        val_batch = next(infinite_val_loader)
        model.val_step(val_batch)
        for _ in trange(g.num_comrounds):
            batch = next(infinite_train_loader)
            model.torch_model.train()
            model.train_step(batch)
            val_batch = next(infinite_val_loader)
            model.torch_model.eval()
            model.val_step(val_batch)
            if model.lr_scheduler is not None:
                model.lr_scheduler.step()
