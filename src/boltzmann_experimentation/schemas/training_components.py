from typing import Callable

import torch
import torch.nn as nn
from boltzmann_experimentation.config.literals import MODEL_TYPE
from pydantic import BaseModel, Field


class TrainingComponents(BaseModel):
    model_type: MODEL_TYPE
    initial_lr: float = Field(..., gt=0)
    criterion: nn.modules.loss._Loss
    lr_scheduler: torch.optim.lr_scheduler.LRScheduler | None = None
    loss_transformation: Callable[[torch.Tensor, nn.Module], torch.Tensor] | None = None

    class Config:
        arbitrary_types_allowed = True
