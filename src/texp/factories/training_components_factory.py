import torch
import torch.nn as nn

from texp.config.literals import MODEL_TYPE
from texp.config.settings import (
    general_settings as g,
)
from texp.schemas.training_components import TrainingComponents


class TrainingComponentsFactory:
    @staticmethod
    def create_components(model_type: MODEL_TYPE) -> TrainingComponents:
        """
        Creates the training components (criterion, initial lr, etc.) based on
        the model_type.
        """
        lr_scheduler = None
        loss_transformation = None

        match model_type:
            case "deit-b":
                criterion = nn.CrossEntropyLoss()
                initial_lr = 1e-4

            case "densenet":
                criterion = nn.CrossEntropyLoss()
                match g.optimizer:
                    case "adam":
                        initial_lr = 0.001 if g.initial_lr is -1.0 else g.initial_lr
                    case "sgd":
                        initial_lr = 0.1
                    case _:
                        raise ValueError(f"Unsupported optimizer type: {g.optimizer}")
                g.batch_size_train = 128

            case "resnet18":
                criterion = nn.CrossEntropyLoss()
                initial_lr = 0.001

            case "simple-cnn":
                criterion = nn.CrossEntropyLoss()
                initial_lr = 0.001

            case "single-neuron-perceptron":
                criterion = nn.MSELoss()
                initial_lr = 0.01

            case "two-layer-perceptron":
                criterion = nn.MSELoss()
                initial_lr = 0.01

                def loss_transformation(
                    loss: torch.Tensor, torch_model: nn.Module
                ) -> torch.Tensor:
                    l1_lambda = 0
                    l1_norm = sum(p.abs().sum() for p in torch_model.parameters())
                    return loss + l1_lambda * l1_norm

            case _:
                raise ValueError(f"Unsupported model type: {model_type}")

        return TrainingComponents(
            model_type=model_type,
            initial_lr=initial_lr,
            criterion=criterion,
            lr_scheduler=lr_scheduler,
            loss_transformation=loss_transformation,
        )
