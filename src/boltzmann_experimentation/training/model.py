from torch.optim.lr_scheduler import MultiStepLR, LRScheduler
import json
from copy import deepcopy
from datetime import datetime
from typing import Callable

import timm
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from pydantic import BaseModel, Field

import wandb
from boltzmann_experimentation.config.literals import MODEL_TYPE
from boltzmann_experimentation import models
from boltzmann_experimentation.logger import general_logger, metrics_logger
from boltzmann_experimentation.settings import (
    perceptron_settings,
    general_settings as g,
)


class MinerSlice(BaseModel):
    miner_id: int = Field(..., description="Unique identifier for the miner")
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="Timestamp of when the slice was uploaded",
    )
    data: torch.Tensor = Field(..., description="Slice of the model's parameters")

    class Config:
        arbitrary_types_allowed = True


class ModelFactory:
    @staticmethod
    def create_model(model_type: MODEL_TYPE):
        lr_scheduler = None
        loss_transformation = None
        match model_type:
            case "deit-b":
                # Load the pretrained DeiT model
                torch_model = timm.create_model(
                    "deit_base_patch16_224", pretrained=False, num_classes=10
                )

                # Define loss and optimizer
                criterion = torch.nn.CrossEntropyLoss()
                optimizer = torch.optim.Adam(torch_model.parameters(), lr=1e-4)
            case "densenet":
                torch_model = torchvision.models.DenseNet(
                    growth_rate=24, num_classes=10
                )
                criterion = nn.CrossEntropyLoss()
                optimizer = optim.SGD(
                    torch_model.parameters(),
                    lr=0.1,
                    momentum=0.9,
                    weight_decay=1e-4,
                    nesterov=True,
                )
                lr_scheduler = MultiStepLR(
                    optimizer,
                    milestones=(
                        int(0.5 * g.num_communication_rounds),
                        int(0.75 * g.num_communication_rounds),
                    ),
                    gamma=0.1,
                )
                g.batch_size_train = 64
            case "resnet18":
                torch_model = torchvision.models.resnet18()
                torch_model.fc = torch.nn.Linear(torch_model.fc.in_features, 10)
                criterion = nn.CrossEntropyLoss()
                optimizer = optim.Adam(
                    torch_model.parameters(), lr=0.001, weight_decay=1e-4
                )
            case "simple-cnn":
                torch_model = models.SimpleCNN()
                criterion = nn.CrossEntropyLoss()
                optimizer = optim.Adam(torch_model.parameters(), lr=0.001)
            case "single-neuron-perceptron":
                torch_model = models.SingleNeuronPerceptron()
                criterion = nn.MSELoss()
                optimizer = optim.SGD(torch_model.parameters(), lr=0.01)
                return Model(
                    torch_model,
                    optimizer,
                    criterion,
                )
            case "two-layer-perceptron":
                torch_model = models.TwoLayerPerceptron(
                    perceptron_settings.input_size,
                    perceptron_settings.hidden_size,
                    perceptron_settings.output_size,
                )
                optimizer = optim.AdamW(torch_model.parameters(), lr=0.01)
                criterion = nn.MSELoss()  # Mean squared error for regression

                def loss_transformation(
                    loss: torch.Tensor, torch_model: nn.Module
                ) -> torch.Tensor:
                    # Add L1 regularization to the loss
                    # Regularization factor (L1 penalty strength)
                    l1_lambda = 0
                    l1_norm = sum(p.abs().sum() for p in torch_model.parameters())
                    return loss + l1_lambda * l1_norm  # Add L1 penalty

            case _:
                raise ValueError(f"Unsupported model type: {model_type}")
        return Model(
            torch_model,
            optimizer,
            criterion,
            lr_scheduler=lr_scheduler,
            loss_transformation=loss_transformation,
        )


class Model:
    def __init__(
        self,
        torch_model: nn.Module,
        optimizer: optim.Optimizer,
        criterion: nn.modules.loss._Loss,
        lr_scheduler: LRScheduler | None = None,
        loss_transformation: Callable | None = None,
    ):
        self.torch_model = torch_model.to(g.device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.lr_scheduler = lr_scheduler
        self.loss_transformation = loss_transformation
        self.val_losses = torch.tensor([])
        self.metrics = {
            "train_loss": [],
            "val_loss": [],
            "train_acc": [],
            "val_acc": [],
        }
        self.log_data = {}

    def add_metric_to_log(self, metric_name: str, value: float) -> None:
        self.log_data[metric_name] = value
        wandb.log(self.log_data, commit=False)

    def commit_log(self) -> None:
        wandb.log(self.log_data, commit=True)

    def val_step(self, batch: tuple[torch.Tensor, torch.Tensor]) -> None:
        general_logger.debug("Validation step")
        inputs, targets = batch

        # Compute loss without updating gradients
        with torch.no_grad():
            outputs = self.torch_model(inputs)
            loss = self.criterion(outputs, targets)

        # Calculate accuracy (assuming classification)
        _, predicted = torch.max(outputs, 1)  # Get the index of the max logit
        correct = (predicted == targets).sum().item()  # Count correct predictions
        num_samples = targets.size(0)  # Number of samples in the batch

        loss = loss.item()
        accuracy = correct / num_samples
        if g.log_to_wandb:
            self.add_metric_to_log("val_loss", loss)
            self.add_metric_to_log("val_acc", accuracy)
            self.commit_log()

    def validate(self, batch: tuple[torch.Tensor, torch.Tensor]):
        """Run validation over one batch."""
        total_loss = 0.0
        total_correct = 0
        num_samples = 0

        loss, correct, num_samples = self.val_step(batch)
        total_loss = loss * num_samples  # Accumulate weighted loss
        total_correct = correct  # Accumulate correct predictions

        # Calculate average loss and accuracy over the entire validation set
        avg_loss = total_loss / num_samples
        avg_accuracy = total_correct / num_samples

        if g.log_to_wandb:
            self.add_metric_to_log("val_loss", avg_loss)
            self.add_metric_to_log("val_acc", avg_accuracy)
            self.commit_log()
        return avg_loss, avg_accuracy

    def log_metrics_to_file(self, metric_name: str, value: float) -> None:
        """Log and store metrics in JSON format."""
        if metric_name not in self.metrics:
            self.metrics[metric_name] = []
        self.metrics[metric_name].append(value)

        # Log the entire metrics dictionary as a JSON string
        metrics_logger.info(json.dumps({metric_name: value}))

    def train_step(self, data: tuple[torch.Tensor, torch.Tensor]):
        general_logger.debug("Training one step")
        inputs, targets = data

        # Zero gradients from previous step
        self.optimizer.zero_grad()

        # Forward pass
        outputs = self.torch_model(inputs)

        # Compute loss
        loss = self.criterion(outputs, targets)
        if self.loss_transformation:
            loss = self.loss_transformation(loss, self.torch_model)

        # Backward pass (compute gradients)
        loss.backward()

        # Update weights
        self.optimizer.step()

        return loss.item()

    def update_with_slice(
        self, slice: torch.Tensor, slice_indices: torch.Tensor
    ) -> None:
        """Insert slice into the model."""
        # Flatten the own model
        own_model_params = torch.nn.utils.parameters_to_vector(
            self.torch_model.parameters()
        )
        own_model_params[slice_indices] = slice
        torch.nn.utils.vector_to_parameters(  # Needed? Or does the previous line suffice?
            own_model_params, self.torch_model.parameters()
        )

    def aggregate_slices(
        self, slices: list[torch.Tensor], slice_indices: torch.Tensor
    ) -> nn.Module:
        # Flatten parameters
        params = torch.nn.utils.parameters_to_vector(self.torch_model.parameters())

        # Aggregate
        slices_elementwise_avg = torch.mean(torch.stack(slices), dim=0)

        # Update model
        params[slice_indices] = slices_elementwise_avg
        new_torch_model = deepcopy(self.torch_model)
        torch.nn.utils.vector_to_parameters(params, new_torch_model.parameters())
        return new_torch_model

    def get_params(self):
        # Get a dictionary of model parameters
        return {
            name: param.clone() for name, param in self.torch_model.named_parameters()
        }

    def update_params(self, param_slice):
        # Update model parameters using received slice
        with torch.no_grad():
            for name, param in self.torch_model.named_parameters():
                if name in param_slice:
                    param.copy_(param_slice[name])

    def compute_loss(self, data):
        # Compute loss for validation
        inputs, targets = data

        with torch.no_grad():
            outputs = self.torch_model(inputs)
            loss = self.criterion(outputs, targets)

        return loss.item()

    def sum_params(self):
        return torch.tensor([p.sum() for p in self.torch_model.parameters()]).sum()

    def num_params(self):
        return sum(p.numel() for p in self.torch_model.parameters())
