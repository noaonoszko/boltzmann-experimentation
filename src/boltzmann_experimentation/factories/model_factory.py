import timm
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.optim.lr_scheduler import MultiStepLR

from boltzmann_experimentation import models
from boltzmann_experimentation.config.literals import MODEL_TYPE
from boltzmann_experimentation.config.settings import (
    general_settings as g,
)
from boltzmann_experimentation.config.settings import (
    perceptron_settings,
)
from boltzmann_experimentation.training.model import Model


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
                torch_model = models.DenseNet121()
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
                g.batch_size_train = 128
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
