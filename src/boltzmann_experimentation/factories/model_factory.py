import timm
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.optim.lr_scheduler import MultiStepLR

from boltzmann_experimentation import models
from boltzmann_experimentation.config.settings import (
    perceptron_settings,
)
from boltzmann_experimentation.schemas.training_components import TrainingComponents
from boltzmann_experimentation.training.model import Model
from boltzmann_experimentation.config.settings import general_settings as g


class ModelFactory:
    @staticmethod
    def create_model(t: TrainingComponents) -> Model:
        # Initialize the torch_model separately based on model_type
        match t.model_type:
            case "deit-b":
                torch_model = timm.create_model(
                    "deit_base_patch16_224", pretrained=False, num_classes=10
                )
                optimizer = optim.Adam(torch_model.parameters(), lr=t.initial_lr)
            case "densenet":
                torch_model = models.DenseNet121()
                optimizer = optim.SGD(
                    torch_model.parameters(),
                    lr=t.initial_lr,
                    momentum=0.9,
                    weight_decay=1e-4,
                    nesterov=True,
                )
                t.lr_scheduler = MultiStepLR(
                    optimizer,
                    milestones=(
                        int(0.5 * g.num_communication_rounds),
                        int(0.75 * g.num_communication_rounds),
                    ),
                    gamma=0.1,
                )
            case "resnet18":
                torch_model = torchvision.models.resnet18()
                torch_model.fc = nn.Linear(torch_model.fc.in_features, 10)
                optimizer = optim.Adam(
                    torch_model.parameters(), lr=t.initial_lr, weight_decay=1e-4
                )
            case "simple-cnn":
                torch_model = models.SimpleCNN()
                optimizer = optim.Adam(torch_model.parameters(), lr=t.initial_lr)
            case "single-neuron-perceptron":
                torch_model = models.SingleNeuronPerceptron()
                optimizer = optim.SGD(torch_model.parameters(), lr=t.initial_lr)
            case "two-layer-perceptron":
                torch_model = models.TwoLayerPerceptron(
                    perceptron_settings.input_size,
                    perceptron_settings.hidden_size,
                    perceptron_settings.output_size,
                )
                optimizer = optim.AdamW(torch_model.parameters(), lr=t.initial_lr)
            case _:
                raise ValueError(f"Unsupported model type: {t.model_type}")

        return Model(
            torch_model=torch_model,
            optimizer=optimizer,
            criterion=t.criterion,
            lr_scheduler=t.lr_scheduler,
            loss_transformation=t.loss_transformation,
        )
