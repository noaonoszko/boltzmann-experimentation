import torch
from timm.data import create_transform
from torch.utils.data import Dataset
from torchvision import datasets, transforms

from boltzmann_experimentation.config.literals import MODEL_TYPE
from boltzmann_experimentation.config.settings import (
    general_settings as g,
)
from boltzmann_experimentation.config.settings import (
    perceptron_settings,
)
from boltzmann_experimentation.data.dataset import LinearRegressionDataset


class DatasetFactory:
    @staticmethod
    def create_dataset(model_type: MODEL_TYPE) -> tuple[Dataset, Dataset]:
        match model_type:
            case (
                "simple-cnn"
                | "resnet18"
                | "densenet"
                | "densenet-kuangliu"
                | "deit-b"
            ):
                # Define data transformation (e.g., normalization)
                if model_type == "deit-b":
                    train_transform = create_transform(
                        input_size=224,  # Resize CIFAR-10 images to 224x224
                        is_training=True,
                        auto_augment="rand-m9-mstd0.5-inc1",
                        interpolation="bicubic",
                        mean=(0.485, 0.456, 0.406),
                        std=(0.229, 0.224, 0.225),
                    )
                    val_transform = train_transform
                else:
                    # Define common transforms
                    val_transform = transforms.Compose(
                        [
                            transforms.ToTensor(),
                            transforms.Normalize(
                                (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)
                            ),
                        ]
                    )

                    # Training set transformations with data augmentation
                    train_transform = transforms.Compose(
                        [
                            transforms.RandomHorizontalFlip(),
                            transforms.RandomCrop(32, padding=4),
                            val_transform,  # Apply base transform at the end
                        ]
                    )

                train_dataset = datasets.CIFAR10(
                    root="./data", train=True, download=True, transform=train_transform
                )
                val_dataset = datasets.CIFAR10(
                    root="./data", train=False, download=True, transform=val_transform
                )
                return train_dataset, val_dataset
            case "two-layer-perceptron" | "single-neuron-perceptron":
                dataset = LinearRegressionDataset(
                    g.data_size, perceptron_settings.input_size
                )
                # Split dataset into training and validation sets
                train_size = int((1 - g.val_data_fraction) * len(dataset))
                val_size = len(dataset) - train_size
                return torch.utils.data.random_split(dataset, [train_size, val_size])
            case _:
                raise ValueError(f"Unsupported model type: {model_type}")
