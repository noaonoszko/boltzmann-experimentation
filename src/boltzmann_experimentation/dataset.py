import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms

from boltzmann_experimentation.training.model import MODEL_TYPE
from boltzmann_experimentation.settings import (
    general_settings as g,
    perceptron_settings,
)
from timm.data import create_transform


def infinite_data_loader_generator(dataset: Dataset[torch.Tensor], train: bool = True):
    """Infinite data loader with automatic reshuffling every 'epoch_length' batches."""
    epoch_length = len(dataset) // g.batch_size_train
    while True:
        # Create a new DataLoader with shuffled data
        loader = DataLoader(
            dataset,
            batch_size=g.batch_size_train if train else g.batch_size_val,
            num_workers=g.num_workers_dataloader,
            shuffle=True,
        )

        # Yield each batch in the current shuffled loader
        for i, (features, targets) in enumerate(loader):
            yield features.to(g.device), targets.to(g.device)
            # After reaching epoch_length, reshuffle by breaking and creating a new DataLoader
            if i + 1 == epoch_length:
                break


class DatasetFactory:
    @staticmethod
    def create_dataset(model_type: MODEL_TYPE) -> tuple[Dataset, Dataset]:
        match model_type:
            case "simple-cnn" | "resnet18" | "densenet" | "deit-b":
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


class LinearRegressionDataset(Dataset):
    def __init__(self, num_samples: int, num_features: int):
        # Generate features and targets
        self.features, self.targets = self._generate_linear_data(
            num_samples, num_features
        )

    def _generate_linear_data(
        self, num_samples: int, num_features: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Generate random features
        X = torch.rand(num_samples, num_features)

        # Generate random coefficient(s) for linear relationship
        coefficients = (torch.rand(num_features) - 0.5) * 10
        coefficients = 10 * torch.ones_like(coefficients)

        # Normalize features
        X = (X - X.mean(dim=0)) / X.std(dim=0)

        # Generate targets
        y = X @ coefficients + torch.randn(num_samples) * 1
        return X, y.unsqueeze(1)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]
