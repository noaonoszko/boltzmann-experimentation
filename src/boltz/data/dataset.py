import torch
from torch.utils.data import Dataset


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
