import torch
from boltzmann_experimentation.config.settings import (
    general_settings as g,
)
from torch.utils.data import DataLoader, Dataset


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
