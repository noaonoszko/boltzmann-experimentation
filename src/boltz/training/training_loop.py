from typing import Iterator

import torch
from boltz.config.settings import general_settings as g
from boltz.training.miner import Miner
from boltz.training.validator import Validator
from boltz.utils.logger import (
    general_logger,
)
from tqdm import trange


class TrainingLoop:
    def __init__(
        self,
        *,
        validator: Validator,
        miners: list[Miner],
        infinite_train_loader: Iterator[tuple[torch.Tensor, torch.Tensor]],
        infinite_val_loader: Iterator[tuple[torch.Tensor, torch.Tensor]],
    ):
        self.validator = validator
        self.miners = miners
        self.infinite_train_loader = infinite_train_loader
        self.infinite_val_loader = infinite_val_loader
        self.general_logger = general_logger

    def miner_local_step(self, miner):
        miner.data = next(self.infinite_train_loader)
        miner.model.train_step(miner.data)
        slice = miner.get_slice_from_indices(self.validator.slice_indices)
        self.log_miner_details(miner, slice)
        self.validator.add_miner_slice(slice)

    def log_miner_details(self, miner, slice):
        self.general_logger.debug(
            f"Total number of parameters: {miner.model.num_params()}"
        )
        self.general_logger.debug(
            f"Selected indices after compressing {g.compression_factor}x:"
            f"{len(slice.data)}, i.e. "
            f"{len(slice.data)/miner.model.num_params()*100:.1f}%"
        )

    def calculate_scores(self):
        for miner in self.miners:
            self.validator.calculate_miner_score(
                self.validator.slices[miner.id],
                miner.id,
                miner.data,
            )

    def aggregate_and_validate(self):
        self.validator.model.torch_model = self.validator.model.aggregate_slices(
            slices=list(self.validator.slices.values()),
            slice_indices=self.validator.slice_indices,
        )
        val_batch = next(self.infinite_val_loader)
        return self.validator.model.val_step(val_batch)

    def update_miners(self):
        validator_model_params = torch.nn.utils.parameters_to_vector(
            self.validator.model.torch_model.parameters()
        )
        for miner in self.miners:
            validator_slice = validator_model_params[self.validator.slice_indices]
            miner.model.update_with_slice(validator_slice, self.validator.slice_indices)
            if miner.model.lr_scheduler is not None:
                miner.model.lr_scheduler.step()

    def run(self):
        for _ in (t := trange(g.num_communication_rounds)):
            self.validator.reset_slices_and_indices()
            for miner in self.miners:
                self.miner_local_step(miner)
            self.calculate_scores()
            accuracy = self.aggregate_and_validate()
            t.set_description(f"Accuracy: {accuracy:.2%}")
            self.update_miners()
