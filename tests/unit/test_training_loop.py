from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.testing
from texp.config.settings import general_settings as g
from texp.training.loss import ExactLoss
from texp.training.miner import Miner
from texp.training.model import Model
from texp.training.training_loop import TrainingLoop
from texp.training.validator import Validator

TrainingLoopFixture = tuple[TrainingLoop, Validator, list[Miner]]


@pytest.fixture
def setup_training_loop() -> TrainingLoopFixture:
    """Fixture to create a TrainingLoop instance with mock dependencies."""
    # Configure general settings
    g.num_comrounds = 2  # Set for test control
    g.compression_factor = 2

    # Set up the Validator with mock Model and ExactLoss
    validator_model = MagicMock(spec=Model)
    validator_model.torch_model = MagicMock(spec=nn.Module)
    validator_model.num_params.return_value = (
        100  # Return an integer for num_params to avoid empty array issues
    )
    loss_calculator = ExactLoss()

    # Initialize Validator with mock model and set included_param_indices
    validator = Validator(
        validator_model,
        compression_factor=g.compression_factor,
        loss_calculator=loss_calculator,
    )
    validator.included_param_indices = np.arange(validator_model.num_params())
    validator.slices = {
        1: MagicMock(),
        2: MagicMock(),
    }  # Mock slices with miner IDs as keys

    # Set up Miners with mock Models
    miner_model1 = MagicMock(spec=Model)
    miner_model1.torch_model = MagicMock()
    miner_model2 = MagicMock(spec=Model)
    miner_model2.torch_model = MagicMock()
    miner1 = Miner(model=miner_model1, id=1)
    miner2 = Miner(model=miner_model2, id=2)
    miners = [miner1, miner2]

    # Mock data loaders
    train_loader = iter([MagicMock(), MagicMock()])
    val_loader = iter([MagicMock()])  # Add additional items if needed

    # Create and return TrainingLoop instance with only required arguments
    training_loop = TrainingLoop(
        validator=validator,
        miners=miners,
        infinite_train_loader=train_loader,
        infinite_val_loader=val_loader,
    )
    return training_loop, validator, miners


def test_miner_local_step(setup_training_loop: TrainingLoopFixture):
    """Test miner training step and logging output."""
    training_loop, validator, miners = setup_training_loop
    miner = miners[0]

    # Patch the 'next' call on infinite_train_loader to return expected_data
    with patch.object(validator, "add_miner_slice") as mock_add_miner_slice:
        # Mock behavior for miner's methods
        miner.get_slice_from_indices = MagicMock(
            return_value=MagicMock(data=torch.tensor([1, 2, 3]))
        )
        miner.model.num_params.return_value = 100

        # Run the train step
        training_loop.miner_local_step(miner)

        # Verify train_step is called with miner's data
        miner.model.train_step.assert_called_once_with(miner.data)

        # Verify slice was retrieved and add_miner_slice was called
        miner.get_slice_from_indices.assert_called_once_with(
            training_loop.validator.slice_indices
        )
        mock_add_miner_slice.assert_called_once_with(
            miner.get_slice_from_indices.return_value
        )


def test_calculate_scores(setup_training_loop: TrainingLoopFixture):
    training_loop, validator, miners = setup_training_loop
    validator.calculate_miner_score = MagicMock()

    training_loop.calculate_scores()

    for miner in miners:
        validator.calculate_miner_score.assert_any_call(
            validator.slices[miner.id], miner.id, miner.data
        )


def test_aggregate_and_validate(setup_training_loop: TrainingLoopFixture):
    """Test that the model slices are aggregated and validation is performed."""
    training_loop, validator, _ = setup_training_loop

    # Mock aggregate_slices and val_step methods
    validator.model.aggregate_slices = MagicMock(return_value=MagicMock())
    validator.model.val_step = MagicMock()

    # Patch 'next' to control what is returned by infinite_val_loader
    expected_val_batch = MagicMock()  # Mocked validation batch
    with patch("builtins.next", return_value=expected_val_batch):
        # Run the aggregate_and_validate step
        training_loop.aggregate_and_validate()

        # Verify aggregate_slices and val_step were called with the correct parameters
        validator.model.aggregate_slices.assert_called_once_with(
            slices=list(validator.slices.values()),
            slice_indices=validator.slice_indices,
        )
        validator.model.val_step.assert_called_once_with(expected_val_batch)


def test_update_miners(setup_training_loop: TrainingLoopFixture):
    """Test that miners are updated with the validator's model parameters."""
    training_loop, validator, miners = setup_training_loop

    # Define slice_indices within bounds of the mock parameters vector size
    validator.slice_indices = np.array([0, 1, 2])

    # Mock parameters_to_vector to return a tensor of at least the size needed for slice_indices
    with patch(
        "torch.nn.utils.parameters_to_vector",
        return_value=torch.tensor([0.1, 0.2, 0.3]),
    ) as mock_params_to_vector:
        # Ensure each miner's model has an lr_scheduler attribute for testing
        for miner in miners:
            miner.model.lr_scheduler = MagicMock()  # Mock lr_scheduler

        # Run the update_miners function
        training_loop.update_miners()

        # Verify that each miner was updated correctly
        for miner in miners:
            validator_slice = mock_params_to_vector.return_value[
                validator.slice_indices
            ]

            # Use torch.testing.assert_close to compare tensors
            actual_call_args = miner.model.update_with_slice.call_args[0]
            torch.testing.assert_close(
                actual_call_args[0], validator_slice
            )  # Compare tensors
            assert (
                actual_call_args[1] == validator.slice_indices
            ).all()  # Compare indices

            # Check if lr_scheduler step is called if lr_scheduler is not None
            if miner.model.lr_scheduler:
                miner.model.lr_scheduler.step.assert_called_once()


def test_run(setup_training_loop: TrainingLoopFixture):
    """Test the entire training loop runs without errors."""
    training_loop, validator, _ = setup_training_loop

    with patch.object(
        validator, "reset_slices_and_indices"
    ) as mock_reset_validator, patch.object(
        training_loop, "miner_local_step", return_value=None
    ) as mock_miner_local_step, patch.object(
        training_loop, "calculate_scores"
    ) as mock_calculate_scores, patch.object(
        training_loop,
        "aggregate_and_validate",
        return_value=0.85,  # Mock a real float for accuracy
    ) as mock_aggregate_and_validate, patch.object(
        training_loop, "update_miners"
    ) as mock_update_miners:
        # Run the training loop
        training_loop.run()

        # Verify each method in the loop was called the expected number of times
        assert mock_reset_validator.call_count == g.num_comrounds
        assert mock_miner_local_step.call_count == g.num_comrounds * len(
            training_loop.miners
        )
        assert mock_calculate_scores.call_count == g.num_comrounds
        assert mock_aggregate_and_validate.call_count == g.num_comrounds
        assert mock_update_miners.call_count == g.num_comrounds
