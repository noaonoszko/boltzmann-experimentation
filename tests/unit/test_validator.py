import torch
from boltz.models import DenseNet121
from boltz.training.validator import Validator
from boltz.training.loss import ExactLoss
from boltz.training.model import Model
from boltz.config.settings import general_settings as g

NUM_COMMUNICATED_PARAMETERS = 100


def setup_validator():
    g.model_kwargs["norm"] = "batch"
    torch_model = DenseNet121()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(torch_model.parameters())
    wrapped_model = Model(torch_model, optimizer, criterion)
    g.compression_factor = wrapped_model.num_params() // NUM_COMMUNICATED_PARAMETERS
    return Validator(
        model=wrapped_model,
        compression_factor=g.compression_factor,
        loss_calculator=ExactLoss(),
    )


def test_set_included_param_indices():
    validator = setup_validator()
    validator.set_included_param_indices()
    total_params = validator.model.num_params()
    included_params = len(validator.included_param_indices)

    assert (
        included_params < total_params
    ), "All parameters were included despite excluding BatchNorm"


def test_aggregate_slices():
    validator = setup_validator()
    slice_data = torch.rand_like(validator.slice_indices, dtype=torch.float32)
    miner_slices = [slice_data for _ in range(5)]
    aggregated_model = validator.model.aggregate_slices(
        miner_slices, validator.slice_indices
    )
    assert len(validator.slice_indices) == NUM_COMMUNICATED_PARAMETERS

    params = torch.nn.utils.parameters_to_vector(aggregated_model.parameters())
    param_sum = params[validator.slice_indices].sum()
    expected_sum = slice_data.sum().item()
    assert torch.isclose(
        param_sum, torch.tensor(expected_sum), atol=1e-4
    ), "Aggregated parameters do not match expected values"
