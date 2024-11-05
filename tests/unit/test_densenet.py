import pytest
import torch.nn as nn
from unittest.mock import patch
from boltz.models.densenet import get_norm_layer
from typing import Generator

ModelKwargsType = dict[str, str | None]

num_features = 16


@pytest.fixture
def mock_model_kwargs() -> Generator[ModelKwargsType, None, None]:
    with patch(
        "boltz.config.settings.general_settings.model_kwargs", {}
    ) as model_kwargs:
        yield model_kwargs


def test_get_norm_layer_batch(mock_model_kwargs: ModelKwargsType) -> None:
    mock_model_kwargs["norm"] = "batch"
    layer = get_norm_layer(num_features)
    assert isinstance(layer, nn.BatchNorm2d)
    assert layer.num_features == num_features


def test_get_norm_layer_group(mock_model_kwargs: ModelKwargsType) -> None:
    mock_model_kwargs["norm"] = "group"
    layer = get_norm_layer(num_features)
    assert isinstance(layer, nn.GroupNorm)
    assert layer.num_groups == 8
    assert layer.num_channels == num_features


def test_get_norm_layer_identity(mock_model_kwargs: ModelKwargsType) -> None:
    mock_model_kwargs["norm"] = None
    layer = get_norm_layer(num_features)
    assert isinstance(layer, nn.Identity)


def test_get_norm_layer_no_norm_key(mock_model_kwargs: ModelKwargsType) -> None:
    mock_model_kwargs.pop("norm", None)
    layer = get_norm_layer(num_features)
    assert isinstance(layer, nn.Identity)


def test_get_norm_layer_invalid(mock_model_kwargs: ModelKwargsType) -> None:
    mock_model_kwargs["norm"] = "invalid_norm_type"
    with pytest.raises(ValueError, match="Unsupported norm type: invalid_norm_type"):
        get_norm_layer(num_features)
