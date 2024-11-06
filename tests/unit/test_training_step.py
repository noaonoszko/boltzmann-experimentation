import torch
from boltz.models.densenet import DenseNet121
from boltz.training.model import Model
from boltz.config.settings import general_settings as g
import torch.optim as optim
import torch.nn as nn


def setup_model():
    model = DenseNet121()  # Adjust based on model definition
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    return Model(torch_model=model, optimizer=optimizer, criterion=criterion)


def test_training_step_reduces_loss():
    wrapped_model = setup_model()
    inputs = torch.randn(10, 3, 32, 32)
    targets = torch.randint(0, 10, (10,))

    initial_loss = wrapped_model.train_step((inputs, targets))
    new_loss = wrapped_model.train_step((inputs, targets))

    assert new_loss < initial_loss, "Loss did not decrease after training step"


def test_validation_step_accuracy_on_unit_interval():
    wrapped_model = setup_model()
    inputs = torch.randn(10, 3, 32, 32)
    targets = torch.randint(0, 10, (10,))
    g.log_to_wandb = False
    accuracy = wrapped_model.val_step((inputs, targets))
    assert 0.0 <= accuracy <= 1.0, "Accuracy is outside expected range (0.0 to 1.0)"
