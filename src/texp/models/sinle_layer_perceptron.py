import torch.nn as nn


class SingleNeuronPerceptron(nn.Module):
    def __init__(self):
        super(SingleNeuronPerceptron, self).__init__()
        self.linear = nn.Linear(1, 1)  # One input and one output

    def forward(self, x):
        return self.linear(x)
