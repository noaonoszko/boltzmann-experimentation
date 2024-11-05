import torch.nn as nn


class TwoLayerPerceptron(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(TwoLayerPerceptron, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(p=0.0)
        self.leaky_relu = nn.LeakyReLU()

    def forward(self, x):
        x = self.dropout(x)
        x = self.leaky_relu(self.fc1(x))
        return self.fc2(x)
