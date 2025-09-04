import torch
import torch.nn as nn

class ANNBinaryVLE(nn.Module):
    def __init__(self, input_dim=3, hidden_sizes=(64, 64, 32), dropout=0.05):
        super().__init__()
        layers = []
        last = input_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(last, h))
            layers.append(nn.ReLU())
            if dropout and dropout > 0:
                layers.append(nn.Dropout(dropout))
            last = h
        layers.append(nn.Linear(last, 1))
        layers.append(nn.Sigmoid())  # y1 in [0,1]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
