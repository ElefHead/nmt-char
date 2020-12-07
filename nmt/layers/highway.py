import torch
from torch import nn
from torch.nn import functional as F


class Highway(nn.Module):
    def __init__(self, in_features: int, out_features: int) -> None:
        super(Highway, self).__init__()
        self.linear = nn.Linear(in_features=in_features,
                                out_features=out_features)
        self.gate = nn.Linear(in_features=in_features,
                              out_features=out_features)

    def forward(self, x: torch.Tensor):
        z = F.relu(self.linear(x))
        t = torch.sigmoid(self.gate(x))

        return t * z + (1 - t) * x
