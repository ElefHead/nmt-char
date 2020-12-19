from torch import nn
from torch.nn import functional as F


class Generator(nn.Module):
    def __init__(self, in_features: int,
                 out_features: int) -> None:
        super(Generator, self).__init__()
        self.linear = nn.Linear(
            in_features=in_features,
            out_features=out_features,
            bias=False
        )

    def forward(self, x):
        return F.log_softmax(self.linear(x), dim=-1)
