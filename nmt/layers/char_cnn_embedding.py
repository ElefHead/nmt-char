import torch
from torch import nn
from torch.nn import functional as F


class CharCNNEmbedding(nn.Module):
    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: int = 5) -> None:
        super(CharCNNEmbedding, self).__init__()
        self.conv = nn.Conv1d(in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=kernel_size)
        self.maxpool = nn.AdaptiveMaxPool1d(output_size=1)

    def forward(self, x: torch.Tensor):
        out = F.relu(self.conv(x))
        out = self.maxpool(out)
        return out
