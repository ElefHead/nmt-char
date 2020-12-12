import torch
from torch import nn
from torch.nn.functional import softmax


class Attention(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super(Attention, self).__init__()
        self.linear = nn.Linear(
            in_features=in_features,
            out_features=out_features,
            bias=False
        )

    def forward(self,
                enc_hidden: torch.Tensor,
                dec_hidden_t: torch.Tensor,
                enc_masks: torch.Tensor = None) -> torch.Tensor:

        enc_projection = self.linear(enc_hidden)
        dec_hidden_unsqueezed_t = dec_hidden_t.unsqueeze(dim=2)
        score = enc_projection.bmm(dec_hidden_unsqueezed_t)
        score = score.squeeze(dim=2)

        if enc_masks:
            score.data.masked_fill_(
                enc_masks.byte().to(torch.bool),
                -float('inf')
            )

        attention_weights = softmax(score, dim=1)
        attention_weights = attention_weights.unsqueeze(dim=1)

        context_vector = attention_weights.bmm(enc_hidden)
        context_vector = context_vector.squeeze(dim=1)

        return attention_weights, context_vector
