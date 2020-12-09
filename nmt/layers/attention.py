import torch
from torch import nn
from torch.nn.functional import softmax


class Attention(nn.Module):
    def forward(self,
                enc_hidden: torch.Tensor,
                dec_hidden_t: torch.Tensor,
                enc_masks: torch.Tensor = None) -> torch.Tensor:
        dec_hidden_unsqueezed_t = dec_hidden_t.unsqueeze(dim=2)
        score_t = enc_hidden.bmm(dec_hidden_unsqueezed_t)
        score_t = score_t.squeeze(dim=2)

        if enc_masks:
            score_t.data.masked_fill_(
                enc_masks.byte().to(torch.bool),
                -float('inf')
            )
        
        alpha_t = softmax(score_t, dim=1)
        return alpha_t
