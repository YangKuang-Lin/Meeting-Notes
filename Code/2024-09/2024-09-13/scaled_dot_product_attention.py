import torch 
import numpy as np

from torch import nn, Tensor
from typing import Optional, Tuple


class ScaledDotProductAttention(nn.Module):
    """
    Scaled Dot-Product Attention proposed in "Attention Is All You Need".

    This function computes the dot products of the query with all keys, divides each by sqrt(dim), 
    and applies a softmax function to obtain the weights on the values.

    Parameters
    ----------
    dim : int
        Dimension of attention.
    mask : torch.Tensor
        Tensor containing indices to be masked.
    query : torch.Tensor
        Tensor of shape (batch, q_len, d_model) containing the projection vector for the decoder.
    key : torch.Tensor
        Tensor of shape (batch, k_len, d_model) containing the projection vector for the encoder.
    value : torch.Tensor
        Tensor of shape (batch, v_len, d_model) containing features of the encoded input sequence.
    mask : torch.Tensor, optional
        Tensor containing indices to be masked.

    Returns
    -------
    context : torch.Tensor
        Tensor containing the context vector from the attention mechanism.
    attn : torch.Tensor
        Tensor containing the attention (alignment) from the encoder outputs.

    Notes
    -----
    .. [1] Srouce Code, https://github.com/sooftware/attentions/blob/master/attentions.py

    """

    def __init__(self, dim: int) -> None:
        super(ScaledDotProductAttention, self).__init__()
        self.sqrt_dim = np.sqrt(dim)

    def forward(self, query: Tensor, key: Tensor, value: Tensor, mask: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        score: torch.Tensor = torch.bmm(query, key.transpose(1, 2)) / self.sqrt_dim

        if mask is not None:
            score.masked_fill_(mask.view(score.size()), -float('Inf'))

        attn = torch.softmax(score, -1)
        context = torch.bmm(attn, value)

        return context, attn