from torch import nn, Tensor
from typing import Tuple, Optional


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention mechanism proposed in "Attention Is All You Need".

    This implementation allows the model to attend to different subspaces of the 
    input information by performing multiple attention functions in parallel, 
    each with a different learned linear projection. The results of these attention 
    heads are then concatenated and projected again to form the final output.

    Multi-head attention enables the model to jointly attend to information from 
    different representation subspaces at different positions.

    Parameters
    ----------
    d_model : int, optional
        The dimension of the keys, values, and queries. Default is 512.
    num_heads : int, optional
        The number of attention heads. Default is 8.
    query : torch.Tensor
        A tensor of shape (batch, q_len, d_model) representing the queries. In a 
        Transformer model, the queries can come from:
        - The previous decoder layer
        - The input embeddings
        - The output embeddings (masked)
        
    key : torch.Tensor
        A tensor of shape (batch, k_len, d_model) representing the keys. In a 
        Transformer model, the keys can come from:
        - The output of the encoder
        - The input embeddings
        - The output embeddings (masked)
        
    value : torch.Tensor
        A tensor of shape (batch, v_len, d_model) representing the values. In a 
        Transformer model, the values can come from:
        - The output of the encoder
        - The input embeddings
        - The output embeddings (masked)

    mask : torch.Tensor, optional
        A tensor containing indices to be masked.

    Returns
    -------
    output : torch.Tensor
        A tensor of shape (batch, output_len, dimensions) containing the attended output features.
    attn : torch.Tensor
        A tensor of shape (batch * num_heads, v_len) containing the attention (alignment) 
        from the encoder outputs.

    Notes
    -----
    .. [1] Source Code, https://github.com/sooftware/attentions/blob/master/attentions.py        

    """

    def __init__(self, d_model: int = 512, num_heads: int = 8) -> None:
        super(MultiHeadAttention, self).__init__()

        assert d_model % num_heads == 0, "d_model % num_heads should be zero."

        self.d_head = int(d_model / num_heads)
        self.num_heads = num_heads
        self.scaled_dot_attn = ScaledDotProductAttention(self.d_head)
        self.query_proj = nn.Linear(d_model, self.d_head * num_heads)
        self.key_proj = nn.Linear(d_model, self.d_head * num_heads)
        self.value_proj = nn.Linear(d_model, self.d_head * num_heads)

    def forward(self, query: Tensor, key: Tensor, value: Tensor, mask: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        batch_size = value.size(0)

        query = self.query_proj(query).view(batch_size, -1, self.num_heads, self.d_head)
        key = self.key_proj(key).view(batch_size, -1, self.num_heads, self.d_head)
        value = self.value_proj(value).view(batch_size, -1, self.num_heads, self.d_head)

        query = query.permute(2, 0, 1, 3).contiguous().view(batch_size * self.num_heads, -1, self.d_head)
        key = key.permute(2, 0, 1, 3).contiguous().view(batch_size * self.num_heads, -1, self.d_head)
        value = value.permute(2, 0, 1, 3).contiguous().view(batch_size * self.num_heads, -1, self.d_head)

        if mask is not None:
            mask = mask.unsqueeze(1).repeat(1, self.num_heads, 1, 1)

        context, attn = self.scaled_dot_attn(query, key, value, mask)

        context = context.view(self.num_heads, batch_size, -1, self.d_head)
        context = context.permute(1, 2, 0, 3).contiguous().view(batch_size, -1, self.num_heads * self.d_head)

        return context, attn