import torch
import torch.nn as nn

from .embedding import Embedding
from .positional_encoder import PositionalEncoder
from .scaled_dot_product_attention import ScaledDotProductAttention
from .multihead_attention import MultiHeadAttention
from .feed_forward_network import FeedForwardNetwork
from .layer_normalization import (
    LayerNormalization,
    ResidualNormalizationWrapper
)   

class Encoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int,    # d_model, hidden_dim 埋め込みの次元数
        num_heads: int,
        num_layers: int = 6,  # デフォルトで6層のエンコーダ
        dropout: float = 0.1,
        max_seq_len: int = 512,  # 最大シーケンス長
        eps: float = 1e-6
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.d_model = d_model
        
        self.embedding = Embedding(vocab_size, d_model)
        self.positional_encoder = PositionalEncoder(d_model, max_seq_len)
        self.dropout = nn.Dropout(dropout)
        self.self_attns = nn.ModuleList([
            MultiHeadAttention(d_model, num_heads, dropout)
            for _ in range(num_layers)
        ])
        self.ffns = nn.ModuleList([
            FeedForwardNetwork(d_model, d_model * 4, dropout)  # d_ff = 4 * d_model
            for _ in range(num_layers)
        ])
        self.attn_norms = nn.ModuleList([
            LayerNormalization(d_model, eps) 
            for _ in range(num_layers)
        ])
        self.ffn_norms = nn.ModuleList([
            LayerNormalization(d_model, eps) 
            for _ in range(num_layers)
        ])
        self.attn_dropouts = nn.ModuleList([
            nn.Dropout(dropout) 
            for _ in range(num_layers)
        ])
        self.ffn_dropouts = nn.ModuleList([
            nn.Dropout(dropout) 
            for _ in range(num_layers)
        ])
        self.output_norm = LayerNormalization(d_model, eps)
        
    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        入力テンソル x に対して、エンコーダの順伝播を実行します。
        :param x: 入力テンソル (B, T)
        :param mask: マスクテンソル (B, T, T)
        :return: エンコードされたテンソル (B, T, d_model)
        """
        x = self.embedding(x)
        x = self.positional_encoder(x)
        x = self.dropout(x)
        
        for i in range(self.num_layers):
            # Self-Attention Layer with Pre-LN and Residual Connection
            normed_x = self.attn_norms[i](x)
            attn_out, _ = self.self_attns[i](normed_x, normed_x, normed_x, mask=mask)
            attn_out = self.attn_dropouts[i](attn_out)
            x = x + attn_out
            
            # Feed Forward Network with Pre-LN and Residual Connection
            normed_x = self.ffn_norms[i](x)
            ffn_out = self.ffns[i](normed_x)
            ffn_out = self.ffn_dropouts[i](ffn_out)
            x = x + ffn_out
        
        x = self.output_norm(x)
        return x
        