import torch
import torch.nn as nn

from .embedding import Embedding
from .positional_encoder import PositionalEncoder
from .scaled_dot_product_attention import ScaledDotProductAttention
from .multihead_attention import MultiHeadAttention
from .feed_forward_network import FeedForwardNetwork
from .residual_normalization import (
    LayerNormalization,
    ResidualNormalizationWrapper
)

class Encoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_size: int,    # d_model, hidden_dim 埋め込みの次元数
        num_heads: int,
        num_layers: int = 6,  # デフォルトで6層のエンコーダ
        dropout: float = 0.1,
        max_seq_len: int = 512,  # 最大シーケンス長
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.d_model = embed_size
        
        self.embedding = Embedding(vocab_size, embed_size)
        self.positional_encoder = PositionalEncoder(embed_size, max_seq_len)
        self.dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                'self_attn': ResidualNormalizationWrapper(
                    MultiHeadAttention(embed_size, num_heads, dropout), 
                    embed_size,
                    dropout
                ),
                'ffn': ResidualNormalizationWrapper(
                    FeedForwardNetwork(embed_size, embed_size * 4, dropout), 
                    embed_size,
                    dropout
                )
            }) for _ in range(num_layers)
        ])
        self.output_norm = LayerNormalization(embed_size)
        
    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        入力テンソル x に対して、エンコーダの順伝播を実行します。
        :param x: 入力テンソル (B, T)
        :param mask: マスクテンソル (B, T, T)
        :return: エンコードされたテンソル (B, T, d_model)
        """
        x = self.embedding(x)
        x = self.positional_encoder(x)
        
        for layer in self.layers:
            # Self-Attention
            x = layer['self_attn'](x, x, x, mask=mask)
            
            # Feed Forward Network
            x = layer['ffn'](x)
        
        x = self.output_norm(x)
        return x
        