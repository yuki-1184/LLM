import torch
import torch.nn as nn

from .embedding import Embedding
from .positional_encoder import PositionalEncoder
from .scaled_dot_product_attention import ScaledDotProductAttention
from .multihead_attention import MultiHeadAttention
from .feed_forward_network import FeedForwardNetwork
from .layer_normalization import LayerNormalization

class Decoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        num_heads: int,
        num_layers: int = 6,
        dropout: float = 0.1,
        max_seq_len: int = 512,
        eps: float = 1e-6
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        
        self.embedding = Embedding(vocab_size, d_model, padding_idx=0)
        self.positional_encoder = PositionalEncoder(d_model, max_seq_len)
        self.dropout = nn.Dropout(dropout)
        
        self.self_attns = nn.ModuleList([
            MultiHeadAttention(d_model, num_heads, dropout)
            for _ in range(num_layers)
        ])
        # Cross-Attention: デコーダがエンコーダの出力を参照するためのアテンション
        self.cross_attns = nn.ModuleList([
            MultiHeadAttention(d_model, num_heads, dropout)
            for _ in range(num_layers)
        ])
        self.ffns = nn.ModuleList([
            FeedForwardNetwork(d_model, d_model * 4, dropout)
            for _ in range(num_layers)
        ])
        self.attn_norms = nn.ModuleList([
            LayerNormalization(d_model, eps)
            for _ in range(num_layers)
        ])
        self.cross_attn_norms = nn.ModuleList([
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
        self.cross_attn_dropouts = nn.ModuleList([
            nn.Dropout(dropout)
            for _ in range(num_layers)
        ])
        self.ffn_dropouts = nn.ModuleList([
            nn.Dropout(dropout)
            for _ in range(num_layers)
        ])
        self.output_norm = LayerNormalization(d_model, eps)
        
    def forward(
        self, 
        x: torch.Tensor, 
        memory: torch.Tensor, 
        src_mask: torch.Tensor = None, 
        tgt_mask: torch.Tensor = None
    ) -> torch.Tensor:
        """
        入力テンソル x に対して、デコーダの順伝播を実行します。
        :param x: 入力テンソル (B, T)
        :param memory: エンコーダからの出力 (B, S, d_model)
        :param src_mask: ソースマスク (B, S, S)
        :param tgt_mask: ターゲットマスク (B, T, T)
        :return: デコードされたテンソル (B, T, d_model)
        """
        x = self.embedding(x)
        x = self.positional_encoder(x)
        x = self.dropout(x)
        
        for i in range(self.num_layers):
            # Self-Attention with Pre-LN and Residual Connection
            normed_x = self.attn_norms[i](x)
            attn_out, _ = self.self_attns[i](normed_x, normed_x, normed_x, tgt_mask) # tgt_mask is used for self-attention to prevent attending to future tokens
            attn_out = self.attn_dropouts[i](attn_out)
            x = x + attn_out
            
            # Cross-Attention with Pre-LN and Residual Connection
            normed_x = self.cross_attn_norms[i](x)
            cross_attn_out, _ = self.cross_attns[i](normed_x, memory, memory, src_mask) # src_mask is used for cross-attention to prevent attending to future tokens in the source sequence
            cross_attn_out = self.cross_attn_dropouts[i](cross_attn_out)
            x = x + cross_attn_out
            
            # Feed Forward Network with Pre-LN and Residual Connection
            normed_x = self.ffn_norms[i](x)
            ffn_out = self.ffns[i](normed_x)
            ffn_out = self.ffn_dropouts[i](ffn_out)
            x = x + ffn_out
        x = self.output_norm(x)
        return x