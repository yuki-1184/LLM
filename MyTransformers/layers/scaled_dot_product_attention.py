from typing import Tuple
import torch
import torch.nn as nn

class ScaledDotProductAttention(nn.Module):
    def __init__(self, head_dim, dropout=0.1):
        super().__init__()
        self.d_k = head_dim  # モデルの次元数 (スケーリングに使用)
        self.dropout = nn.Dropout(dropout)  # ドロップアウト層の初期化
    
    def forward(
        self, 
        query: torch.Tensor,  # クエリテンソル (B, H, Q_len, d_k)
        key: torch.Tensor,    # キーテンソル (B, H, K_len, d_k)
        value: torch.Tensor,  # バリューテンソル (B, H, K_len, d_v)
        mask: torch.Tensor = None  # マスクテンソル (B, H, Q_len, K_len)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        scalar = self.d_k ** 0.5
        attn_logits = torch.matmul(query, key.transpose(-2, -1))    # (B, H, Q_len, K_len)
        
        if mask is not None:
            if mask.dim() != attn_logits.dim():
                raise ValueError(f"Mask dimension {mask.dim()} does not match attention logits dimension {attn_logits.dim()}.")
            attn_logits = attn_logits.masked_fill(mask == 0, float('-inf'))
        
        attn_weights = torch.softmax(attn_logits / scalar, dim=-1)
        attn_weights = self.dropout(attn_weights)    # (B, H, Q_len, K_len)
        attn_output = torch.matmul(attn_weights, value)    # (B, H, Q_len, d_v)
        return attn_output, attn_weights
        

"""
Example usage:
        
query = torch.tensor([[[[1.0, 0.0, 1.0, 0.0],   # query 1
                        [0.0, 1.0, 0.0, 1.0]]]], # query 2
                     dtype=torch.float32)  # shape: (1, 1, 2, 4)

key = torch.tensor([[[[1.0, 0.0, 0.0, 1.0],     # key 1
                      [0.0, 1.0, 1.0, 0.0],     # key 2
                      [1.0, 1.0, 0.0, 0.0]]]],  # key 3
                   dtype=torch.float32)  # shape: (1, 1, 3, 4)

# 転置することで key: (1, 1, 4, 3)
key_t = key.transpose(-2, -1)

# 内積をとる
attn_logits = torch.matmul(query, key_t)
print(attn_logits.shape)  # (1, 1, 2, 3)
print(attn_logits)

tensor([[[[1., 1., 1.],
          [1., 1., 1.]]]])

attn_logits.shape = (batch_size, num_heads, query_len, key_len)

attn_weights = torch.softmax(attn_logits, dim=-1)

QK^T = [[ 1.0, 2.0, 3.0],
        [ 2.0, 3.0, 4.0],
        [ 3.0, 4.0, 5.0]]

mask = [[1, 1, 0],
        [1, 1, 0],
        [1, 1, 0]]

masked_logits =
[[ 1.0, 2.0, -inf],
 [ 2.0, 3.0, -inf],
 [ 3.0, 4.0, -inf]]

attn_weights ≈
[[0.27, 0.73, 0.0],
 [0.27, 0.73, 0.0],
 [0.27, 0.73, 0.0]]
"""