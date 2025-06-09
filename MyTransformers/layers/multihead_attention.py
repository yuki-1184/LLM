import torch
import torch.nn as nn
from .scaled_dot_product_attention import ScaledDotProductAttention

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0
        self.head_dim = d_model // num_heads  # d_k
        self.num_heads = num_heads
        self.d_model = d_model
        
        # nn.Parameterでやってみると面白いかも
        self.q_linear = nn.Linear(d_model, d_model) # W_q * input
        self.k_linear = nn.Linear(d_model, d_model) # W_k * input
        self.v_linear = nn.Linear(d_model, d_model) # W_v * input
        self.attention = ScaledDotProductAttention(self.head_dim, dropout=0.1)
        self.out_linear = nn.Linear(d_model, d_model)        

    def forward(self, q, k, v, mask=None):
        batch_size, seq_len, _ = q.size()
        
        # クエリ、キー、バリューを線形変換
        Q = self.q_linear(q)
        K = self.k_linear(k)
        v = self.v_linear(v)
        
        # Q, K, Vをヘッド数に分割 (batch_size, seq_len, d_model) -> (batch_size, num_heads, seq_len, head_dim)
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, Q_len, d_k)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, K_len, d_k)
        V = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, K_len, d_v)
        
        if mask is not None:
            # mask: (batch_size, seq_len, seq_len) -> (batch_size, num_heads, seq_len, seq_len)
            mask = mask.unsqueeze(1).repeat(1, self.num_heads, 1, 1)
        
        attn_output, attn_weights = self.attention(Q, K, V, mask)
        
        # (B, H, Q_len, d_k) → (B, Q_len, D)
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.d_model
        )
        
        output = self.out_linear(attn_output)
        
        return output, attn_weights
        
        
        
"""
Pytorchについて
__init__ 
-> 初期化メソッド。クラスのインスタンスが生成されるときに呼び出され、オブジェクトの属性を設定します。
   定数、学習されるパラメータ、レイヤーなどを定義します。

forward()
-> 順伝播メソッド。入力データを受け取り、モデルの出力を計算します。
   入力テンソルや演算の処理ロジックはここに定義する

--------------------------------------------------------------------------------------

"私" → [64次元①, 64次元②, ..., 64次元⑧]
"は" → [64次元①, 64次元②, ..., 64次元⑧]

各64次元が「綺麗に」位置情報・意味的関係などに 明確に分かれるわけではない です。
	•	各ヘッドはランダム初期化された W_Q, W_K, W_V（線形変換）を持つ
	•	その中で 学習を通じて それぞれのヘッドが「得意な注意の方向性」を見つけてい
 
 -> 各ヘッドは、64次元の中で 独自のパターンを学習する。それがたまたま位置っぽいことだったり、意味的関係だったりする。
 
--------------------------------------------------------------------------------------

W_Q, W_K, W_V (学習されるパラメータ)はnn.Linearで定義されてる

nn.Linear(in_features, out_features) は 内部で重み W（およびバイアス）を自動的に定義し、学習対象に含めてくれるモジュール だからです。
self.q_linear = nn.Linear(d_model, d_model)
# → 中では self.weight: (d_model, d_model), self.bias: (d_model,) が作られてる

for name, param in model.named_parameters():
    print(name, param.shape)

multihead.q_linear.weight torch.Size([512, 512])
multihead.q_linear.bias torch.Size([512])
multihead.k_linear.weight torch.Size([512, 512])
"""