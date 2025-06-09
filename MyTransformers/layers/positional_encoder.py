import torch
import torch.nn as nn

class PositionalEncoder(nn.Module):
    def __init__(self, embed_size, max_seq_len=512):
        super().__init__()
        self.embed_size = embed_size    # 埋め込みベクトルの次元数 (m)
        self.max_seq_len = max_seq_len  # 最大シーケンス長 (n)
        pe = torch.zeros(max_seq_len, embed_size)   # (n, m)の0行列の生成
        
        # 位置エンコーディングの計算
        for pos in range(max_seq_len):  # 各位置について
            for i in range(0, embed_size, 2):  # 偶数インデックスと奇数インデックスで異なる計算
                pos_tensor = torch.tensor(pos, dtype=torch.float32)  # 現在の位置をテンソル化
                base = torch.tensor(10000.0, dtype=torch.float32)    # スケーリング係数
                denominator = torch.pow(base, (2 * i) / embed_size)  # 分母の計算
                pe[pos, i] = torch.sin(pos_tensor / denominator)     # 偶数インデックス: sin関数
                pe[pos, i+1] = torch.cos(pos_tensor / denominator)   # 奇数インデックス: cos関数
        
        # バッチ次元を追加して形状を (1, max_seq_len, embed_size) に変更
        pe = pe.unsqueeze(0)    # (batch, max_seq_len, embed_size)
        
        # 計算済みの位置エンコーディングをバッファとして登録 (モデルの一部として保存されるが学習されない)
        self.register_buffer("pe", pe)
        
    def forward(self, x):
        # 入力テンソルに位置エンコーディングを加算
        # x.size(1) は入力シーケンスの長さ
        x = x + self.pe[:, :x.size(1)].detach()  # detach() で勾配計算を無効化
        return x
