import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.nn.utils.rnn import pad_sequence


def main():
    # デバイスの設定 (GPUが利用可能ならGPUを使用、そうでなければCPUを使用)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # トレーニングデータ (トークンIDのリスト)
    trainx = [
        torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 1], dtype=torch.long),
        torch.tensor([3, 9, 3, 4, 7], dtype=torch.long),
        torch.tensor([7, 5, 8], dtype=torch.long),
        torch.tensor([1, 5, 8], dtype=torch.long),
        torch.tensor([3, 9, 3, 4, 6], dtype=torch.long),
        torch.tensor([7, 3, 4, 1], dtype=torch.long),
        torch.tensor([1, 3], dtype=torch.long),
        torch.tensor([3, 9, 3, 4, 1], dtype=torch.long),
        torch.tensor([7, 5, 5, 7, 7, 5], dtype=torch.long)
    ]
    print(trainx)

    # ラベル (各シーケンスに対応するクラスラベル)
    traint = np.asarray([0, 0, 0, 1, 1, 1, 2, 2, 2], dtype=np.int64)  # labels

    # 各シーケンス内の最大トークンIDを確認
    for seq in trainx:
        print(max(seq))

    # 語彙サイズの計算 (最大トークンID + 1)
    vocab_size = max(max(seq) for seq in trainx) + 1

    # シーケンスをパディングして同じ長さに揃える
    trainx = pad_sequence(trainx, batch_first=True, padding_value=0)
    traint = torch.tensor(traint, dtype=torch.long).to(device)  # (batch_size, max_seq_len)
    
    embed_size = 16 # 埋め込みベクトルの次元数　→ 任意で設定可能（大きくすれば保持できる情報も増えるが、メモリ・計算コスト・過学習とトレードオフ）
    attention_unit_size = 32 # アテンションユニットのサイズ
    output_size = 3
    minibatch_size = 3
    
    model = Network(vocab_size, embed_size, attention_unit_size, output_size).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())

    # データセットとデータローダーの作成
    dataset = TensorDataset(trainx, traint)
    dataloader = DataLoader(dataset, batch_size=minibatch_size, shuffle=True)

    # トレーニングループ
    for epoch in range(1000 + 1):
        totalcost, totalacc = 0, 0
        for tx, tt in dataloader:  # tx = 入力トークン, tt = ラベル
            optimizer.zero_grad()  # 勾配の初期化
            ty = model(tx)  # モデルの出力 (batch_size, num_classes)
            cost = criterion(ty, tt)  # 損失関数の計算
            totalcost += cost.item()
            tp = ty.argmax(dim=1)  # 予測結果
            totalacc += (tp == tt).sum().item()  # 正解数の計算
            cost.backward()  # 勾配の計算
            optimizer.step()  # パラメータの更新
        totalcost /= len(dataloader)
        totalacc /= len(dataloader.dataset)
        if epoch % 50 ==0:
            print(f"Epoch {epoch}: Cost {totalcost:.4f}, Acc {totalacc:.4f}")
            with torch.no_grad():
                model.eval()
                trainp = model(trainx)
                print(f"Train Predictions: {trainp.argmax(dim=1)}")
    

class Network(nn.Module):
    def __init__(self, vocab_size, embed_size, attention_unit_size, output_size):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embed_size, padding_idx=0)
        self.positional_encoder = PositionalEncoder(embed_size)
        self.attention = nn.MultiheadAttention(embed_size, num_heads=1, batch_first=True)
        self.fc1 = nn.Linear(embed_size, attention_unit_size) # 全結合層1
        self.dropout = nn.Dropout(0.1) # ドロップアウト層
        self.fc2 = nn.Linear(attention_unit_size, output_size) # 全結合層2
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.embedding(x)
        x = self.positional_encoder(x)
        x, _ = self.attention(x, x, x) # アテンション層 3x's because Q, K, V are the same, (attn_output, attn_output_weights)
        x = self.fc1(x[:, 0, :]) # (batch_size, embed_size) -> 1つのトークンのみに対して全結合層を適用
        x = self.relu(x)
        x = self.dropout(x) # dropout -> ReLUはダメ学習が不安定になる可能性あり
        x = self.fc2(x)
        return x

class PositionalEncoder(nn.Module):
    def __init__(self, input_size, max_seq_length=512):
        super().__init__()
        self.input_size = input_size
        self.max_seq_length = max_seq_length
        pe = torch.zeros(max_seq_length, input_size) # 位置エンコーディング用の0行列生成
        for pos in range(max_seq_length):
            for i in range(0, input_size, 2):
                pos_tensor = torch.tensor(pos, dtype=torch.float32)
                base = torch.tensor(10000.0, dtype=torch.float32)
                denominator = torch.pow(base, (2 * i) / input_size)
                pe[pos, i] = torch.sin(pos_tensor / denominator)
                pe[pos, i + 1] = torch.cos(pos_tensor / denominator)
        pe = pe.unsqueeze(0) # (1(batch), max_seq_length, input_size) の形に変形
        self.register_buffer("pe", pe)
        
    def forward(self, x):
        x = x + self.pe[:, :x.size(1)].detach()
        return x

if __name__ == "__main__":
    main()
