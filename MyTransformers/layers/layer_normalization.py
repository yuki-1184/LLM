import torch
import torch.nn as nn

class LayerNormalization(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))  # 学習可能なスケーリングパラメータ -> 1.0
        self.beta = nn.Parameter(torch.zeros(d_model))  # 学習可能なバイアスパラメータ -> 0.0
        self.eps = eps
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.gamma * norm_x + self.beta


"""
層正規化
過剰に大きい値によって勾配が消失するのを防ぐために、各層の出力を正規化します。
層正規化は、Transformerの各層で使用され、入力テンソルの各要素をその平均と分散で正規化します。

🔸 gamma = 1（スケーリング初期値）
	•	正規化後の出力は平均0・分散1のデータになります。
	•	ここに gamma=1 をかけると「そのままのスケール」を維持する。
	•	→ 正規化後の値を変えずに出力に流せる。

🔸 beta = 0（バイアス初期値）
	•	出力にバイアスを加えない（平均0のまま）。
	•	→ 非正規化方向への変化は起きない。
 
Pre-LN (BERT, GPT系。　最近はこっちが主流)
層正規化　→ 任意のレイヤー　→ 残差結合
この方が学習が安定するらしい

Post-LN (元祖Transformer論文)
任意のレイヤー　→ 層正規化＋残差結合
こっちは学習が不安定らしい
"""