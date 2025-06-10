import torch
import torch.nn as nn

from ..layers.encoder import Encoder
from ..layers.decoder import Decoder

class Transformer(nn.Module):
    def __init__(
        self,
        vocab_size: int,  # 語彙サイズ
        d_model: int,
        num_heads: int,
        num_layers: int = 6,
        dropout: float = 0.1,
        max_seq_len: int = 512,
        eps: float = 1e-6
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.dropout = dropout
        self.max_seq_len = max_seq_len
        self.eps = eps
        
        self.encoder = Encoder(
            vocab_size=vocab_size,  # 仮の値、実際の語彙サイズに合わせて変更
            d_model=d_model,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout,
            max_seq_len=max_seq_len,
            eps=eps
        )
        
        self.decoder = Decoder(
            vocab_size=vocab_size,  # 仮の値、実際の語彙サイズに合わせて変更
            d_model=d_model,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout,
            max_seq_len=max_seq_len,
            eps=eps
        )
        
    def forward(
        self,
        src: torch.Tensor,  # エンコーダへの入力 (B, T)
        tgt: torch.Tensor,  # デコーダへの入力 (B, T)
        src_mask: torch.Tensor = None,  # ソースマスク (B, T, T)
        tgt_mask: torch.Tensor = None   # ターゲットマスク (B, T, T)
    ) -> None:
        """
        Transformerの順伝播を実行します。
        :param src: エンコーダへの入力テンソル (B, S) - ソース系列
        :param tgt: デコーダへの入力テンソル (B, T) - ターゲット系列
        :param src_mask: ソースマスク - パディングトークンをマスク
        :param tgt_mask: ターゲットマスク - 因果マスク（未来のトークンを隠す）
        :return: 出力確率分布 (B, T, tgt_vocab_size)
        """
        encoder_output = self.encoder(src, src_mask)
        decoder_output = self.decoder(tgt, encoder_output, src_mask, tgt_mask)
        
        return decoder_output