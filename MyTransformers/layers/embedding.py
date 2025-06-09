import torch
import torch.nn as nn

class Embedding(nn.Module):
    def __init__(self, vocab_size: int, embed_size: int):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embed_size, padding_idx=0)
        self.embed_size = embed_size
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.embedding(x) * (self.embed_size ** 0.5)