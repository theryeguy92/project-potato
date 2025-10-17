# src/models/transformer.py
from .mla import MultiHeadLatentAttention
import torch.nn as nn

class TransformerBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, latent_dim, ff_dim, dropout, max_position_embeddings=2048, base=10000):
        super().__init__()
        self.attn = MultiHeadLatentAttention(
            hidden_size, 
            num_heads, 
            latent_dim, 
            dropout,
            max_position_embeddings=max_position_embeddings,
            base=base
        )
        self.ff = nn.Sequential(
            nn.Linear(hidden_size, ff_dim),
            nn.GELU(),
            nn.Linear(ff_dim, hidden_size),
            nn.Dropout(dropout)
        )
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)

    def forward(self, x, attn_mask=None):
        x = x + self.attn(self.norm1(x), attn_mask)
        x = x + self.ff(self.norm2(x))
        return x

class MiniLLM(nn.Module):
    def __init__(self, vocab_size, hidden_size, num_layers, num_heads, latent_dim, ff_dim, dropout, max_position_embeddings=2048, base=10000):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_size)
        self.blocks = nn.ModuleList([
            TransformerBlock(
                hidden_size, 
                num_heads, 
                latent_dim, 
                ff_dim, 
                dropout,
                max_position_embeddings=max_position_embeddings,
                base=base
            ) for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(hidden_size)
        self.head = nn.Linear(hidden_size, vocab_size)

    def forward(self, input_ids, attn_mask=None):
        x = self.embed(input_ids)
        for block in self.blocks:
            x = block(x, attn_mask)
        x = self.norm(x)
        return self.head(x)
