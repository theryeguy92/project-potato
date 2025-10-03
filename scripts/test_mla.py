# scripts/test_mla.py
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.mla import MultiHeadLatentAttention
import torch

model = MultiHeadLatentAttention(hidden_size=512, num_heads=8, latent_dim=128).cuda()
x = torch.randn(1, 128, 512).cuda()
attn_mask = torch.triu(torch.ones(128, 128) * float('-inf'), diagonal=1).cuda()
output = model(x, attn_mask)
print(f"Output shape: {output.shape}")
print(f"Memory used: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")