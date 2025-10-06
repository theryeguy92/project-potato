# scripts/test_mla.py
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.mla import MultiHeadLatentAttention
import torch
import yaml

# Load config
with open('configs/model_config.yaml', 'r') as f:
    config = yaml.safe_load(f)

print(torch.__version__, torch.cuda.is_available(), torch.version.cuda)
print(torch.cuda.get_device_name(0))

model = MultiHeadLatentAttention(
    hidden_size=config['hidden_size'],
    num_heads=config['num_heads'],
    latent_dim=config['latent_dim'],
    dropout=config['dropout'],
    max_position_embeddings=config['max_position_embeddings'],
    base=config['base']
).cuda()

x = torch.randn(1, 128, config['hidden_size']).cuda()
attn_mask = torch.triu(torch.ones(128, 128) * float('-inf'), diagonal=1).cuda()
output = model(x, attn_mask)
print(f"Output shape: {output.shape}")
print(f"Memory used: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")