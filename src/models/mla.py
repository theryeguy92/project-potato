# src/models/mla.py
import torch
import torch.nn as nn
import math

class DeepSeekRotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self._set_cos_sin_cache(seq_len=max_position_embeddings)

    def _set_cos_sin_cache(self, seq_len):
        t = torch.arange(seq_len).float()
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos()[:, :self.dim])
        self.register_buffer("sin_cached", emb.sin()[:, :self.dim])

    def forward(self, x, seq_len=None):
        if seq_len is None:
            seq_len = x.size(1)
        return self.cos_cached[:seq_len].to(x.device), self.sin_cached[:seq_len].to(x.device)

def apply_rotary_emb(x, cos, sin):
    half_dim = x.shape[-1] // 2
    x1, x2 = x[..., :half_dim], x[..., half_dim:]
    cos = cos[..., :half_dim]
    sin = sin[..., :half_dim]
    return torch.cat((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1)

class MultiHeadLatentAttention(nn.Module):
    def __init__(self, hidden_size, num_heads, latent_dim, dropout, max_position_embeddings=2048, base=10000):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.latent_dim = latent_dim

        self.q_down = nn.Linear(hidden_size, latent_dim)
        self.kv_down = nn.Linear(hidden_size, latent_dim)
        self.q_up = nn.Linear(latent_dim, hidden_size)
        self.k_nope_up = nn.Linear(latent_dim, hidden_size // 2)
        self.k_rope_up = nn.Linear(latent_dim, hidden_size // 2)
        self.v_up = nn.Linear(latent_dim, hidden_size)
        self.o_proj = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.rotary_emb = DeepSeekRotaryEmbedding(hidden_size // 2, max_position_embeddings, base)

    def forward(self, hidden_states, attention_mask=None):
        bsz, seq_len, _ = hidden_states.size()

        # Projections
        q_latent = self.q_down(hidden_states)
        kv_latent = self.kv_down(hidden_states)
        q = self.q_up(q_latent)
        k_nope = self.k_nope_up(kv_latent)
        k_rope = apply_rotary_emb(self.k_rope_up(kv_latent), *self.rotary_emb(kv_latent))
        k = torch.cat([k_nope, k_rope], dim=-1)
        v = self.v_up(kv_latent)

        # Reshape for multi-head
        q = q.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Attention
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if attention_mask is not None:
            attn_weights += attention_mask
        attn_weights = torch.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, seq_len, self.hidden_size)
        
        return self.o_proj(attn_output)