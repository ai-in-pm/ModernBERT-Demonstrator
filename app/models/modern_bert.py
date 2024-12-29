import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class GeGLU(nn.Module):
    """GeGLU activation function as described in the ModernBERT paper."""
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, x):
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * F.gelu(gate)

class RotaryPositionalEmbedding(nn.Module):
    """Rotary positional embeddings (RoPE) implementation."""
    def __init__(self, dim, max_seq_len=8192):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.max_seq_len = max_seq_len
        self.seq_len_cached = None
        self.cos_cached = None
        self.sin_cached = None

    def forward(self, x, seq_len=None):
        if seq_len > self.max_seq_len:
            raise ValueError(f"Sequence length {seq_len} exceeds maximum length {self.max_seq_len}")
        
        if seq_len != self.seq_len_cached:
            self.seq_len_cached = seq_len
            t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
            freqs = torch.einsum("i,j->ij", t, self.inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1).unsqueeze(0)
            self.cos_cached = emb.cos()
            self.sin_cached = emb.sin()

        return self.cos_cached, self.sin_cached

class AttentionBlock(nn.Module):
    """Implements alternating global/local attention patterns."""
    def __init__(self, dim, num_heads=8, head_dim=64, dropout=0.1, window_size=256):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.window_size = window_size
        
        self.qkv = nn.Linear(dim, 3 * num_heads * head_dim)
        self.proj = nn.Linear(num_heads * head_dim, dim)
        self.dropout = nn.Dropout(dropout)
        
        self.rope = RotaryPositionalEmbedding(head_dim)
        self.scale = head_dim ** -0.5

    def forward(self, x, is_global=False):
        B, L, _ = x.shape
        
        # QKV projection
        qkv = self.qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: t.view(B, L, self.num_heads, self.head_dim), qkv)
        
        # Apply RoPE
        cos, sin = self.rope(q, L)
        q, k = map(lambda t: self._apply_rotary_pos_emb(t, cos, sin), (q, k))
        
        # Attention pattern based on global/local flag
        if is_global:
            attn_output = self._global_attention(q, k, v)
        else:
            attn_output = self._local_attention(q, k, v)
        
        # Output projection
        output = self.proj(attn_output.view(B, L, -1))
        return self.dropout(output)

    def _apply_rotary_pos_emb(self, t, cos, sin):
        return t * cos + self._rotate_half(t) * sin

    def _rotate_half(self, x):
        x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
        return torch.cat((-x2, x1), dim=-1)

    def _global_attention(self, q, k, v):
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        return attn @ v

    def _local_attention(self, q, k, v):
        B, L, H, D = q.shape
        
        # Implement sliding window attention
        pad_size = (self.window_size - L % self.window_size) % self.window_size
        if pad_size > 0:
            padding = torch.zeros(B, pad_size, H, D, device=q.device)
            q = torch.cat([q, padding], dim=1)
            k = torch.cat([k, padding], dim=1)
            v = torch.cat([v, padding], dim=1)
        
        # Reshape for local attention
        new_len = q.shape[1]
        windows = new_len // self.window_size
        q = q.view(B, windows, self.window_size, H, D)
        k = k.view(B, windows, self.window_size, H, D)
        v = v.view(B, windows, self.window_size, H, D)
        
        # Calculate attention within windows
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        output = attn @ v
        
        # Reshape back
        output = output.view(B, -1, H, D)
        if pad_size > 0:
            output = output[:, :-pad_size]
        
        return output

class ModernBERTBlock(nn.Module):
    """Main transformer block with alternating attention patterns."""
    def __init__(self, dim, num_heads=8, mlp_ratio=4, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = AttentionBlock(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            GeGLU(dim, dim * mlp_ratio // 2),
            nn.Dropout(dropout),
            nn.Linear(dim * mlp_ratio // 2, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, is_global=False):
        x = x + self.attn(self.norm1(x), is_global)
        x = x + self.mlp(self.norm2(x))
        return x

class ModernBERT(nn.Module):
    """Complete ModernBERT implementation."""
    def __init__(
        self,
        vocab_size=50257,
        max_seq_len=8192,
        dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        dropout=0.1
    ):
        super().__init__()
        self.max_seq_len = max_seq_len
        
        self.token_emb = nn.Embedding(vocab_size, dim)
        self.dropout = nn.Dropout(dropout)
        
        self.blocks = nn.ModuleList([
            ModernBERTBlock(dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])
        
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab_size, bias=False)
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def forward(self, x):
        B, L = x.shape
        if L > self.max_seq_len:
            raise ValueError(f"Sequence length {L} exceeds maximum length {self.max_seq_len}")
        
        x = self.token_emb(x)
        x = self.dropout(x)
        
        # Alternate between global and local attention
        for i, block in enumerate(self.blocks):
            x = block(x, is_global=(i % 2 == 0))
        
        x = self.norm(x)
        return self.head(x)

    def get_num_params(self):
        """Return the number of parameters in the model."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
