"""
Transformer encoder (pre-LN, encoder-only, no CLS token) and CTC output head
for HTR-VT.

4 layers, dim 768, 6 heads, FFN hidden
dim 3072, GELU activation.
"""

import torch
import torch.nn as nn

class TransformerEncoderBlock(nn.Module):
    def __init__(self, dim=768, num_heads=6, mlp_hidden_dim=3072, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Linear(mlp_hidden_dim, dim),
        )

    def forward(self, x):
        # Pre-LN: normalize BEFORE the sublayer, residual add AFTER
        normed = self.norm1(x)
        attn_out, _ = self.attn(normed, normed, normed, need_weights=False)
        y = x + attn_out

        normed2 = self.norm2(y)
        mlp_out = self.mlp(normed2)
        out = y + mlp_out
        return out


class TransformerEncoder(nn.Module):
    def __init__(self, dim=768, num_heads=6, mlp_hidden_dim=3072, num_layers=4, dropout=0.0):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerEncoderBlock(dim, num_heads, mlp_hidden_dim, dropout)
            for _ in range(num_layers)
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class CTCHead(nn.Module):
    """Projects each token's dim-C representation to per-character logits.
    num_classes should be vocab size WITHOUT blank; blank is added as one
    extra output unit (matching build_vocab's blank_idx = len(alphabet))."""

    def __init__(self, dim=768, num_classes=79):
        super().__init__()
        self.head = nn.Linear(dim, num_classes + 1)  # +1 for CTC blank

    def forward(self, x):
        logits = self.head(x)                       # [B, L, num_classes+1]
        log_probs = torch.log_softmax(logits, dim=-1)
        return log_probs

