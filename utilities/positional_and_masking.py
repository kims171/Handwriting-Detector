"""
Positional embeddings and span masking for HTR-VT, applied to the [B, L, C]
token sequence produced by the CNN feature extractor.

Order of operations: positional embeddings are added first,
THEN span masking replaces a subset of those (already position-encoded)
tokens with a shared learnable mask token. Masking is training-time only.
"""

import math
import random
import torch
import torch.nn as nn

# ---------- Sinusoidal positional embeddings (fixed, not learned) ----------

class SinusoidalPositionalEmbedding(nn.Module):
    def __init__(self, max_len, dim):
        super().__init__()
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)  # fixed, not a trainable parameter

    def forward(self, x):
        # x: [B, L, C]
        L = x.size(1)
        return x + self.pe[:L].unsqueeze(0)


# ---------- Span masking ----------

class SpanMasking(nn.Module):
    """Replaces contiguous spans of tokens with a shared learnable mask
    embedding. Spans are all exactly `span_length` tokens long. Spans
    are sampled repeatedly (with possible overlap) until the total masked
    fraction reaches `mask_ratio`. Training-time only -- returns the input
    unchanged in eval mode."""

    def __init__(self, dim, mask_ratio=0.4, span_length=8, max_attempts=1000):
        super().__init__()
        self.mask_ratio = mask_ratio
        self.span_length = span_length
        self.max_attempts = max_attempts
        self.mask_token = nn.Parameter(torch.zeros(dim))
        nn.init.normal_(self.mask_token, std=0.02)

    def sample_mask(self, B, L, device):
        """Sample a fresh boolean mask, [B, L]. Separated out from forward()
        so a training loop (e.g. SAM's two-pass step) can sample once and
        reuse the SAME mask across multiple forward calls -- otherwise each
        forward gets independently-random masking, which for SAM specifically
        means its two loss evaluations are no longer measuring sharpness of
        one consistent objective."""
        s = min(self.span_length, L)
        target_count = int(round(self.mask_ratio * L))

        mask = torch.zeros(B, L, dtype=torch.bool, device=device)
        for b in range(B):
            attempts = 0
            while mask[b].sum().item() < target_count and attempts < self.max_attempts:
                start = random.randint(0, L - s)
                mask[b, start:start + s] = True
                attempts += 1
        return mask

    def apply_mask(self, x, mask):
        B, L, C = x.shape
        out = x.clone()
        num_masked_total = mask.sum().item()
        out[mask] = self.mask_token.to(x.dtype).unsqueeze(0).expand(num_masked_total, C)
        return out

    def forward(self, x, mask=None):
        # x: [B, L, C]. If mask is provided, reuse it (see sample_mask's
        # docstring for why this matters under SAM). Otherwise sample fresh.
        if not self.training:
            return x, None

        B, L, C = x.shape
        if mask is None:
            mask = self.sample_mask(B, L, x.device)

        out = self.apply_mask(x, mask)
        return out, mask
