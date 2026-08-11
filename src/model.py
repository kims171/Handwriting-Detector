"""
Full HTR-VT model, wired for use with a SAM optimizer: forward() accepts an
optional externally-provided span mask so a training step can sample it once
and reuse it across SAM's two forward/backward passes.
"""
import torch.nn as nn

from src.cnn_feature_extractor import ResNet18FeatureExtractor
from utilities.positional_and_masking import SinusoidalPositionalEmbedding, SpanMasking
from utilities.transformer_encoder import TransformerEncoder, CTCHead


class HTRVT(nn.Module):
    def __init__(self, num_classes, dim=768, num_heads=6, mlp_hidden_dim=3072,
                 num_layers=4, max_len=128, mask_ratio=0.4, span_length=8):
        super().__init__()
        self.max_len = max_len
        self.cnn = ResNet18FeatureExtractor(out_channels=dim)
        self.pos_embed = SinusoidalPositionalEmbedding(max_len=max_len, dim=dim)
        self.span_mask = SpanMasking(dim=dim, mask_ratio=mask_ratio, span_length=span_length)
        self.encoder = TransformerEncoder(dim=dim, num_heads=num_heads,
                                           mlp_hidden_dim=mlp_hidden_dim, num_layers=num_layers)
        self.head = CTCHead(dim=dim, num_classes=num_classes)

    def sample_mask(self, batch_size, device):
        """Sample one span mask to be reused across SAM's two forward passes
        in a single training step."""
        return self.span_mask.sample_mask(batch_size, self.max_len, device)

    def forward(self, images, mask=None):
        tokens = self.cnn(images)
        tokens = self.pos_embed(tokens)
        tokens, used_mask = self.span_mask(tokens, mask=mask)
        encoded = self.encoder(tokens)
        log_probs = self.head(encoded)
        return log_probs
