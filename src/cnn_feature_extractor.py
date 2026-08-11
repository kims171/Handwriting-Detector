"""
CNN feature extractor for HTR-VT: a modified ResNet-18 that collapses height
to 1 and reduces width by 4x, matching the paper's stated 1x128 feature map
for a 64x512 input line image (see Section 4.3's attention-map discussion).

The exact stride surgery isn't spelled out in the paper text -- this is a
principled reconstruction targeting the stated output shape, not a direct
transcription of the official repo (which we don't have visibility into for
this file). Worth cross-checking against the repo if you get access to it.
"""

import torch
import torch.nn as nn
import torchvision.models as models


class ResNet18FeatureExtractor(nn.Module):
    def __init__(self, out_channels=768):
        super().__init__()
        resnet = models.resnet18(weights=None)  # no pretraining -- matches the paper's data-efficient premise

        # Grayscale input instead of RGB
        resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # Stem: conv1 (stride 2) + maxpool (stride 2) = 4x downsample, isotropic.
        # 64x512 -> 16x128. Width target (128) is already reached here.
        self.stem = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
        )

        self.layer1 = resnet.layer1  # 64 channels
        self.layer2 = resnet.layer2  # 128 channels
        self.layer3 = resnet.layer3  # 256 channels
        # layer4 intentionally dropped -- removes one more isotropic
        # downsample stage and keeps channel dim at 256 instead of 512.

        # From here on, every stage is height-only stride so width stays
        # frozen at 128 while height keeps shrinking: 16 -> 8 -> 4 -> 2.
        self._make_first_block_height_only_stride(self.layer1, in_ch=64, out_ch=64)
        self._make_first_block_height_only_stride(self.layer2, in_ch=64, out_ch=128)
        self._make_first_block_height_only_stride(self.layer3, in_ch=128, out_ch=256)

        # Safety net: force height to exactly 1 regardless of small
        # arithmetic mismatches (e.g. if input height isn't a perfect
        # power-of-2 multiple). Leaves width untouched.
        self.final_pool = nn.AdaptiveAvgPool2d((1, None))

        # 256 -> 768 to match the transformer encoder's embedding dim
        self.proj = nn.Conv2d(256, out_channels, kernel_size=1)

    def _make_first_block_height_only_stride(self, layer, in_ch, out_ch):
        """Change the first block of a ResNet layer to stride (2,1) instead
        of its default (height AND width, or no stride at all), and ensure
        the downsample (skip connection) branch matches, adding one if it
        doesn't already exist (layer1's default block has none)."""
        block0 = layer[0]
        block0.conv1.stride = (2, 1)

        if block0.downsample is not None:
            # layer2/layer3 already have a downsample (channel count changes);
            # just fix its stride to match the new height-only pattern.
            block0.downsample[0].stride = (2, 1)
        else:
            # layer1: channels don't change (64->64) so torchvision didn't
            # give it a downsample branch by default. We just introduced a
            # spatial stride, so the identity branch no longer matches the
            # conv branch's shape -- need to add one.
            block0.downsample = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=(2, 1), bias=False),
                nn.BatchNorm2d(out_ch),
            )

    def forward(self, x):
        # x: [B, 1, H, W], e.g. [B, 1, 64, 512]
        x = self.stem(x)      # [B, 64, 16, 128]
        x = self.layer1(x)    # [B, 64, 8, 128]
        x = self.layer2(x)    # [B, 128, 4, 128]
        x = self.layer3(x)    # [B, 256, 2, 128]
        x = self.final_pool(x)  # [B, 256, 1, 128]
        x = self.proj(x)        # [B, 768, 1, 128]
        x = x.squeeze(2)         # [B, 768, 128]
        x = x.permute(0, 2, 1)   # [B, 128, 768]  <- token sequence, ready for the transformer
        return x
