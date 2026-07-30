"""Model definitions for the PyTorch FOMO object-detection learning block.

FOMO (Faster Objects, More Objects) is a fully-convolutional detector: instead
of regressing bounding boxes it predicts a coarse class grid, where each cell
holds the class of the object centred on it (or background). The spatial output
is the input resolution downsampled by 8, matching Edge Impulse's FOMO layout.

``FomoNet`` is a small backbone with three stride-2 stages (total downsample of
8) followed by a 1x1 classification head. It outputs raw logits per grid cell so
it can be trained with a weighted cross-entropy loss. For export we wrap it in
``FomoExport`` which applies a channel-wise softmax, giving per-cell class
probabilities in the shape Edge Impulse's FOMO output layer expects.
"""

from __future__ import annotations

import torch
from torch import nn


class FomoNet(nn.Module):
    """A compact FOMO-style backbone that downsamples the input by 8.

    Args:
        in_channels: Number of input channels (1 for grayscale, 3 for RGB).
        num_classes: Number of output classes including background (index 0).
    """

    def __init__(self, in_channels: int, num_classes: int) -> None:
        super().__init__()
        self.features = nn.Sequential(
            # /2
            nn.Conv2d(in_channels, 16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            # /4
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            # /8
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        # 1x1 classification head keeps the model fully convolutional.
        self.head = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, num_classes, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Returns logits of shape (batch, num_classes, grid_h, grid_w).
        return self.head(self.features(x))


class FomoExport(nn.Module):
    """Wraps a trained FomoNet with a channel-wise softmax for export.

    Edge Impulse's FOMO output layer expects per-cell class probabilities. We
    keep training in logit space (weighted cross-entropy) and only add the
    softmax at export time.
    """

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.model(x), dim=1)


def build_model(in_channels: int, num_classes: int) -> FomoNet:
    """Factory used by the training script."""
    return FomoNet(in_channels=in_channels, num_classes=num_classes)
