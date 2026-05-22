"""Minimal 2D U-Net implementation."""

from collections.abc import Sequence

import torch
from torch import nn
from torch.nn import functional as F


class DoubleConv(nn.Module):
    """Two convolution layers with ReLU activations."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class UNet2D(nn.Module):
    """Small 2D U-Net for segmentation logits.

    Args:
        in_channels: Number of input image channels.
        out_channels: Number of output segmentation channels.
        features: Encoder feature widths from shallow to deep.

    The forward pass returns raw logits with the same spatial height and width as
    the input tensor.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        features: Sequence[int] = (16, 32, 64),
    ) -> None:
        super().__init__()
        self._validate_args(in_channels, out_channels, features)

        feature_list = tuple(features)
        self.down_blocks = nn.ModuleList()
        self.up_transposes = nn.ModuleList()
        self.up_blocks = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        current_channels = in_channels
        for feature_count in feature_list:
            self.down_blocks.append(DoubleConv(current_channels, feature_count))
            current_channels = feature_count

        self.bottleneck = DoubleConv(feature_list[-1], feature_list[-1] * 2)

        current_channels = feature_list[-1] * 2
        for feature_count in reversed(feature_list):
            self.up_transposes.append(
                nn.ConvTranspose2d(
                    current_channels,
                    feature_count,
                    kernel_size=2,
                    stride=2,
                )
            )
            self.up_blocks.append(DoubleConv(feature_count * 2, feature_count))
            current_channels = feature_count

        self.output_conv = nn.Conv2d(feature_list[0], out_channels, kernel_size=1)

    @staticmethod
    def _validate_args(
        in_channels: int,
        out_channels: int,
        features: Sequence[int],
    ) -> None:
        if in_channels < 1:
            raise ValueError("in_channels must be positive")
        if out_channels < 1:
            raise ValueError("out_channels must be positive")
        if not features:
            raise ValueError("features must contain at least one value")
        if any(feature < 1 for feature in features):
            raise ValueError("all feature counts must be positive")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips: list[torch.Tensor] = []

        for down_block in self.down_blocks:
            x = down_block(x)
            skips.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)

        for skip, up_transpose, up_block in zip(
            reversed(skips),
            self.up_transposes,
            self.up_blocks,
            strict=True,
        ):
            x = up_transpose(x)
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(
                    x,
                    size=skip.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
            x = torch.cat((skip, x), dim=1)
            x = up_block(x)

        return self.output_conv(x)
