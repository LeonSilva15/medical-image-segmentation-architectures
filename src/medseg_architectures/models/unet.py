"""Minimal 2D U-Net implementation with shape-preserving segmentation logits.

The model expects image tensors in PyTorch's common image layout:
``(batch, channels, height, width)``. It produces raw segmentation logits with
shape ``(batch, out_channels, height, width)`` so the caller can choose the
appropriate loss or activation outside the model.
"""

from collections.abc import Sequence

import torch
from torch import nn
from torch.nn import functional as F


class DoubleConv(nn.Module):
    """U-Net convolution block used at each encoder and decoder resolution.

    Each ``3x3`` convolution uses ``padding=1``. With stride ``1``, this keeps
    the feature map height and width unchanged inside the block, so spatial
    size changes happen only at explicit pooling or upsampling steps.

    Args:
        in_channels: Number of channels entering the block.
        out_channels: Number of channels produced by both convolutions.
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply two convolution-ReLU pairs without changing spatial size."""

        return self.layers(x)


class UNet2D(nn.Module):
    """Small 2D U-Net for segmentation logits.

    The architecture has four conceptual parts:

    1. Encoder blocks extract increasingly abstract features.
    2. A bottleneck processes the smallest feature map.
    3. Decoder blocks upsample and fuse encoder skip features.
    4. A ``1x1`` output convolution maps features to class logits per pixel.

    ``features`` controls the channel width at each encoder stage from shallow
    to deep. For example, ``features=(16, 32, 64)`` builds three encoder stages,
    a ``128`` channel bottleneck, and three matching decoder stages.

    Args:
        in_channels: Number of input image channels.
        out_channels: Number of output segmentation channels. Use ``1`` for a
            single binary logit map or one channel per class for multiclass
            segmentation.
        features: Encoder feature widths from shallow to deep. Larger values
            increase model capacity and memory use.

    The forward pass returns raw logits, not probabilities. Use
    ``torch.sigmoid`` for binary probabilities or ``torch.softmax`` for
    multiclass probabilities during evaluation, and pair the raw logits with a
    loss such as ``BCEWithLogitsLoss`` or ``CrossEntropyLoss`` during training.
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

        # ModuleList stores repeated blocks while still registering parameters
        # correctly with PyTorch. The lists stay aligned by construction:
        # encoder depth equals decoder depth.
        self.down_blocks = nn.ModuleList()
        self.up_transposes = nn.ModuleList()
        self.up_blocks = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Encoder: each block preserves the current spatial size, then pooling
        # halves height and width for the next resolution level.
        current_channels = in_channels
        for feature_count in feature_list:
            self.down_blocks.append(DoubleConv(current_channels, feature_count))
            current_channels = feature_count

        # Bottleneck: the deepest representation has the most context and twice
        # the channels of the last encoder stage in this compact baseline.
        self.bottleneck = DoubleConv(feature_list[-1], feature_list[-1] * 2)

        # Decoder: transposed convolutions learn the 2x upsampling step. After
        # upsampling, concatenation doubles the channels because encoder skip
        # features and decoder features are joined along the channel dimension.
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

        # A 1x1 convolution mixes channels independently at each pixel and
        # produces one raw score map per requested output channel.
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
        """Return raw logits with the same height and width as ``x``.

        Args:
            x: Input tensor shaped ``(batch, in_channels, height, width)``.

        Returns:
            Tensor shaped ``(batch, out_channels, height, width)``.
        """

        skips: list[torch.Tensor] = []

        # Save each encoder output before pooling. These skip tensors keep the
        # high-resolution detail that the decoder will later reuse.
        for down_block in self.down_blocks:
            x = down_block(x)
            skips.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)

        # Walk from deepest skip to shallowest skip while the decoder restores
        # resolution. The lists have equal length because both are built from
        # the same feature widths.
        for skip, up_transpose, up_block in zip(
            reversed(skips),
            self.up_transposes,
            self.up_blocks,
            strict=True,
        ):
            x = up_transpose(x)
            if x.shape[-2:] != skip.shape[-2:]:
                # Odd input sizes can produce off-by-one spatial differences
                # after repeated pooling and transposed convolution. Resizing to
                # the skip shape keeps concatenation valid and preserves the
                # original input size at the output.
                x = F.interpolate(
                    x,
                    size=skip.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
            x = torch.cat((skip, x), dim=1)
            x = up_block(x)

        return self.output_conv(x)
