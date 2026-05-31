"""Minimal 2D U-Net implementation with shape-preserving segmentation logits.

The model expects image tensors in PyTorch's common image layout:
``(batch, channels, height, width)``. It produces raw segmentation logits with
shape ``(batch, out_channels, height, width)`` so the caller can choose the
appropriate loss or activation outside the model.
"""

from collections.abc import Sequence

import torch
from torch import nn
from torch.nn import functional

_VALID_NORMS = ("none", "batch", "instance", "group")
_VALID_ACTIVATIONS = ("relu", "leaky_relu", "gelu")
_VALID_UP_MODES = ("transpose", "interpolate")


def _validate_choice(name: str, value: str, options: tuple[str, ...]) -> str:
    if value not in options:
        option_text = ", ".join(options)
        raise ValueError(f"{name} must be one of [{option_text}]; got {value!r}")
    return value


def _validate_dropout(dropout: float) -> None:
    if not 0.0 <= dropout < 1.0:
        raise ValueError("dropout must be in [0.0, 1.0)")


def _safe_group_count(channels: int) -> int:
    """Return the largest safe GroupNorm group count up to eight groups."""

    for group_count in range(min(8, channels), 0, -1):
        if channels % group_count == 0:
            return group_count
    return 1


def _build_norm(norm: str, channels: int) -> nn.Module | None:
    norm = _validate_choice("norm", norm, _VALID_NORMS)
    if norm == "none":
        return None
    if norm == "batch":
        return nn.BatchNorm2d(channels)
    if norm == "instance":
        return nn.InstanceNorm2d(channels, affine=True)
    return nn.GroupNorm(num_groups=_safe_group_count(channels), num_channels=channels)


def _build_activation(activation: str) -> nn.Module:
    activation = _validate_choice("activation", activation, _VALID_ACTIVATIONS)
    if activation == "relu":
        return nn.ReLU(inplace=True)
    if activation == "leaky_relu":
        return nn.LeakyReLU(negative_slope=0.01, inplace=True)
    return nn.GELU()


def _add_conv_step(
    layers: list[nn.Module],
    in_channels: int,
    out_channels: int,
    norm: str,
    activation: str,
    dropout: float,
) -> None:
    layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
    norm_layer = _build_norm(norm, out_channels)
    if norm_layer is not None:
        layers.append(norm_layer)
    layers.append(_build_activation(activation))
    if dropout > 0.0:
        layers.append(nn.Dropout2d(p=dropout))


class DoubleConv(nn.Module):
    """U-Net convolution block used at each encoder and decoder resolution.

    Each ``3x3`` convolution uses ``padding=1``. With stride ``1``, this keeps
    the feature map height and width unchanged inside the block, so spatial
    size changes happen only at explicit pooling or upsampling steps.

    Args:
        in_channels: Number of channels entering the block.
        out_channels: Number of channels produced by both convolutions.
        norm: Optional normalization after each convolution. Supported values
            are ``"none"``, ``"batch"``, ``"instance"``, and ``"group"``.
        activation: Activation after each convolution. Supported values are
            ``"relu"``, ``"leaky_relu"``, and ``"gelu"``.
        dropout: Optional ``Dropout2d`` probability applied after activations.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        norm: str = "none",
        activation: str = "relu",
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        _validate_dropout(dropout)

        layers: list[nn.Module] = []
        _add_conv_step(
            layers,
            in_channels,
            out_channels,
            norm,
            activation,
            dropout,
        )
        _add_conv_step(
            layers,
            out_channels,
            out_channels,
            norm,
            activation,
            dropout,
        )
        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply two configurable convolution steps without changing spatial size."""

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
        norm: Optional normalization in ``DoubleConv`` blocks. Defaults to
            ``"none"`` to match the original compact baseline.
        activation: Activation in ``DoubleConv`` blocks. Defaults to
            ``"relu"``.
        dropout: Optional ``Dropout2d`` probability in ``DoubleConv`` blocks.
            Defaults to ``0.0``.
        up_mode: Decoder upsampling mode. ``"transpose"`` uses learned
            transposed convolutions. ``"interpolate"`` uses bilinear
            interpolation followed by a ``1x1`` channel projection.

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
        *,
        norm: str = "none",
        activation: str = "relu",
        dropout: float = 0.0,
        up_mode: str = "transpose",
    ) -> None:
        super().__init__()
        self._validate_args(
            in_channels,
            out_channels,
            features,
            norm,
            activation,
            dropout,
            up_mode,
        )

        feature_list = tuple(features)
        self.norm = norm
        self.activation = activation
        self.dropout = dropout
        self.up_mode = up_mode

        # ModuleList stores repeated blocks while still registering parameters
        # correctly with PyTorch. The lists stay aligned by construction:
        # encoder depth equals decoder depth.
        self.down_blocks = nn.ModuleList()
        self.up_layers = nn.ModuleList()
        self.up_blocks = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Encoder: each block preserves the current spatial size, then pooling
        # halves height and width for the next resolution level.
        current_channels = in_channels
        for feature_count in feature_list:
            self.down_blocks.append(
                DoubleConv(
                    current_channels,
                    feature_count,
                    norm=norm,
                    activation=activation,
                    dropout=dropout,
                )
            )
            current_channels = feature_count

        # Bottleneck: the deepest representation has the most context and twice
        # the channels of the last encoder stage in this compact baseline.
        self.bottleneck = DoubleConv(
            feature_list[-1],
            feature_list[-1] * 2,
            norm=norm,
            activation=activation,
            dropout=dropout,
        )

        # Decoder: each up layer restores the skip feature channel count before
        # concatenation. The default transposed convolution path preserves the
        # original implementation's learned 2x upsampling behavior.
        current_channels = feature_list[-1] * 2
        for feature_count in reversed(feature_list):
            if up_mode == "transpose":
                self.up_layers.append(
                    nn.ConvTranspose2d(
                        current_channels,
                        feature_count,
                        kernel_size=2,
                        stride=2,
                    )
                )
            else:
                self.up_layers.append(
                    nn.Conv2d(current_channels, feature_count, kernel_size=1)
                )
            self.up_blocks.append(
                DoubleConv(
                    feature_count * 2,
                    feature_count,
                    norm=norm,
                    activation=activation,
                    dropout=dropout,
                )
            )
            current_channels = feature_count

        # A 1x1 convolution mixes channels independently at each pixel and
        # produces one raw score map per requested output channel.
        self.output_conv = nn.Conv2d(feature_list[0], out_channels, kernel_size=1)

    @staticmethod
    def _validate_args(
        in_channels: int,
        out_channels: int,
        features: Sequence[int],
        norm: str,
        activation: str,
        dropout: float,
        up_mode: str,
    ) -> None:
        if in_channels < 1:
            raise ValueError("in_channels must be positive")
        if out_channels < 1:
            raise ValueError("out_channels must be positive")
        if not features:
            raise ValueError("features must contain at least one value")
        if any(feature < 1 for feature in features):
            raise ValueError("all feature counts must be positive")
        _validate_choice("norm", norm, _VALID_NORMS)
        _validate_choice("activation", activation, _VALID_ACTIVATIONS)
        _validate_dropout(dropout)
        _validate_choice("up_mode", up_mode, _VALID_UP_MODES)

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
        for skip, up_layer, up_block in zip(
            reversed(skips),
            self.up_layers,
            self.up_blocks,
            strict=True,
        ):
            if self.up_mode == "transpose":
                x = up_layer(x)
                if x.shape[-2:] != skip.shape[-2:]:
                    # Odd input sizes can produce off-by-one spatial differences
                    # after repeated pooling and transposed convolution.
                    # Resizing to the skip shape keeps concatenation valid and
                    # preserves the original input size at the output.
                    x = functional.interpolate(
                        x,
                        size=skip.shape[-2:],
                        mode="bilinear",
                        align_corners=False,
                    )
            else:
                x = functional.interpolate(
                    x,
                    size=skip.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
                x = up_layer(x)
            x = torch.cat((skip, x), dim=1)
            x = up_block(x)

        return self.output_conv(x)
