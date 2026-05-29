import pytest
import torch
from torch import nn

from medseg_architectures import UNet2D


@pytest.mark.parametrize(
    ("height", "width"),
    [
        (64, 64),
        (65, 73),
    ],
)
@pytest.mark.parametrize("up_mode", ["transpose", "interpolate"])
def test_unet_2d_output_preserves_spatial_shape(
    height: int,
    width: int,
    up_mode: str,
) -> None:
    model = UNet2D(in_channels=1, out_channels=3, features=(8, 16, 32), up_mode=up_mode)
    model.eval()
    x = torch.randn(2, 1, height, width)

    with torch.no_grad():
        y = model(x)

    assert y.shape == (2, 3, height, width)


@pytest.mark.parametrize("up_mode", ["transpose", "interpolate"])
def test_unet_2d_uses_requested_upsampling_mode(up_mode: str) -> None:
    model = UNet2D(in_channels=1, out_channels=2, features=(4, 8), up_mode=up_mode)

    expected_layer_type = nn.ConvTranspose2d if up_mode == "transpose" else nn.Conv2d

    assert all(isinstance(layer, expected_layer_type) for layer in model.up_layers)


def test_unet_2d_supports_binary_output_channel() -> None:
    model = UNet2D(in_channels=1, out_channels=1, features=(8, 16))
    model.eval()
    x = torch.randn(1, 1, 32, 48)

    with torch.no_grad():
        y = model(x)

    assert y.shape == (1, 1, 32, 48)


@pytest.mark.parametrize("norm", ["none", "batch", "instance", "group"])
def test_unet_2d_supports_normalization_options(norm: str) -> None:
    model = UNet2D(in_channels=1, out_channels=2, features=(8, 16), norm=norm)
    model.eval()
    x = torch.randn(1, 1, 32, 40)

    with torch.no_grad():
        y = model(x)

    assert y.shape == (1, 2, 32, 40)

    normalization_types = (nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm)
    if norm == "none":
        assert not any(isinstance(module, normalization_types) for module in model.modules())
    else:
        expected_type = {
            "batch": nn.BatchNorm2d,
            "instance": nn.InstanceNorm2d,
            "group": nn.GroupNorm,
        }[norm]
        assert any(isinstance(module, expected_type) for module in model.modules())


def test_unet_2d_group_norm_falls_back_to_one_group_for_prime_channels() -> None:
    model = UNet2D(in_channels=1, out_channels=2, features=(11,), norm="group")

    group_norms = [
        module for module in model.modules() if isinstance(module, nn.GroupNorm)
    ]

    assert group_norms
    assert any(
        group_norm.num_channels == 11 and group_norm.num_groups == 1
        for group_norm in group_norms
    )


@pytest.mark.parametrize(
    ("activation", "expected_type"),
    [
        ("relu", nn.ReLU),
        ("leaky_relu", nn.LeakyReLU),
        ("gelu", nn.GELU),
    ],
)
def test_unet_2d_supports_activation_options(
    activation: str,
    expected_type: type[nn.Module],
) -> None:
    model = UNet2D(in_channels=1, out_channels=2, features=(4, 8), activation=activation)
    model.eval()
    x = torch.randn(1, 1, 32, 32)

    with torch.no_grad():
        y = model(x)

    assert y.shape == (1, 2, 32, 32)
    assert any(isinstance(module, expected_type) for module in model.modules())


def test_unet_2d_adds_dropout_when_enabled() -> None:
    model = UNet2D(in_channels=1, out_channels=2, features=(4, 8), dropout=0.25)
    model.eval()
    x = torch.randn(1, 1, 32, 32)

    with torch.no_grad():
        y = model(x)

    assert y.shape == (1, 2, 32, 32)
    assert any(
        isinstance(module, nn.Dropout2d) and module.p == 0.25
        for module in model.modules()
    )


def test_unet_2d_rejects_non_positive_in_channels() -> None:
    with pytest.raises(ValueError, match="in_channels"):
        UNet2D(in_channels=0, out_channels=1)


def test_unet_2d_rejects_non_positive_out_channels() -> None:
    with pytest.raises(ValueError, match="out_channels"):
        UNet2D(in_channels=1, out_channels=0)


def test_unet_2d_rejects_empty_features() -> None:
    with pytest.raises(ValueError, match="features"):
        UNet2D(in_channels=1, out_channels=1, features=())


@pytest.mark.parametrize("features", [(0,), (8, -16)])
def test_unet_2d_rejects_non_positive_feature_values(features: tuple[int, ...]) -> None:
    with pytest.raises(ValueError, match="positive"):
        UNet2D(in_channels=1, out_channels=1, features=features)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"norm": "layer"}, "norm"),
        ({"activation": "swish"}, "activation"),
        ({"up_mode": "nearest"}, "up_mode"),
        ({"dropout": -0.1}, "dropout"),
        ({"dropout": 1.0}, "dropout"),
    ],
)
def test_unet_2d_rejects_invalid_config_options(
    kwargs: dict[str, object],
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        UNet2D(in_channels=1, out_channels=1, features=(8, 16), **kwargs)
