import pytest
import torch

from medseg_architectures import UNet2D


@pytest.mark.parametrize(
    ("height", "width"),
    [
        (64, 64),
        (65, 73),
    ],
)
def test_unet_2d_output_preserves_spatial_shape(height: int, width: int) -> None:
    model = UNet2D(in_channels=1, out_channels=3, features=(8, 16, 32))
    model.eval()
    x = torch.randn(2, 1, height, width)

    with torch.no_grad():
        y = model(x)

    assert y.shape == (2, 3, height, width)


def test_unet_2d_supports_binary_output_channel() -> None:
    model = UNet2D(in_channels=1, out_channels=1, features=(8, 16))
    model.eval()
    x = torch.randn(1, 1, 32, 48)

    with torch.no_grad():
        y = model(x)

    assert y.shape == (1, 1, 32, 48)


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
