import pytest

from medseg_architectures import UNet2D, available_models, create_model


def test_available_models_includes_unet2d() -> None:
    assert available_models() == ("unet2d",)


def test_create_model_returns_unet2d() -> None:
    model = create_model("unet2d", in_channels=1, out_channels=2, features=(8, 16))

    assert isinstance(model, UNet2D)


def test_create_model_rejects_unknown_name() -> None:
    with pytest.raises(ValueError, match="Unknown model"):
        create_model("missing-model")
