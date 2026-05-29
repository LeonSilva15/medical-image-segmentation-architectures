import pytest

from medseg_architectures import UNet2D, available_models, create_model


def test_available_models_includes_unet2d() -> None:
    assert "unet2d" in available_models()


def test_create_model_returns_unet2d() -> None:
    model = create_model("unet2d", in_channels=1, out_channels=2)

    assert isinstance(model, UNet2D)


def test_create_model_is_case_insensitive() -> None:
    model = create_model("UNET2D", in_channels=1, out_channels=2)

    assert isinstance(model, UNet2D)


def test_create_model_rejects_unknown_name() -> None:
    with pytest.raises(ValueError) as exc_info:
        create_model("unknown")

    message = str(exc_info.value)
    assert "unknown" in message
    for model_name in available_models():
        assert model_name in message
