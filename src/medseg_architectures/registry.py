"""Small registry for implemented architecture constructors."""

from collections.abc import Callable

from torch import nn

from medseg_architectures.models import UNet2D

ModelFactory = Callable[..., nn.Module]

_MODEL_FACTORIES: dict[str, ModelFactory] = {
    "unet2d": UNet2D,
}


def available_models() -> tuple[str, ...]:
    """Return the registered model names."""

    return tuple(sorted(_MODEL_FACTORIES))


def create_model(name: str, **kwargs: object) -> nn.Module:
    """Create a model by registry name."""

    normalized_name = name.lower()
    try:
        factory = _MODEL_FACTORIES[normalized_name]
    except KeyError as exc:
        options = ", ".join(available_models())
        raise ValueError(f"Unknown model '{name}'. Available models: {options}") from exc
    return factory(**kwargs)
