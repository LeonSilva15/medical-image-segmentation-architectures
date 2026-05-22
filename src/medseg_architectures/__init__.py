"""Medical image segmentation architecture implementations."""

from medseg_architectures.models import UNet2D
from medseg_architectures.registry import available_models, create_model

__all__ = ["UNet2D", "available_models", "create_model"]
