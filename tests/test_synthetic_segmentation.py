import numpy as np
import torch

from medseg_architectures import UNet2D
from medseg_architectures.demos import (
    generate_synthetic_segmentation_batch,
    generate_synthetic_segmentation_sample,
)


def test_synthetic_segmentation_sample_is_deterministic() -> None:
    first = generate_synthetic_segmentation_sample(seed=42, image_size=(48, 64))
    second = generate_synthetic_segmentation_sample(seed=42, image_size=(48, 64))

    np.testing.assert_array_equal(first.image, second.image)
    np.testing.assert_array_equal(first.mask, second.mask)


def test_synthetic_segmentation_sample_shapes_dtypes_and_ranges() -> None:
    sample = generate_synthetic_segmentation_sample(seed=7, image_size=64)

    assert sample.image.shape == (64, 64)
    assert sample.mask.shape == (64, 64)
    assert sample.image.dtype == np.float32
    assert sample.mask.dtype == np.float32
    assert 0.0 <= float(sample.image.min()) <= float(sample.image.max()) <= 1.0
    assert set(np.unique(sample.mask)).issubset({0.0, 1.0})
    assert sample.mask.sum() > 0.0


def test_synthetic_segmentation_batch_is_model_ready() -> None:
    images, masks = generate_synthetic_segmentation_batch([1, 2, 3], image_size=32)

    assert images.shape == (3, 1, 32, 32)
    assert masks.shape == (3, 1, 32, 32)
    assert images.dtype == np.float32
    assert masks.dtype == np.float32


def test_tiny_unet_forward_pass_accepts_synthetic_batch() -> None:
    images, _ = generate_synthetic_segmentation_batch([11, 12], image_size=32)
    model = UNet2D(in_channels=1, out_channels=1, features=(2, 4))
    model.eval()

    with torch.no_grad():
        logits = model(torch.from_numpy(images))

    assert logits.shape == (2, 1, 32, 32)
