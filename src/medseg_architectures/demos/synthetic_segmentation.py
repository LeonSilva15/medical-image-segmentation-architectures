"""Deterministic synthetic segmentation samples for educational demos.

The helpers in this module generate simple grayscale images with geometric
foreground regions and matching binary masks. They are intentionally abstract:
no patient data, clinical metadata, medical files, DICOM headers, or external
datasets are read or downloaded.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float32]
ImageSize = int | tuple[int, int]


@dataclass(frozen=True)
class SyntheticSegmentationSample:
    """A generated grayscale image and its matching foreground mask.

    Attributes:
        image: Grayscale image values in ``[0, 1]`` with shape ``(H, W)``.
        mask: Binary foreground mask values in ``{0, 1}`` with shape ``(H, W)``.
    """

    image: FloatArray
    mask: FloatArray


def generate_synthetic_segmentation_sample(
    seed: int,
    image_size: ImageSize = 64,
    *,
    min_shapes: int = 1,
    max_shapes: int = 3,
    noise_scale: float = 0.04,
) -> SyntheticSegmentationSample:
    """Generate one deterministic synthetic image/mask pair.

    Args:
        seed: Seed for NumPy's random generator.
        image_size: Either a square size or ``(height, width)``.
        min_shapes: Minimum number of foreground shapes.
        max_shapes: Maximum number of foreground shapes.
        noise_scale: Standard deviation of mild visual noise.

    Returns:
        A ``SyntheticSegmentationSample`` with ``float32`` arrays.
    """

    height, width = _normalize_image_size(image_size)
    _validate_shape_counts(min_shapes, max_shapes)
    if noise_scale < 0.0:
        raise ValueError("noise_scale must be non-negative")

    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)

    gradient = (0.65 * xx / max(width - 1, 1)) + (0.35 * yy / max(height - 1, 1))
    image = np.full((height, width), 0.08, dtype=np.float32)
    image += (0.16 * gradient).astype(np.float32)
    mask = np.zeros((height, width), dtype=np.float32)

    shape_count = int(rng.integers(min_shapes, max_shapes + 1))
    for _ in range(shape_count):
        shape_mask = _random_ellipse_mask(rng, yy, xx, height, width)
        mask = np.maximum(mask, shape_mask)
        contrast = np.float32(rng.uniform(0.5, 0.75))
        image += contrast * shape_mask

    if noise_scale:
        noise = rng.normal(0.0, noise_scale, size=(height, width)).astype(np.float32)
        image += noise

    image = np.clip(image, 0.0, 1.0).astype(np.float32)
    mask = (mask > 0.0).astype(np.float32)
    return SyntheticSegmentationSample(image=image, mask=mask)


def generate_synthetic_segmentation_batch(
    seeds: Iterable[int],
    image_size: ImageSize = 64,
) -> tuple[FloatArray, FloatArray]:
    """Generate model-ready image and mask batches.

    Args:
        seeds: One seed per requested sample.
        image_size: Either a square size or ``(height, width)``.

    Returns:
        ``(images, masks)`` shaped ``(N, 1, H, W)`` with ``float32`` values.
    """

    samples = [
        generate_synthetic_segmentation_sample(seed=seed, image_size=image_size)
        for seed in seeds
    ]
    if not samples:
        raise ValueError("seeds must contain at least one value")

    images = np.stack([sample.image for sample in samples], axis=0)[:, np.newaxis, :, :]
    masks = np.stack([sample.mask for sample in samples], axis=0)[:, np.newaxis, :, :]
    return images.astype(np.float32), masks.astype(np.float32)


def _normalize_image_size(image_size: ImageSize) -> tuple[int, int]:
    if isinstance(image_size, int):
        height = width = image_size
    else:
        height, width = image_size

    if height < 16 or width < 16:
        raise ValueError("image_size height and width must each be at least 16")
    return height, width


def _validate_shape_counts(min_shapes: int, max_shapes: int) -> None:
    if min_shapes < 1:
        raise ValueError("min_shapes must be at least 1")
    if max_shapes < min_shapes:
        raise ValueError("max_shapes must be greater than or equal to min_shapes")


def _random_ellipse_mask(
    rng: np.random.Generator,
    yy: FloatArray,
    xx: FloatArray,
    height: int,
    width: int,
) -> FloatArray:
    center_y = np.float32(rng.uniform(height * 0.25, height * 0.75))
    center_x = np.float32(rng.uniform(width * 0.25, width * 0.75))
    radius_y = np.float32(rng.uniform(height * 0.08, height * 0.2))
    radius_x = np.float32(rng.uniform(width * 0.08, width * 0.2))
    angle = np.float32(rng.uniform(-np.pi / 4.0, np.pi / 4.0))

    sin_angle = np.float32(np.sin(angle))
    cos_angle = np.float32(np.cos(angle))
    centered_y = yy - center_y
    centered_x = xx - center_x
    rotated_y = (sin_angle * centered_x) + (cos_angle * centered_y)
    rotated_x = (cos_angle * centered_x) - (sin_angle * centered_y)

    ellipse = ((rotated_y / radius_y) ** 2) + ((rotated_x / radius_x) ** 2)
    return (ellipse <= 1.0).astype(np.float32)
