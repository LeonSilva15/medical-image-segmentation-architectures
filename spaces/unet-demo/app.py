"""Gradio app for a CPU-only synthetic U-Net segmentation demo."""

from __future__ import annotations

import os
import sys
import time
from functools import lru_cache
from pathlib import Path
from typing import Any

import gradio as gr
import numpy as np
import torch
from PIL import Image

os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")

REPO_SRC = Path(__file__).resolve().parents[2] / "src"
if REPO_SRC.exists() and str(REPO_SRC) not in sys.path:
    sys.path.insert(0, str(REPO_SRC))

from medseg_architectures import UNet2D, count_parameters  # noqa: E402
from medseg_architectures.demos import (  # noqa: E402
    generate_synthetic_segmentation_batch,
    generate_synthetic_segmentation_sample,
)

SAFETY_WARNING = (
    "Educational demo only. Not for clinical diagnosis. Do not upload private medical "
    "images, PHI, DICOM headers, or clinical data."
)
IMAGE_SIZE = 64
MODEL_FEATURES = (2, 4)
TRAIN_SEEDS = tuple(range(1000, 1096))
EXAMPLE_SEEDS = {
    "Synthetic shapes A": 7,
    "Synthetic shapes B": 19,
    "Synthetic shapes C": 31,
    "Synthetic shapes D": 43,
}


def train_demo_model() -> tuple[UNet2D, str]:
    """Train a tiny U-Net on deterministic synthetic shapes."""

    torch.manual_seed(13)
    torch.set_num_threads(max(1, min(torch.get_num_threads(), 2)))

    model = UNet2D(
        in_channels=1,
        out_channels=1,
        features=MODEL_FEATURES,
        norm="none",
        up_mode="transpose",
    )
    model.train()

    images, masks = generate_synthetic_segmentation_batch(TRAIN_SEEDS, image_size=IMAGE_SIZE)
    image_tensor = torch.from_numpy(images)
    mask_tensor = torch.from_numpy(masks)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.02)
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([6.0]))
    generator = torch.Generator().manual_seed(29)
    batch_size = 8
    steps = 0
    start = time.perf_counter()

    for _ in range(8):
        order = torch.randperm(image_tensor.shape[0], generator=generator)
        for start_index in range(0, image_tensor.shape[0], batch_size):
            batch_indices = order[start_index : start_index + batch_size]
            optimizer.zero_grad(set_to_none=True)
            logits = model(image_tensor[batch_indices])
            loss = loss_fn(logits, mask_tensor[batch_indices])
            loss.backward()
            optimizer.step()
            steps += 1

    model.eval()
    elapsed = time.perf_counter() - start
    status = (
        "No checkpoint is loaded. A tiny U-Net was deterministically trained in memory "
        f"on synthetic geometric shapes ({steps} CPU optimization steps, {elapsed:.2f}s)."
    )
    return model, status


@lru_cache(maxsize=1)
def get_demo_model() -> tuple[UNet2D, str]:
    """Return the initialized model or raise a clear setup error."""

    try:
        return train_demo_model()
    except Exception as exc:
        raise RuntimeError(
            "Could not initialize the synthetic U-Net demo model. "
            "The app does not use external checkpoints; retry or inspect the Space logs."
        ) from exc


def run_segmentation(
    example_name: str,
    uploaded_image: Any | None,
    threshold: float,
) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None, str]:
    """Run the demo model on a synthetic example or safe educational upload."""

    try:
        model, model_status = get_demo_model()
        image_array, target_mask, source_text = prepare_input(example_name, uploaded_image)
        input_tensor = torch.from_numpy(image_array[np.newaxis, np.newaxis, :, :])

        with torch.no_grad():
            logits = model(input_tensor)
            probability = torch.sigmoid(logits)[0, 0].cpu().numpy()

        predicted_mask = probability >= threshold
        input_preview = grayscale_to_rgb(image_array)
        overlay = predicted_region_overlay(image_array, predicted_mask)
        target_preview = (
            target_mask_overlay(image_array, target_mask)
            if target_mask is not None
            else None
        )
        details = model_details(
            model=model,
            model_status=model_status,
            source_text=source_text,
            threshold=threshold,
            input_shape=tuple(input_tensor.shape),
            logits_shape=tuple(logits.shape),
        )
        return input_preview, overlay, target_preview, details
    except Exception as exc:
        message = (
            "### Demo error\n\n"
            f"{exc}\n\n"
            "The app is intentionally limited to synthetic or safe educational images. "
            "It does not process DICOM files, PHI, or clinical data."
        )
        return None, None, None, message


def prepare_input(
    example_name: str,
    uploaded_image: Any | None,
) -> tuple[np.ndarray, np.ndarray | None, str]:
    """Prepare the selected synthetic example or uploaded educational image."""

    if uploaded_image is not None:
        image = uploaded_image_to_grayscale(uploaded_image)
        return image, None, "Uploaded safe educational image resized to 64x64 grayscale."

    seed = EXAMPLE_SEEDS.get(example_name, next(iter(EXAMPLE_SEEDS.values())))
    sample = generate_synthetic_segmentation_sample(seed=seed, image_size=IMAGE_SIZE)
    return sample.image, sample.mask, f"Built-in deterministic synthetic example, seed {seed}."


def uploaded_image_to_grayscale(uploaded_image: Any) -> np.ndarray:
    """Convert a Gradio image value to normalized 64x64 grayscale."""

    array = np.asarray(uploaded_image)
    if array.ndim == 2:
        pil_image = Image.fromarray(_to_uint8(array), mode="L")
    elif array.ndim == 3:
        pil_image = Image.fromarray(_to_uint8(array)).convert("L")
    else:
        raise ValueError("Uploaded image must be a 2D grayscale or 3D RGB-like image.")

    pil_image = pil_image.resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BILINEAR)
    return (np.asarray(pil_image, dtype=np.float32) / 255.0).astype(np.float32)


def grayscale_to_rgb(image: np.ndarray) -> np.ndarray:
    """Render a normalized grayscale image as RGB uint8."""

    channel = _to_uint8(image)
    return np.stack([channel, channel, channel], axis=-1)


def predicted_region_overlay(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Overlay the predicted region in cyan."""

    base = grayscale_to_rgb(image).astype(np.float32)
    color = np.zeros_like(base)
    color[..., 1] = 215.0
    color[..., 2] = 255.0
    blended = np.where(mask[..., np.newaxis], (0.55 * base) + (0.45 * color), base)
    return np.clip(blended, 0, 255).astype(np.uint8)


def target_mask_overlay(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Overlay the synthetic target region in amber."""

    base = grayscale_to_rgb(image).astype(np.float32)
    color = np.zeros_like(base)
    color[..., 0] = 255.0
    color[..., 1] = 176.0
    blended = np.where(mask[..., np.newaxis] > 0.0, (0.55 * base) + (0.45 * color), base)
    return np.clip(blended, 0, 255).astype(np.uint8)


def model_details(
    *,
    model: UNet2D,
    model_status: str,
    source_text: str,
    threshold: float,
    input_shape: tuple[int, ...],
    logits_shape: tuple[int, ...],
) -> str:
    """Format model and inference details for the UI."""

    parameter_count = count_parameters(model)
    return (
        "### Model details\n\n"
        f"- Source: {source_text}\n"
        f"- Model status: {model_status}\n"
        f"- Input tensor shape: `{input_shape}`\n"
        f"- Output tensor/logits shape: `{logits_shape}`\n"
        f"- Parameter count: `{parameter_count:,}`\n"
        "- Model configuration: "
        "`UNet2D(in_channels=1, out_channels=1, features=(2, 4), "
        "norm=\"none\", up_mode=\"transpose\")`\n"
        f"- Threshold for predicted region overlay: `{threshold:.2f}`\n"
    )


def _to_uint8(array: np.ndarray) -> np.ndarray:
    array = np.asarray(array)
    if np.issubdtype(array.dtype, np.integer):
        clipped = np.clip(array, 0, 255)
        return clipped.astype(np.uint8)

    finite_array = np.nan_to_num(array.astype(np.float32), nan=0.0, posinf=1.0, neginf=0.0)
    if float(finite_array.max(initial=0.0)) > 1.0:
        finite_array = finite_array / 255.0
    return np.clip(finite_array * 255.0, 0, 255).astype(np.uint8)


def build_app() -> gr.Blocks:
    """Build the Gradio Blocks interface."""

    with gr.Blocks(title="U-Net Segmentation Demo") as demo:
        gr.Markdown(
            "# U-Net Segmentation Demo\n\n"
            f"**{SAFETY_WARNING}**\n\n"
            "This app trains a tiny U-Net in memory on deterministic geometric "
            "shapes and shows a segmentation-style predicted region overlay."
        )

        with gr.Row():
            with gr.Column(scale=1):
                example = gr.Dropdown(
                    choices=list(EXAMPLE_SEEDS),
                    value=next(iter(EXAMPLE_SEEDS)),
                    label="Built-in synthetic example",
                )
                upload = gr.Image(
                    label=(
                        "Optional safe educational image upload "
                        "(PNG/JPEG-style only; no DICOM, PHI, or clinical data)"
                    ),
                    type="numpy",
                    image_mode="RGB",
                    sources=["upload"],
                )
                threshold = gr.Slider(
                    minimum=0.05,
                    maximum=0.95,
                    value=0.50,
                    step=0.05,
                    label="Predicted region threshold",
                )
                run_button = gr.Button("Run synthetic U-Net demo", variant="primary")

            with gr.Column(scale=2):
                input_view = gr.Image(label="Input image", type="numpy")
                prediction_view = gr.Image(label="Predicted region overlay", type="numpy")
                target_view = gr.Image(
                    label="Synthetic target overlay (built-in examples only)",
                    type="numpy",
                )
                details = gr.Markdown()

        inputs = [example, upload, threshold]
        outputs = [input_view, prediction_view, target_view, details]
        run_button.click(run_segmentation, inputs=inputs, outputs=outputs)
        demo.load(run_segmentation, inputs=inputs, outputs=outputs)

    return demo


if __name__ == "__main__":
    build_app().launch()
