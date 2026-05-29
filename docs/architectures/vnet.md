# V-Net

## Plain-Language Overview

V-Net adapts the encoder-decoder segmentation idea to 3D volumes. Instead of
processing one 2D slice at a time, the network uses 3D convolutions so depth,
height, and width are modeled together.

For the direct 2D U-Net-to-volume translation, see
[3D U-Net](3d-unet.md). V-Net is a related volumetric encoder-decoder branch
with its own design choices rather than a dependency of that page.

## What Problem It Solved

Medical scans are often volumetric. V-Net addresses this by using 3D operations
for dense segmentation of full volume patches.

## Visual Architecture Schematic

This is an original schematic for this book, not a copied paper figure.

```mermaid
graph LR
    Input["Volume patch<br/>(B, C, D, H, W)"]
    Enc["3D encoder"]
    Bottleneck["3D bottleneck"]
    Dec["3D decoder"]
    Output["Volume logits<br/>(B, K, D, H, W)"]

    Input --> Enc --> Bottleneck --> Dec --> Output
    Enc -. 3D skip .-> Dec
```

## Step-By-Step Walkthrough

1. 3D convolution blocks extract local volumetric features.
2. Downsampling increases the receptive field across depth, height, and width.
3. The decoder upsamples and reuses encoder features through skip connections.
4. A final `1x1x1` convolution returns per-voxel logits.

## Minimum Architecture Form

Core building blocks:

- 3D convolution blocks.
- A 3D downsampling path.
- A 3D upsampling path with skip fusion.
- A `1x1x1` output projection.

Tensor shape flow:

```text
Input volume:     (B, C, D, H, W)
Encoder skip:     (B, F, D, H, W)
Bottleneck:       (B, 2F, D/2, H/2, W/2)
Output logits:    (B, K, D, H, W)
```

Repo-authored pseudocode:

```text
extract a 3D skip tensor
downsample to a smaller 3D feature map
process the bottleneck
upsample to the skip tensor size
concatenate skip and decoder features
project to per-voxel logits
```

??? example "Minimum runnable PyTorch sketch"

    ```python
    import torch
    from torch import nn


    class MinimumVNet(nn.Module):
        def __init__(self, in_channels: int, out_channels: int) -> None:
            super().__init__()
            self.enc = nn.Sequential(
                nn.Conv3d(in_channels, 8, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
            )
            self.down = nn.Conv3d(8, 16, kernel_size=3, stride=2, padding=1)
            self.up = nn.ConvTranspose3d(16, 8, kernel_size=2, stride=2)
            self.fuse = nn.Sequential(
                nn.Conv3d(16, 8, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
            )
            self.out = nn.Conv3d(8, out_channels, kernel_size=1)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            skip = self.enc(x)
            x = torch.relu(self.down(skip))
            x = self.up(x)
            if x.shape[-3:] != skip.shape[-3:]:
                x = nn.functional.interpolate(
                    x,
                    size=skip.shape[-3:],
                    mode="trilinear",
                    align_corners=False,
                )
            x = torch.cat((skip, x), dim=1)
            return self.out(self.fuse(x))


    model = MinimumVNet(in_channels=1, out_channels=2)
    volume = torch.randn(1, 1, 16, 32, 32)
    logits = model(volume)
    assert logits.shape == (1, 2, 16, 32, 32)
    ```

## Implementation Walkthrough

This repository does not provide a tested local V-Net implementation yet. The
minimum code sketch above is educational only. It is not registered as a package
model, does not include a demo, and does not claim to reproduce the full paper.

## Learning Notes For Practitioners

- The minimum form is intentionally small so the 3D tensor path is visible.
- Real 3D models need careful memory planning because volume tensors grow
  quickly.
- Tests and demos for any future implementation should use small synthetic
  volumes unless a public, properly licensed dataset is configured.

## What Changed Relative To U-Net

V-Net moves the U-Net-style encoder-decoder idea from 2D image tensors to 3D
volume tensors.

## Strengths

- Models local structure across depth, height, and width.
- Fits volumetric segmentation tasks more directly than a slice-only 2D model.

## Limitations

- The local page is reference-only and does not include tested package code.
- 3D convolutional models are more memory intensive than 2D slice models.

## Implementation Status

| Field | Value |
| --- | --- |
| Status | reference-only |
| Code | Not implemented locally |
| Tests | Not implemented locally |
| Demo | Not implemented locally |
| Data used in examples | synthetic tensors only |
| Metadata ID | `vnet` |

!!! note "Educational scope"
    This repository is for education and research. This page does not claim
    clinical readiness.

## Model Details

| Field | Value |
| --- | --- |
| Year | 2016 |
| Parent | U-Net |
| Family | U-Net family, 3D |
| Paper title | Fully Convolutional Neural Networks for Volumetric Medical Image Segmentation |
| DOI | `10.1109/3DV.2016.79` |
| arXiv | `1606.04797` |

## Read The Original Paper

- DOI: [10.1109/3DV.2016.79](https://doi.org/10.1109/3DV.2016.79)
- arXiv: [1606.04797](https://arxiv.org/abs/1606.04797)
