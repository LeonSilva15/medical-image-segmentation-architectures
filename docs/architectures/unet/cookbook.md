# U-Net Cookbook

This page collects practical U-Net recipes that build on the synthetic demo and
tested `UNet2D` implementation.

All examples on this page use synthetic tensors unless a public, properly
licensed dataset is explicitly configured. Do not add private medical images,
PHI, patient identifiers, DICOM headers, or clinical data.

## Default Forward Pass

The default constructor keeps the compact baseline behavior: no normalization,
ReLU activations, no dropout, and transposed-convolution upsampling.

```python
import torch

from medseg_architectures import UNet2D

model = UNet2D(in_channels=1, out_channels=2, features=(16, 32, 64))
model.eval()

x = torch.randn(1, 1, 65, 73)
with torch.no_grad():
    logits = model(x)

assert logits.shape == (1, 2, 65, 73)
```

## Normalization, Activation, And Dropout

Use these options for small ablations while keeping the same input and output
shape contract.

```python
model = UNet2D(
    in_channels=1,
    out_channels=2,
    features=(16, 32, 64),
    norm="group",
    activation="gelu",
    dropout=0.1,
)
```

Supported normalization values are `"none"`, `"batch"`, `"instance"`, and
`"group"`. Supported activation values are `"relu"`, `"leaky_relu"`, and
`"gelu"`. Dropout must be greater than or equal to `0.0` and less than `1.0`.

## Interpolation Upsampling

Use `up_mode="interpolate"` to replace transposed-convolution upsampling with
bilinear interpolation followed by a `1x1` channel projection.

```python
model = UNet2D(
    in_channels=1,
    out_channels=2,
    features=(16, 32, 64),
    up_mode="interpolate",
)

x = torch.randn(2, 1, 64, 64)
with torch.no_grad():
    logits = model(x)

assert logits.shape == (2, 2, 64, 64)
```
