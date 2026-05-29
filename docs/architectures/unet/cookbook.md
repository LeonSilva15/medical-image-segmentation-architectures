# U-Net Cookbook

This page collects practical U-Net recipes that build on the synthetic demo and
tested `UNet2D` implementation. For the architecture walkthrough, start with the
[U-Net overview](../unet.md). For the complete implementation, read the
[full code](code.md).

All examples on this page use synthetic tensors unless a public, properly
licensed dataset is explicitly configured. Do not add private medical images,
PHI, patient identifiers, DICOM headers, or clinical data.

The examples use `torch.randn` for synthetic image-like tensors, `torch.randint`
for synthetic masks, `UNet2D` for the model, `model.eval()` for inference-style
examples, and `torch.no_grad()` when gradients are not needed.

## Binary Segmentation With One Output Channel

Use `out_channels=1` when the model should produce one foreground-vs-background
logit map per image. The output shape is `(B, 1, H, W)`.

```python
import torch

from medseg_architectures import UNet2D

model = UNet2D(in_channels=1, out_channels=1, features=(8, 16))
model.eval()

x = torch.randn(2, 1, 48, 64)
with torch.no_grad():
    logits = model(x)
    probabilities = torch.sigmoid(logits)

assert logits.shape == (2, 1, 48, 64)
assert probabilities.shape == logits.shape
```

During training, pass the raw logits to a logits-aware binary loss such as
`torch.nn.BCEWithLogitsLoss`.

## Multiclass Segmentation With N Output Channels

Use `out_channels=N` when each pixel belongs to exactly one of `N` classes. The
model returns one raw logit map per class, so the output shape is `(B, N, H, W)`.

```python
import torch

from medseg_architectures import UNet2D

num_classes = 4
model = UNet2D(in_channels=1, out_channels=num_classes, features=(16, 32, 64))
model.eval()

x = torch.randn(2, 1, 64, 64)
with torch.no_grad():
    logits = model(x)
    probabilities = torch.softmax(logits, dim=1)

assert logits.shape == (2, num_classes, 64, 64)
assert probabilities.shape == logits.shape
```

During training, pass multiclass logits directly to a loss such as
`torch.nn.CrossEntropyLoss`, with target masks shaped `(B, H, W)`.

## Choose `in_channels` From The Input Tensor

Set `in_channels` to match the channel dimension of the tensor you feed to the
model. In PyTorch image layout, the input shape is `(B, C, H, W)`.

```python
import torch

from medseg_architectures import UNet2D

grayscale = torch.randn(2, 1, 64, 64)
rgb = torch.randn(2, 3, 64, 64)
multi_modal = torch.randn(2, 4, 64, 64)

gray_model = UNet2D(in_channels=1, out_channels=2)
rgb_model = UNet2D(in_channels=3, out_channels=2)
multi_modal_model = UNet2D(in_channels=4, out_channels=2)

gray_model.eval()
rgb_model.eval()
multi_modal_model.eval()

with torch.no_grad():
    gray_logits = gray_model(grayscale)
    rgb_logits = rgb_model(rgb)
    multi_modal_logits = multi_modal_model(multi_modal)

assert gray_logits.shape == (2, 2, 64, 64)
assert rgb_logits.shape == (2, 2, 64, 64)
assert multi_modal_logits.shape == (2, 2, 64, 64)
```

A multi-modal tensor can stack modalities along the channel dimension, such as
four aligned synthetic channels shaped `(B, 4, H, W)`.

## Choose Feature Widths

The `features` sequence controls encoder widths from shallow to deep. Larger
values usually increase capacity, parameter count, and memory use.

```python
from medseg_architectures import UNet2D

tiny = UNet2D(in_channels=1, out_channels=2, features=(8, 16))
baseline = UNet2D(in_channels=1, out_channels=2, features=(16, 32, 64))
wider = UNet2D(in_channels=1, out_channels=2, features=(32, 64, 128))
```

- `features=(8, 16)`: fastest tiny model for CPU-friendly experiments.
- `features=(16, 32, 64)`: default educational baseline used by the demo.
- `features=(32, 64, 128)`: wider model with higher memory and compute cost.

Keep the sequence shallow-to-deep and use positive integers only.

## Keep Logits Separate From Probabilities

`UNet2D` returns raw logits instead of applying `sigmoid` or `softmax` inside
the model. This keeps the model compatible with stable PyTorch losses that
combine activation and loss computation.

```python
import torch

from medseg_architectures import UNet2D

x = torch.randn(2, 1, 32, 32)

binary_model = UNet2D(in_channels=1, out_channels=1, features=(8, 16))
multiclass_model = UNet2D(in_channels=1, out_channels=3, features=(8, 16))
binary_model.eval()
multiclass_model.eval()

with torch.no_grad():
    binary_logits = binary_model(x)
    binary_probabilities = torch.sigmoid(binary_logits)
    multiclass_logits = multiclass_model(x)
    multiclass_probabilities = torch.softmax(multiclass_logits, dim=1)

assert binary_probabilities.shape == (2, 1, 32, 32)
assert multiclass_probabilities.shape == (2, 3, 32, 32)
```

Use probabilities for interpretation, thresholds, or metrics. Keep raw logits
for losses such as `BCEWithLogitsLoss` and `CrossEntropyLoss`.

## Handle Odd Input Sizes

Repeated pooling and upsampling can create one-pixel mismatches for odd spatial
sizes. The implementation aligns decoder tensors to skip tensor sizes, so the
final logits preserve the input height and width.

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

This is useful for synthetic checks and real preprocessing pipelines where image
sizes are not always divisible by every pooling level.

## Compute Simple Dice And IoU

These tiny metric helpers work on synthetic binary tensors. They threshold
binary probabilities into predicted masks before computing overlap.

```python
import torch

logits = torch.randn(2, 1, 32, 32)
target = torch.randint(0, 2, (2, 1, 32, 32)).float()

pred = (torch.sigmoid(logits) > 0.5).float()
eps = 1e-6

intersection = (pred * target).sum()
pred_sum = pred.sum()
target_sum = target.sum()
union = ((pred + target) > 0).float().sum()

dice = (2 * intersection + eps) / (pred_sum + target_sum + eps)
iou = (intersection + eps) / (union + eps)

assert 0.0 <= dice <= 1.0
assert 0.0 <= iou <= 1.0
```

For multiclass experiments, compute per-class masks first, then apply the same
overlap idea class by class.

## Run A Synthetic Forward Pass

This compact forward pass mirrors the repository demo: one synthetic grayscale
batch enters the model, and the output keeps the same batch size, height, and
width while changing the channel count to `out_channels`.

```python
import torch

from medseg_architectures import UNet2D

torch.manual_seed(7)

model = UNet2D(in_channels=1, out_channels=2, features=(16, 32, 64))
model.eval()

x = torch.randn(1, 1, 65, 73)
with torch.no_grad():
    logits = model(x)

assert logits.shape == (1, 2, 65, 73)
```
