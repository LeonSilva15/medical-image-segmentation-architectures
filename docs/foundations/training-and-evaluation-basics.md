# Training And Evaluation Basics

Segmentation training is about turning model logits into useful masks while
keeping the tensor shapes, target labels, losses, and metrics aligned. This page
explains the common choices in an architecture-neutral way. For the basic task
definition, start with [What Is Segmentation?](what-is-segmentation.md). For
shape notation such as `(B, C, H, W)`, see
[How To Read An Architecture](how-to-read-an-architecture.md).

## Logits And Probabilities

A segmentation model usually returns logits: raw per-pixel or per-voxel scores.
Logits are not probabilities, so they are not constrained to `[0, 1]` and do not
need to sum to one across channels.

Use logits directly with logits-aware training losses. Convert logits to
probabilities only when you need interpretation, thresholding, or metrics:

- Binary segmentation: apply `sigmoid` to one foreground logit channel.
- Multiclass segmentation: apply `softmax` across the class channel.

The [U-Net overview](../architectures/unet.md) follows this pattern: the model
returns segmentation logits, and downstream code chooses the appropriate loss or
activation.

## Binary And Multiclass Targets

Binary segmentation predicts whether each location is foreground or background.
A common PyTorch shape is one output channel:

```text
logits: (B, 1, H, W)
target: (B, 1, H, W), values 0 or 1
```

```python
logits = model(image)
target = synthetic_mask.float()
loss = torch.nn.BCEWithLogitsLoss()(logits, target)
probabilities = torch.sigmoid(logits)
prediction = probabilities > 0.5
```

Multiclass segmentation predicts exactly one class per location. The model
returns one logit channel per class, while the target stores integer class IDs:

```text
logits: (B, K, H, W)
target: (B, H, W), values from 0 to K - 1
```

```python
logits = model(image)
target = synthetic_class_mask.long()
loss = torch.nn.CrossEntropyLoss()(logits, target)
probabilities = torch.softmax(logits, dim=1)
prediction = probabilities.argmax(dim=1)
```

For runnable synthetic examples with the local `UNet2D` implementation, see the
[U-Net cookbook](../architectures/unet/cookbook.md).

## Check Yourself

- For binary segmentation with one foreground channel, should the model apply
  `sigmoid` before `BCEWithLogitsLoss`?
- For multiclass segmentation with `K` mutually exclusive classes, what target
  shape does `CrossEntropyLoss` expect?
- If a batch-level Dice score is high, what per-case failure could still be
  hidden?

## Losses

`BCEWithLogitsLoss` is commonly used for binary foreground-vs-background masks.
It combines a sigmoid operation with binary cross entropy in a numerically stable
form, so training code should pass raw logits rather than pre-sigmoid
probabilities.

`CrossEntropyLoss` is commonly used for mutually exclusive multiclass masks. It
combines log-softmax and negative log likelihood internally, so training code
should pass raw logits rather than pre-softmax probabilities.

Dice loss is based on overlap between prediction and target. It is often useful
when the foreground occupies a small part of the image because it focuses on the
relationship between predicted and target mask regions rather than treating every
background location as equally informative.

Combined losses add complementary signals. A common educational pattern is:

```python
ce_loss = torch.nn.CrossEntropyLoss()(logits, target)
dice_loss = soft_dice_loss(logits, target)
loss = ce_loss + dice_loss
```

The exact weighting should be treated as an experiment setting, not an
architecture property.

## Metrics

Metrics are computed after converting logits into predictions or probabilities
in a way that matches the task.

Dice score measures overlap:

```text
Dice = 2 * intersection / (predicted area + target area)
```

IoU, also called Jaccard, measures overlap divided by union:

```text
IoU = intersection / union
```

A compact binary sketch:

```python
probabilities = torch.sigmoid(logits)
pred = probabilities > 0.5
target = synthetic_mask.bool()

intersection = (pred & target).sum()
union = (pred | target).sum()
eps = 1e-6

dice = (2 * intersection + eps) / (pred.sum() + target.sum() + eps)
iou = (intersection + eps) / (union + eps)
```

Report aggregation policy explicitly. A pooled Dice score sums intersections
across a whole batch or dataset, which can hide small-structure misses or a
single failed patient case. A safer educational report includes per-case values,
then summarizes the distribution with mean, median, and an interval such as a
bootstrap confidence interval. For multiclass segmentation, compute metrics per
class before macro-averaging or weighting by task importance.

Empty masks need a stated convention. If both prediction and target are empty,
some projects score the case as perfect, some exclude it from foreground Dice,
and some report a separate empty-case accuracy. If the target is empty but the
prediction is not, the false positive should be visible in the metric report.

Threshold policy is part of evaluation. Binary outputs need a threshold, and
changing it can trade precision for sensitivity. Choose the threshold on a
validation set, document it, and keep the test set untouched.

Sensitivity, also called recall, asks how much of the target foreground was
found:

```text
Sensitivity = true positives / (true positives + false negatives)
```

Precision asks how much predicted foreground was correct:

```text
Precision = true positives / (true positives + false positives)
```

Hausdorff distance compares boundaries by measuring the largest surface distance
between predicted and target masks. It is sensitive to isolated outlier pixels or
voxels, so reports often need clear preprocessing and postprocessing details.
HD95 reports the 95th percentile surface distance and is less dominated by one
worst outlier. Surface Dice asks whether predicted and target surfaces agree
within a chosen physical tolerance, so voxel spacing and resampling must be
documented.

For full formulas, clinical interpretation, failure modes, and runnable
examples, see [Evaluation Metrics](evaluation-metrics.md).

## Instance Segmentation Metrics

Instance segmentation outputs object IDs, not only semantic foreground pixels.
Evaluation should therefore include object-level behavior:

- Match predicted and target objects by IoU or another overlap rule.
- Report precision, recall, and average precision across IoU thresholds.
- Count split errors, where one object becomes several predictions.
- Count merge errors, where several objects become one prediction.
- Report object counts separately from semantic foreground overlap.
- When a method produces both a semantic map and final instances, evaluate both
  so post-processing failures are visible.

## Imbalance And Small Structures

Medical segmentation masks can be dominated by background. When the foreground is
small, a model can achieve a low average pixel loss while missing the structure
that matters for the experiment.

Common responses include foreground-aware sampling, patch selection that includes
small targets, class-weighted losses, Dice-style losses, and per-class metrics.
These choices belong to the training and evaluation setup. They should not be
confused with the architecture itself.

## Patch-Based Training

Patch-based training uses cropped regions instead of full images or volumes.
This can reduce memory use, increase the number of training examples, and make
it easier to sample patches that contain small foreground structures.

Patch choices affect what context the model sees. Very small patches may hide
important anatomy-level context. Very large patches may exceed memory limits or
make batches too small for stable experiments.

## Public Dataset Leakage Checks

Public and properly licensed datasets can still be misused. Before adding any
dataset-backed example, document how leakage is prevented:

- Split by patient, subject, case, or study, not by independent slices or
  patches from the same volume.
- Fit normalization, resampling decisions, cropping statistics, and other
  preprocessing parameters on the training split only.
- Keep slices, patches, augmentations, and duplicate studies from the same case
  inside the same split.
- Check for repeated acquisitions or near-duplicate studies across splits.
- Report whether evaluation is internal only or includes an external site,
  scanner, institution, or acquisition protocol.

## Sliding-Window Inference

Sliding-window inference applies a model to overlapping patches, then stitches
the patch predictions back into one full-size prediction. This is useful when a
full image or volume is too large for one forward pass.

```python
full_logits = torch.zeros(output_shape)
count = torch.zeros(output_shape)

for patch, location in sliding_windows(image, patch_size, overlap):
    patch_logits = model(patch)
    full_logits[..., location] += patch_logits
    count[..., location] += 1

full_logits = full_logits / count.clamp_min(1)
```

Real implementations often use weighted blending so overlapping patch edges do
not dominate the final prediction.

## 2D, 2.5D, And 3D Training

2D training processes one slice or image at a time with tensors such as
`(B, C, H, W)`. It is usually lighter on memory and easier to test on CPU.

2.5D training stacks nearby slices as input channels while still producing a 2D
prediction. It gives the model limited through-plane context without using full
3D convolutions.

3D training processes volumes or volume patches with tensors such as
`(B, C, D, H, W)`. It can model volumetric context directly, but memory use grows
quickly and patch-based workflows are often needed.

## Voxel Spacing And Resampling

Voxel spacing describes the physical size represented by each voxel along each
axis. Two volumes can have the same array shape but different physical spacing,
so preprocessing decisions can affect what a model learns.

Resampling changes an image or mask to a chosen spacing or grid. Image data and
label masks usually need different interpolation choices: continuous image
values can use smooth interpolation, while label masks should preserve class IDs.

Spacing and resampling are preprocessing choices. They should be documented with
the training setup because they change the relationship between array indices
and physical structure size.

## Why This Repository Uses Synthetic Demos

This repository uses synthetic tensors for tests and demos so examples can verify
model behavior without adding private medical images, PHI, patient identifiers,
DICOM headers, or clinical data. Synthetic demos also keep examples small,
CPU-friendly, reproducible, and focused on architecture mechanics.

Synthetic data cannot validate clinical performance. It is used here to check
shape contracts, logits, losses, and code paths in a safe educational setting.
