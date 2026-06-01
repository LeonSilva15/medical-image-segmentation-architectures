# What Is Segmentation?

Image segmentation means assigning a label to each pixel or voxel in an image.
Instead of answering "what is in this image?", a segmentation model answers
"where is each structure?".

In medical imaging, segmentation is often used to outline anatomy, lesions,
organs, vessels, tumors, or surgical targets. The output can be a binary mask,
where each location is foreground or background, or a multiclass mask with one
class per anatomical or pathological structure.

## Why It Is Hard

Medical segmentation is difficult because targets can be small, boundaries can
be ambiguous, scans can be noisy, and datasets are often limited. A model must
combine local detail with broader image context. Many architecture changes in
this book are different answers to that same problem.

## Common Output Shapes

For a 2D image batch, a segmentation model often maps:

```text
(batch, image_channels, height, width)
```

to:

```text
(batch, output_classes, height, width)
```

The height and width are usually preserved so each output location corresponds
to the same spatial location in the input image.

## What To Watch For

- How the model captures context.
- How the model preserves spatial detail.
- Whether the model handles 2D slices or 3D volumes.
- Whether the model is an architecture only or a full training pipeline.
- Which assumptions come from the original paper and which come from this repo's
  simplified implementation.

## Check Yourself

- If a model outputs `(B, 3, H, W)`, what does the `3` usually represent?
- Why is a segmentation mask different from an image-level class label?
- When might a 2D slice model miss information that a 3D volume model can see?
