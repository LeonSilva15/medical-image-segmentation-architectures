# U-Net Live Example

This page mirrors the repository's command-line synthetic demo. It does not
load medical images, clinical files, DICOM headers, model weights, or external
datasets.

Run the demo from the repository root:

```sh
uv run --locked --python 3.11 python demos/demo_forward_pass.py
```

The demo creates a synthetic tensor shaped `(1, 1, 65, 73)`, runs `UNet2D` in
evaluation mode, prints the output shape, summarizes trainable parameters, and
shows selected layer-by-layer shape transitions.

Expected leading output:

```text
input_shape=(1, 1, 65, 73)
output_shape=(1, 2, 65, 73)
parameters=total:481762, trainable:481762, frozen:0
shape_trace:
```

## What To Notice

- The output keeps the input height and width even though both spatial
  dimensions are odd.
- The output channel count changes from input channels to `out_channels`.
- Pooling halves spatial size in the encoder, while decoder stages restore it.
- The last transposed convolution reaches `(64, 72)`, then interpolation aligns
  it to the skip tensor size `(65, 73)` before concatenation.

## Check Yourself

Before running the command, predict which modules will change spatial size and
which modules will only change channel count. After running it, compare your
prediction with the trace.
