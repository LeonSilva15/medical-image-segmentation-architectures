# U-Net

## Plain-Language Overview

U-Net is a segmentation architecture built around a simple idea: first compress
the image to understand context, then expand it back to the original resolution
while reusing fine details from earlier layers.

It became a core medical image segmentation baseline because medical boundaries
often require both global context and precise local information.

## What Problem It Solved

Early dense prediction models could produce pixel-level outputs, but medical
segmentation often needs sharper localization from limited annotated data. U-Net
made this easier by combining a contracting encoder, an expanding decoder, and
skip connections that copy high-resolution features into the decoder.

## Visual Architecture Schematic

This is an original schematic for this book, not a copied paper figure.

```mermaid
graph LR
    Input["Input image<br/>(B, C, H, W)"]
    E1["Encoder block 1<br/>local edges + texture"]
    E2["Encoder block 2<br/>larger patterns"]
    E3["Encoder block 3<br/>semantic context"]
    B["Bottleneck<br/>deep context"]
    D3["Decoder block 3"]
    D2["Decoder block 2"]
    D1["Decoder block 1"]
    Output["Segmentation logits<br/>(B, classes, H, W)"]

    Input --> E1 --> E2 --> E3 --> B --> D3 --> D2 --> D1 --> Output
    E3 -. skip .-> D3
    E2 -. skip .-> D2
    E1 -. skip .-> D1
```

## Step-By-Step Walkthrough

1. The encoder applies convolution blocks and downsampling. Each stage reduces
   spatial resolution and increases feature channels.
2. The bottleneck processes the smallest feature map, where the model has the
   widest contextual view.
3. The decoder upsamples the feature map step by step.
4. At each decoder stage, the model concatenates the upsampled features with the
   matching encoder features.
5. A final 1x1 convolution maps decoder features to segmentation logits.

## Implementation Walkthrough

The repository implementation lives in
`src/medseg_architectures/models/unet.py`. It is a compact 2D PyTorch baseline,
not a full training pipeline and not a claim to reproduce every detail from the
paper.

### Module Structure

`DoubleConv` is the repeated local feature extractor. It applies two `3x3`
convolutions with ReLU activations. Both convolutions use padding, so the block
changes channel count but preserves height and width.

The core block is intentionally small: padding keeps each block shape-preserving,
while the second convolution gives the model another local mixing step before
any pooling or upsampling changes resolution.

??? example "Code: `DoubleConv` block"

    ```python
    class DoubleConv(nn.Module):
        def __init__(self, in_channels: int, out_channels: int) -> None:
            super().__init__()
            self.layers = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.layers(x)
    ```

`UNet2D` wires those blocks into an encoder, bottleneck, decoder, and output
head. The `features` argument controls the encoder widths from shallow to deep.
For example, `features=(16, 32, 64)` creates three encoder stages, a 128-channel
bottleneck, and three mirrored decoder stages.

### Encoder And Bottleneck

The input tensor uses the PyTorch image layout `(batch, channels, height, width)`.
Each encoder stage first applies `DoubleConv`, then stores the result as a skip
tensor before max pooling. Saving the tensor before pooling matters because that
is the high-resolution feature map the decoder will later reuse.

The encoder uses `ModuleList` so PyTorch registers each repeated block as part
of the model while still letting the depth be controlled by `features`.

??? example "Code: encoder construction"

    ```python
    feature_list = tuple(features)
    self.down_blocks = nn.ModuleList()
    self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    current_channels = in_channels
    for feature_count in feature_list:
        self.down_blocks.append(DoubleConv(current_channels, feature_count))
        current_channels = feature_count
    ```

The bottleneck receives the smallest feature map after all pooling steps. In
this implementation, it doubles the deepest encoder width. With
`features=(16, 32, 64)`, the bottleneck has 128 channels.

The bottleneck and decoder are built from the same feature list in reverse, so
the decoder has one upsampling and skip-fusion stage for every encoder stage.

??? example "Code: bottleneck and decoder construction"

    ```python
    self.bottleneck = DoubleConv(feature_list[-1], feature_list[-1] * 2)

    current_channels = feature_list[-1] * 2
    for feature_count in reversed(feature_list):
        self.up_transposes.append(
            nn.ConvTranspose2d(
                current_channels,
                feature_count,
                kernel_size=2,
                stride=2,
            )
        )
        self.up_blocks.append(DoubleConv(feature_count * 2, feature_count))
        current_channels = feature_count
    ```

### Decoder And Skip Fusion

Each decoder stage starts with a transposed convolution that learns a `2x`
upsampling operation. The upsampled decoder tensor is then concatenated with the
matching encoder skip tensor along the channel dimension. The following
`DoubleConv` mixes those copied high-resolution features with the decoder's
coarser semantic features.

Odd input sizes can create one-pixel mismatches after repeated pooling and
upsampling. The forward pass handles that case by interpolating the decoder
tensor to the skip tensor's spatial size before concatenation. This keeps the
model usable for shapes like the synthetic demo input `(1, 1, 65, 73)`.

The forward pass shows the main U-Net data movement: save skips on the way down,
restore resolution on the way up, align odd shapes if needed, then concatenate
encoder detail with decoder context.

??? example "Code: forward pass skip fusion"

    ```python
    skips: list[torch.Tensor] = []

    for down_block in self.down_blocks:
        x = down_block(x)
        skips.append(x)
        x = self.pool(x)

    x = self.bottleneck(x)

    for skip, up_transpose, up_block in zip(
        reversed(skips),
        self.up_transposes,
        self.up_blocks,
        strict=True,
    ):
        x = up_transpose(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(
                x,
                size=skip.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        x = torch.cat((skip, x), dim=1)
        x = up_block(x)
    ```

### Output Head

The final `1x1` convolution maps the last decoder feature map to
`out_channels`. It does not change spatial size. The model returns raw logits,
so training code should pass them directly to a compatible loss, and evaluation
code should apply the appropriate activation when probabilities are needed.

The output head is a per-pixel channel projection. It converts decoder features
into raw scores without changing the final height or width.

??? example "Code: output logits"

    ```python
    self.output_conv = nn.Conv2d(feature_list[0], out_channels, kernel_size=1)

    return self.output_conv(x)
    ```

### Tensor Shape Example

With `in_channels=1`, `out_channels=2`, and `features=(16, 32, 64)`, the demo
input `(1, 1, 65, 73)` flows through the model as follows:

| Stage | Shape |
| --- | --- |
| Input | `(1, 1, 65, 73)` |
| Encoder block 1 skip | `(1, 16, 65, 73)` |
| Encoder block 2 skip | `(1, 32, 32, 36)` |
| Encoder block 3 skip | `(1, 64, 16, 18)` |
| Bottleneck | `(1, 128, 8, 9)` |
| Decoder block 3 | `(1, 64, 16, 18)` |
| Decoder block 2 | `(1, 32, 32, 36)` |
| Decoder block 1 | `(1, 16, 65, 73)` |
| Output logits | `(1, 2, 65, 73)` |

## Learning Notes For Practitioners

- Use `out_channels=1` for a single binary logit map. Use one channel per class
  for multiclass segmentation.
- Keep logits and probabilities separate. Train binary outputs with a
  logits-aware loss such as `BCEWithLogitsLoss`; train multiclass outputs with a
  loss such as `CrossEntropyLoss`. Apply `sigmoid` or `softmax` only when
  probabilities are needed for interpretation or metrics.
- The tests and demo use synthetic tensors to verify model behavior without
  introducing medical images, PHI, dataset licensing issues, or preprocessing
  assumptions.
- Change `features` by keeping a shallow-to-deep sequence of positive integers.
  More stages increase the amount of pooling and context; wider stages increase
  parameters and memory use.
- The shape tests protect the contract that output logits preserve the input
  height and width, including odd spatial sizes where pooling and upsampling do
  not divide evenly.

## What Changed Relative To FCN

U-Net keeps dense prediction but adds a symmetric encoder-decoder shape and
explicit skip connections between matching resolutions. These skip connections
help restore spatial detail that would otherwise be weakened by downsampling.

## Strengths

- Strong baseline for biomedical segmentation.
- Easy to understand and adapt.
- Preserves spatial detail through skip connections.
- Works with synthetic tests and small forward-pass demos without clinical data.

## Limitations

- The basic version is local-convolution dominated and may miss long-range
  context.
- The 2D version processes slices, not full 3D volumes.
- Strong real-world performance depends on data quality, preprocessing, loss
  choice, augmentation, and evaluation design.

## Implementation Status

| Field | Value |
| --- | --- |
| Status | implemented |
| Code | `src/medseg_architectures/models/unet.py` |
| Registry name | `unet2d` |
| Tests | `tests/test_model_shapes.py` |
| Demo | `demos/demo_forward_pass.py` |
| Data used in tests/demo | synthetic tensors only |

## Model Details

| Field | Value |
| --- | --- |
| Year | 2015 |
| Parent | FCN |
| Family | U-Net family |
| Paper title | U-Net: Convolutional Networks for Biomedical Image Segmentation |
| DOI | `10.1007/978-3-319-24574-4_28` |
| arXiv | `1505.04597` |

## Read The Original Paper

- DOI: [10.1007/978-3-319-24574-4_28](https://doi.org/10.1007/978-3-319-24574-4_28)
- arXiv: [1505.04597](https://arxiv.org/abs/1505.04597)
