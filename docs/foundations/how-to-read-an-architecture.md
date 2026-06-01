# How To Read An Architecture

Architecture diagrams are easier to read when you break them into repeated
questions.

## Tensor Shape Notation

Architecture pages use short tensor shapes to show how data changes as it moves
through a model. A shape such as `(B, C, H, W)` describes the size of each axis
in a PyTorch image tensor:

- `B` is the batch size, or how many images are processed together.
- `C` is the number of input channels or modalities, such as one grayscale
  channel or three RGB channels.
- `K` is the number of output segmentation channels, classes, or masks.
- `F` is an internal feature-channel count chosen by the model.
- `D`, `H`, and `W` are depth, height, and width. 2D models usually use
  `(H, W)`, while 3D volume models use `(D, H, W)`.

For example, `(B, K, H, W)` means the model returns one `H` by `W` score map for
each of `K` segmentation outputs in each batch item. Shapes like `H/2`, `W/2`,
or `D/2` mean the feature map has been downsampled conceptually by about half.
Real code has to handle odd sizes carefully, because pooling a size such as
`65` cannot split into two equal integer halves.

Segmentation pages also use the word `logits`. Logits are raw model scores
before an interpretation step such as `sigmoid` for binary masks or `softmax`
for mutually exclusive classes.

## Inspecting Feature Widths In Code

Feature widths control how many channels the model carries at each resolution.
Two U-Net variants can preserve the same input and output shapes while using
very different numbers of parameters internally. The inspection utilities use
synthetic CPU tensors, so they are safe for quick educational checks without
loading medical images or external datasets.

```python
from medseg_architectures import UNet2D, shape_trace, summarize_parameters

input_shape = (1, 1, 65, 73)

for features in [(8, 16), (16, 32, 64)]:
    model = UNet2D(in_channels=1, out_channels=2, features=features)
    summary = summarize_parameters(model)
    trace = shape_trace(model, input_shape)

    print(f"features={features}")
    print(
        "parameters="
        f"total:{summary.total}, "
        f"trainable:{summary.trainable}, "
        f"frozen:{summary.frozen}"
    )

    for entry in trace:
        if entry.module_type in {"Conv2d", "ConvTranspose2d", "MaxPool2d"}:
            print(f"{entry.module_name}: {entry.input_shape} -> {entry.output_shape}")
```

The parameter summary shows the capacity cost of wider feature maps. The shape
trace shows where spatial size shrinks, where channels widen, and where the
decoder restores the output to `(B, K, H, W)`.

## Check Yourself

Before running the code above, predict which layers will change spatial size
and which layers will only change channel count. Then compare your prediction
with the printed trace. If the input is `(1, 1, 65, 73)`, explain why odd sizes
need explicit alignment in the decoder.

## Encoder

The encoder compresses the image into lower-resolution feature maps. It usually
gains context while losing spatial precision.

Ask:

- How many downsampling stages are there?
- What operation reduces resolution?
- Does the encoder use convolutions, attention, Transformers, or a mix?

## Bottleneck

The bottleneck is the deepest part of the network. It usually sees the largest
context window and has the lowest spatial resolution.

Ask:

- What information is available at the smallest feature map?
- Does this part model local patterns, global context, or both?

## Decoder

The decoder rebuilds a dense segmentation map. It usually upsamples features
until the output has the same spatial size as the input.

Ask:

- How does resolution increase?
- Are features copied from the encoder through skip connections?
- Are predictions made at one scale or multiple scales?

## Skip Connections

Skip connections carry spatial detail from encoder stages into decoder stages.
In U-Net-style models, they are a central reason boundaries can remain precise.

Ask:

- Are skip connections direct, nested, attention-filtered, or full-scale?
- What problem does the modified skip connection solve?

## Paper vs. Repo Implementation

The book separates paper concepts from code status. A reference-only
architecture is documented but not implemented. An implemented architecture has
source code, tests, and a synthetic demo in this repository.
