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
