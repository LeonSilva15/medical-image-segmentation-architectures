# How To Read An Architecture

Architecture diagrams are easier to read when you break them into repeated
questions.

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
