# Medical Image Segmentation Architectures

This book is a guided map of medical image segmentation architectures. It is
designed for readers who want to understand the ideas, see how architectures
relate, inspect visual schematics, and then follow citations back to the
original papers.

## How To Use This Book

Start with the foundations if segmentation is new to you. Then read FCN and
U-Net before using the lineage map and selection guide to branch into later
architectures.

## Learning Path

1. Read [What Is Segmentation?](foundations/what-is-segmentation.md).
2. Read [How To Read An Architecture](foundations/how-to-read-an-architecture.md).
3. Read [Training And Evaluation Basics](foundations/training-and-evaluation-basics.md).
4. Read [FCN](architectures/fcn.md) for the dense-prediction starting point.
5. Deep dive into [U-Net](architectures/unet.md), the first complete chapter.
6. Use the [Architecture Selection Guide](architectures/selection-guide.md) and
   [Architecture Lineage](evolution/lineage.md) to choose the next branch.

## Check Yourself

Before opening later architecture pages, you should be able to say what shape a
2D segmentation model usually receives, why logits are not probabilities, and
why U-Net adds skip connections to an encoder-decoder.

## Repository Status

The repository currently implements a minimal `UNet2D` baseline. Other
architectures are included as reference entries until their chapters, tests,
demos, and implementations are added.

## Safety Boundary

This project is for research and education, not clinical diagnosis. Tests and
demos use synthetic data by default. Do not add private medical images, PHI,
patient identifiers, DICOM headers, or clinical data.
