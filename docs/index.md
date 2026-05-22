# Medical Image Segmentation Architectures

This book is a guided map of medical image segmentation architectures. It is
designed for readers who want to understand the ideas, see how architectures
relate, inspect visual schematics, and then follow citations back to the
original papers.

## How To Use This Book

Start with the foundations if segmentation is new to you. Then read the lineage
map to see how model families evolved. Finally, open individual architecture
chapters for deeper walkthroughs, model details, and paper links.

## Learning Path

1. Read [What Is Segmentation?](foundations/what-is-segmentation.md).
2. Read [How To Read An Architecture](foundations/how-to-read-an-architecture.md).
3. Explore the [Architecture Lineage](evolution/lineage.md).
4. Deep dive into [U-Net](architectures/unet.md), the first complete chapter.

## Repository Status

The repository currently implements a minimal `UNet2D` baseline. Other
architectures are included as reference entries until their chapters, tests,
demos, and implementations are added.

## Safety Boundary

This project is for research and education, not clinical diagnosis. Tests and
demos use synthetic data by default. Do not add private medical images, PHI,
patient identifiers, DICOM headers, or clinical data.
