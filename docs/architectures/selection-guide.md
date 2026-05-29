# Architecture Selection Guide

This guide explains why the book starts with the current architecture set and
how to read its coverage. The goal is not to rank models or claim clinical
readiness. The goal is to give beginners a stable map of ideas that repeatedly
appear in medical image segmentation papers, codebases, and discussions.

!!! warning "Educational scope"

    This project is for research and education, not clinical diagnosis. Local
    tests and demos use synthetic data by default. A documented architecture,
    implemented model, or successful synthetic forward pass does not establish
    clinical safety, regulatory approval, or deployment readiness.

## Why This Set Comes First

The current set starts with architectures that teach the main design patterns in
medical image segmentation:

- dense prediction from classification-style CNNs;
- encoder-decoder segmentation with skip connections;
- 3D volumetric segmentation;
- skip-connection and attention variants;
- self-configuring segmentation pipelines;
- Transformer and hybrid encoder-decoder models;
- promptable and foundation-model-style segmentation.

That sequence is intentional. FCN teaches the idea of predicting a label at each
pixel. U-Net then becomes the central medical segmentation baseline because many
later models are easiest to understand as changes to its encoder, decoder, skip
connections, or training pipeline. The later chapters show how researchers
extended that baseline with 3D operations, denser skip paths, attention gates,
self-configuration, Transformers, and prompts.

## How To Read Coverage

Architecture coverage has two different meanings in this book:

- `implemented`: the repository provides local code, CPU-friendly tests, and a
  synthetic demo for the architecture.
- `reference-only`: the architecture is included for learning and citation, but
  the repository does not provide a tested local implementation.

The book also separates where an idea came from:

- medical-specific architectures are designed for biomedical or medical
  segmentation problems;
- general computer-vision segmentation roots are included when they explain a
  concept that medical architectures build on;
- promptable or foundation-model background is included when it helps explain a
  newer medical segmentation workflow, even if the base family is broader than
  medical imaging.

The canonical architecture list lives in `data/architectures.yml`. Related names
such as 3D U-Net, base SAM, SAM-Med2D, MedSAM2, and Swin UNETR are useful
context, but they are not separate local architecture pages in the current
registry.

## Taxonomy At A Glance

Dense prediction roots begin with [FCN](fcn.md), a general computer-vision
segmentation architecture that explains how CNNs can produce dense pixel-level
outputs.

Core medical CNN baselines begin with [U-Net](unet.md) and [V-Net](vnet.md).
U-Net is the main 2D biomedical segmentation reference point in this repository.
V-Net represents the volumetric branch. 3D U-Net is related background for this
branch, but it is not a separate current page.

U-Net skip and attention variants are represented by [U-Net++](unetpp.md) and
[Attention U-Net](attention-unet.md). These pages show how later architectures
modify the information passed from encoder to decoder.

Pipeline and self-configuring systems are represented by [nnU-Net](nnunet.md).
It is included because many practical segmentation results depend on
preprocessing, training, inference, and postprocessing choices, not only on the
network block.

Transformer and hybrid models are represented by [TransUNet](transunet.md),
[Swin-Unet](swin-unet.md), and [UNETR](unetr.md). They show different ways to add
attention-based context to U-Net-style segmentation. Swin UNETR is related
context for this family, but it is not a separate current page.

Promptable and foundation-model-style segmentation is represented by
[MedSAM](medsam.md). The broader SAM family explains the promptable segmentation
idea, while MedSAM is the current canonical medical page in this book. SAM-Med2D
and MedSAM2 are related context, but they are not separate current pages.

## Architecture Coverage Table

| Architecture | Category | Why it is included | Good first use case | Implementation status |
| --- | --- | --- | --- | --- |
| [FCN](fcn.md) | General CV dense prediction root | Introduces fully convolutional pixel-level prediction, which later medical architectures build on. | Understand how a CNN becomes a segmentation model. | reference-only |
| [U-Net](unet.md) | Medical CNN baseline | Establishes the encoder-decoder and skip-connection pattern used by many medical segmentation models. | Start here for a tested local model and the core medical segmentation baseline. | implemented |
| [V-Net](vnet.md) | Medical 3D CNN baseline | Extends encoder-decoder segmentation ideas to volumetric inputs. | Learn why 3D scans need different memory and tensor-shape thinking. | reference-only |
| [U-Net++](unetpp.md) | Medical U-Net skip variant | Shows how nested skip pathways refine the information passed into the decoder. | Compare direct U-Net skips with denser skip designs. | reference-only |
| [Attention U-Net](attention-unet.md) | Medical U-Net attention variant | Adds attention gates that filter skip-connection features. | Learn how attention can focus decoder fusion. | reference-only |
| [nnU-Net](nnunet.md) | Medical self-configuring pipeline | Shows why preprocessing, training, and inference policy can matter as much as the model block. | Understand segmentation as a full pipeline rather than only a neural network. | reference-only |
| [TransUNet](transunet.md) | Medical Transformer hybrid | Combines U-Net-style decoding with Transformer context modeling. | Learn the bridge from CNN baselines to Transformer encoders. | reference-only |
| [Swin-Unet](swin-unet.md) | Medical Transformer U-shape | Rebuilds a U-shaped segmentation model around shifted-window Transformer blocks. | Study a more Transformer-native U-shaped design. | reference-only |
| [UNETR](unetr.md) | Medical 3D Transformer | Applies Transformer encoding to volumetric segmentation. | Learn how Transformer ideas are adapted to 3D medical volumes. | reference-only |
| [MedSAM](medsam.md) | Promptable medical foundation-model adaptation | Represents prompt-conditioned medical segmentation workflows. | Understand how prompts change the segmentation interface. | reference-only |

## What This Guide Does Not Claim

This guide does not claim that the covered architectures are clinically ready,
best for a particular dataset, or complete as a survey of every important model.
It also does not imply that a reference-only page has local package code.

Use the guide as a learning path. For experiments, read the original paper links,
check the implementation status, use properly licensed data, and validate the
full training and evaluation setup for the task at hand.
