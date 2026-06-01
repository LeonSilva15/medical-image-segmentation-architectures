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
- atrous/dilated convolution, ASPP, and multi-scale context;
- encoder-decoder segmentation with skip connections;
- 3D volumetric segmentation;
- residual, recurrent, multi-resolution, skip-connection, nested-block, and
  attention variants;
- self-configuring segmentation pipelines;
- Transformer and hybrid encoder-decoder models;
- instance segmentation with object-level shape predictions;
- promptable and foundation-model-style segmentation.

That sequence is intentional. FCN teaches the idea of predicting a label at each
pixel. DeepLabv3+ adds general computer-vision context for atrous convolution,
ASPP, multi-scale context, and lightweight decoder refinement. U-Net then
becomes the central medical segmentation baseline because many later models are
easiest to understand as changes to its encoder, decoder, skip connections, or
training pipeline. The later chapters show how researchers extended that
baseline with 3D operations, recurrent residual blocks, multi-resolution blocks,
denser skip paths, nested U-structure blocks, attention gates,
self-configuration, Transformers, and prompts.

## How To Read Coverage

Architecture coverage uses these reader-facing status labels:

- `implemented`: the repository provides local code, CPU-friendly tests, and a
  synthetic demo for the architecture.
- `reference-only`: the architecture is included for learning and citation, but
  the repository does not provide a tested local implementation.
- `planned`: the architecture is intended for future documentation or
  implementation work, but that work is not complete.
- `external pipeline`: the entry describes a framework-style pipeline that is
  not reimplemented as local package code.

The book also separates where an idea came from:

- medical-specific architectures are designed for biomedical or medical
  segmentation problems;
- general computer-vision segmentation roots are included when they explain a
  concept that medical architectures build on;
- promptable or foundation-model background is included when it helps explain a
  newer medical segmentation workflow, even if the base family is broader than
  medical imaging.

The canonical architecture list lives in `data/architectures.yml`. Base SAM and
SAM2 are useful general-computer-vision context, while [MedSAM](medsam.md),
[SAM-Med2D](sam-med2d.md), [SAM-Med3D](sam-med3d.md), [SegVol](segvol.md), and
[MedSAM2](medsam2.md) are tracked as reference-only medical promptable entries.

## Taxonomy At A Glance

Dense prediction roots begin with [FCN](fcn.md), a general computer-vision
segmentation architecture that explains how CNNs can produce dense pixel-level
outputs. [DeepLabv3+](deeplabv3plus.md) is also general computer-vision context,
not a medical-specific architecture, but it is included because atrous
convolution, ASPP, multi-scale context, and boundary-refining decoders appear in
many later segmentation discussions.

Core medical CNN baselines begin with [U-Net](unet.md),
[3D U-Net](3d-unet.md), and [V-Net](vnet.md). U-Net is the main 2D biomedical
segmentation reference point in this repository. 3D U-Net is the direct
volumetric extension of that idea, while V-Net represents a related volumetric
encoder-decoder branch.

U-Net block, skip, and attention variants are represented by
[Residual U-Net / ResUNet-style variants](resunet-style-variants.md),
[R2U-Net](r2unet.md), [MultiResUNet](multiresunet.md),
[U-Net++](unetpp.md), [UNet 3+](unet3plus.md),
[Attention U-Net](attention-unet.md), and [U²-Net](u2net.md). These pages show
how later architectures modify local feature blocks, nested block structure, or
the information passed from encoder to decoder.

Pipeline and self-configuring systems are represented by [nnU-Net](nnunet.md).
It is included because many practical segmentation results depend on
preprocessing, training, inference, and postprocessing choices, not only on the
network block.

Transformer and hybrid models are represented by [TransUNet](transunet.md),
[Swin-Unet](swin-unet.md), [UNETR](unetr.md), and
[Swin UNETR](swin-unetr.md). They show different ways to add attention-based
context to U-Net-style segmentation in 2D, 3D, convolutional-hybrid, and
shifted-window forms.

Instance segmentation is represented by [StarDist-3D](stardist-3d.md). It shows
how a 3D U-Net-style backbone can predict object probability and star-convex
polyhedron geometry so touching microscopy nuclei can be separated into
instances.

## Transformer Branch Comparison

| Architecture | Distinction |
| --- | --- |
| [TransUNet](transunet.md) | CNN/U-Net hybrid with Transformer context. |
| [Swin-Unet](swin-unet.md) | Swin Transformer U-shaped segmentation idea. |
| [UNETR](unetr.md) | Transformer encoder with U-Net-like decoder for 3D medical segmentation. |
| [Swin UNETR](swin-unetr.md) | Shifted-window Transformer encoder for 3D medical segmentation. |

## Instance Segmentation Branch

| Architecture | Distinction |
| --- | --- |
| [StarDist-3D](stardist-3d.md) | Predicts per-voxel object probability and radial distances for star-convex polyhedra, then prunes candidates with 3D NMS. |

Promptable and foundation-model-style segmentation is represented by
[MedSAM](medsam.md), [SAM-Med2D](sam-med2d.md),
[SAM-Med3D](sam-med3d.md), [SegVol](segvol.md), and
[MedSAM2](medsam2.md). MedSAM introduces the medical SAM-style idea,
SAM-Med2D focuses on 2D medical adaptation, SAM-Med3D and SegVol cover native
volumetric promptable branches, and MedSAM2 focuses on 3D image and video-style
prompting with memory across slices or frames.

## Architecture Coverage Table

| Architecture | Category | Why it is included | Good first use case | Implementation status |
| --- | --- | --- | --- | --- |
| [Fully Convolutional Network (FCN)](fcn.md) | General CV dense prediction root | Introduces fully convolutional pixel-level prediction, which later medical architectures build on. | Understand how a CNN becomes a segmentation model. | reference-only |
| [DeepLabv3+](deeplabv3plus.md) | General CV atrous encoder-decoder context | Explains atrous convolution, ASPP, multi-scale context, and lightweight decoder refinement. | Learn context modules and boundary refinement before comparing medical encoder-decoder variants. | reference-only |
| [U-Net](unet.md) | Medical CNN baseline | Establishes the encoder-decoder and skip-connection pattern used by many medical segmentation models. | Start here for a tested local model and the core medical segmentation baseline. | implemented |
| [3D U-Net](3d-unet.md) | Medical 3D CNN baseline | Extends U-Net's encoder-decoder and skip-connection pattern from 2D images to volumetric patches. | Learn why CT/MRI segmentation often needs through-plane context, patch-based training, and sparse-label handling. | reference-only |
| [V-Net](vnet.md) | Medical 3D CNN baseline | Extends encoder-decoder segmentation ideas to volumetric inputs. | Learn why 3D scans need different memory and tensor-shape thinking. | reference-only |
| [Residual U-Net / ResUNet-style variants](resunet-style-variants.md) | Medical U-Net residual variant | Explains the common pattern of replacing plain U-Net convolution blocks with residual blocks. | Compare local block changes with skip-path and attention changes. | reference-only |
| [R2U-Net](r2unet.md) | Medical U-Net recurrent residual variant | Combines recurrent feature refinement with residual shortcuts inside U-Net-style blocks. | Learn how recurrence and residual connections can work inside each scale. | reference-only |
| [MultiResUNet](multiresunet.md) | Medical U-Net multi-resolution variant | Adds MultiRes blocks and ResPaths for multi-scale feature extraction inside levels and refined skip paths. | Compare block-level multi-scale extraction with cross-level pyramid structure. | reference-only |
| [U-Net++](unetpp.md) | Medical U-Net skip variant | Shows how nested skip pathways refine the information passed into the decoder. | Compare direct U-Net skips with denser skip designs. | reference-only |
| [UNet 3+](unet3plus.md) | Medical U-Net full-scale skip variant | Extends U-Net++ by connecting decoder nodes to all encoder scales. | Study full-scale skip fusion and deep supervision. | reference-only |
| [Attention U-Net](attention-unet.md) | Medical U-Net attention variant | Adds attention gates that filter skip-connection features. | Learn how attention can focus decoder fusion. | reference-only |
| [U²-Net](u2net.md) | Nested U-structure block variant | Replaces each outer U-Net node with a smaller U-shaped RSU block. | Learn how nested blocks differ from changing skip pathways. | reference-only |
| [nnU-Net](nnunet.md) | Medical self-configuring pipeline | Shows why preprocessing, training, and inference policy can matter as much as the model block. | Understand segmentation as a full pipeline rather than only a neural network. | external pipeline |
| [TransUNet](transunet.md) | Medical Transformer hybrid | Combines U-Net-style decoding with Transformer context modeling. | Learn the bridge from CNN baselines to Transformer encoders. | reference-only |
| [Swin-Unet](swin-unet.md) | Medical Transformer U-shape | Rebuilds a U-shaped segmentation model around shifted-window Transformer blocks. | Study a more Transformer-native U-shaped design. | reference-only |
| [UNETR](unetr.md) | Medical 3D Transformer | Applies Transformer encoding to volumetric segmentation. | Learn how Transformer ideas are adapted to 3D medical volumes. | reference-only |
| [Swin UNETR](swin-unetr.md) | Medical 3D shifted-window Transformer | Applies Swin-style shifted-window attention to UNETR-style volumetric segmentation. | Compare full-token and windowed Transformer encoders for 3D segmentation. | reference-only |
| [StarDist-3D](stardist-3d.md) | 3D microscopy instance segmentation | Predicts star-convex polyhedra from a 3D U-Net-style backbone and uses NMS to separate object instances. | Study how object-level outputs differ from semantic voxel labels and where star-convex assumptions fail. | reference-only |
| [Cellpose](cellpose.md) | Microscopy instance segmentation | Predicts flow fields and cell probability maps, then groups pixels or voxels by flow convergence. | Compare flow-based grouping with star-convex object proposals. | reference-only |
| [WNet3D](wnet3d.md) | Self-supervised 3D microscopy segmentation | Uses coupled U-Nets with NCut and reconstruction objectives to learn region structure from unlabelled volumes. | Study self-supervised pretraining and post-hoc instance extraction. | reference-only |
| [MedSAM](medsam.md) | Promptable medical foundation-model adaptation | Represents prompt-conditioned medical segmentation workflows. | Understand how prompts change the segmentation interface. | reference-only |
| [SAM-Med2D](sam-med2d.md) | Promptable 2D medical adaptation | Shows how SAM-style prompting is adapted to 2D medical images. | Compare point, box, and mask prompts for 2D medical segmentation. | reference-only |
| [SAM-Med3D](sam-med3d.md) | Promptable 3D medical foundation model | Moves SAM-style prompting to native volumetric medical segmentation. | Compare one 3D point prompt with slice-by-slice 2D prompting. | reference-only |
| [SegVol](segvol.md) | Text-promptable volumetric foundation model | Combines volumetric image features with semantic text prompts and spatial prompts. | Learn the text-promptable branch of 3D medical foundation models. | reference-only |
| [MedSAM2](medsam2.md) | Promptable 3D and video medical adaptation | Adds memory-conditioned prompting for medical slices and frames. | Understand why 3D and video prompting need continuity validation. | reference-only |

## Current Coverage Gaps

This catalog is not a complete medical segmentation survey. The current set is
weak or missing for ultrasound, histopathology whole-slide imaging, retinal OCT
and fundus imaging, endoscopy and surgical video, cardiac cine imaging, fetal
imaging, and vessel/tree/topology-aware segmentation. Add these only with
verified metadata, source-supported claims, and clear implementation status.

## What This Guide Does Not Claim

This guide does not claim that the covered architectures are clinically ready,
best for a particular dataset, or complete as a survey of every important model.
It also does not imply that a reference-only or external pipeline page has local
package code.

Use the guide as a learning path. For experiments, read the original paper links,
check the implementation status, use properly licensed data, and validate the
full training and evaluation setup for the task at hand.
