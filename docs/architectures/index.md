# Architecture Index

This index separates documented concepts from implemented code. `reference-only`
means the architecture is included for learning and citations, but this
repository does not yet provide a tested implementation.

For a beginner-friendly explanation of why these architectures come first, see
the [Architecture Selection Guide](selection-guide.md).

| Architecture | Year | Family | Status | Parent | Chapter |
| --- | ---: | --- | --- | --- | --- |
| FCN | 2015 | Dense prediction | reference-only | None | [Read](fcn.md) |
| U-Net | 2015 | U-Net family | implemented | FCN | [Read](unet.md) |
| DeepLabv3+ | 2018 | General CV dense prediction, atrous encoder-decoder | reference-only | FCN | [Read](deeplabv3plus.md) |
| 3D U-Net | 2016 | U-Net family, 3D | reference-only | U-Net | [Read](3d-unet.md) |
| V-Net | 2016 | U-Net family, 3D | reference-only | U-Net | [Read](vnet.md) |
| U-Net++ | 2018 | U-Net family, skip variants | reference-only | U-Net | [Read](unetpp.md) |
| Attention U-Net | 2018 | U-Net family, attention gates | reference-only | U-Net | [Read](attention-unet.md) |
| nnU-Net | 2021 | Self-configuring pipeline | reference-only | U-Net | [Read](nnunet.md) |
| TransUNet | 2021 | Transformer hybrid | reference-only | U-Net | [Read](transunet.md) |
| Swin-Unet | 2021 | Transformer U-shape | reference-only | TransUNet | [Read](swin-unet.md) |
| UNETR | 2022 | 3D Transformer | reference-only | TransUNet | [Read](unetr.md) |
| Swin UNETR | 2022 | 3D shifted-window Transformer | reference-only | UNETR | [Read](swin-unetr.md) |
| MedSAM | 2024 | Promptable foundation model | reference-only | None | [Read](medsam.md) |
| SAM-Med2D | 2023 | Promptable foundation model, 2D medical adaptation | reference-only | None | [Read](sam-med2d.md) |
| MedSAM2 | 2025 | Promptable foundation model, 3D and video | reference-only | None | [Read](medsam2.md) |

## How Chapters Will Grow

Each full chapter should include an understandable overview, visual schematic,
minimum architecture form, technical walkthrough, model details, limitations,
implementation status, and links to the original paper. U-Net is the first
complete local implementation example.
