# Architecture Index

This index separates documented concepts, local implementations, and external
pipeline coverage. Status labels use the reader-facing forms `implemented`,
`reference-only`, `planned`, and `external pipeline`. `implemented` means local
code, tests, and a demo exist. `reference-only` means the architecture is
included for learning and citations without a tested local implementation.
`planned` means future documentation or implementation work is intended but not
complete. `external pipeline` means the entry describes a framework-style
pipeline that is not reimplemented locally.

For a beginner-friendly explanation of why these architectures come first, see
the [Architecture Selection Guide](selection-guide.md).

| Architecture | Year | Family | Status | Parent | Chapter |
| --- | ---: | --- | --- | --- | --- |
| Fully Convolutional Network (FCN) | 2015 | Dense prediction | reference-only | None | [Read](fcn.md) |
| U-Net | 2015 | U-Net family | implemented | FCN | [Read](unet.md) |
| DeepLabv3+ | 2018 | General CV dense prediction, atrous encoder-decoder | reference-only | FCN | [Read](deeplabv3plus.md) |
| 3D U-Net | 2016 | U-Net family, 3D | reference-only | U-Net | [Read](3d-unet.md) |
| V-Net | 2016 | U-Net family, 3D | reference-only | U-Net | [Read](vnet.md) |
| Residual U-Net / ResUNet-style variants | 2018 | U-Net family, residual variants | reference-only | U-Net | [Read](resunet-style-variants.md) |
| R2U-Net | 2018 | unet | reference-only | Residual U-Net / ResUNet-style variants | [Read](r2unet.md) |
| MultiResUNet | 2020 | unet | reference-only | Residual U-Net / ResUNet-style variants | [Read](multiresunet.md) |
| U-Net++ | 2018 | U-Net family, skip variants | reference-only | U-Net | [Read](unetpp.md) |
| UNet 3+ | 2020 | unet | reference-only | U-Net++ | [Read](unet3plus.md) |
| Attention U-Net | 2018 | U-Net family, attention gates | reference-only | U-Net | [Read](attention-unet.md) |
| U²-Net | 2020 | unet | reference-only | U-Net | [Read](u2net.md) |
| nnU-Net | 2021 | Self-configuring pipeline | external pipeline | U-Net | [Read](nnunet.md) |
| SegResNet | 2018 | unet | reference-only | Residual U-Net / ResUNet-style variants | [Read](segresnet.md) |
| TransUNet | 2021 | Transformer hybrid | reference-only | U-Net | [Read](transunet.md) |
| Swin-Unet | 2021 | Transformer U-shape | reference-only | TransUNet | [Read](swin-unet.md) |
| UNETR | 2022 | 3D Transformer | reference-only | TransUNet | [Read](unetr.md) |
| Swin UNETR | 2022 | 3D shifted-window Transformer | reference-only | UNETR | [Read](swin-unetr.md) |
| StarDist-3D | 2020 | instance-segmentation | reference-only | 3D U-Net | [Read](stardist-3d.md) |
| MedSAM | 2024 | Promptable foundation model | reference-only | None | [Read](medsam.md) |
| SAM-Med2D | 2023 | Promptable foundation model, 2D medical adaptation | reference-only | None | [Read](sam-med2d.md) |
| SAM-Med3D | 2023 | foundation-models | reference-only | MedSAM | [Read](sam-med3d.md) |
| SegVol | 2023 | foundation-models | reference-only | MedSAM | [Read](segvol.md) |
| MedSAM2 | 2025 | Promptable foundation model, 3D and video | reference-only | None | [Read](medsam2.md) |

## How Chapters Will Grow

Each full chapter should include an understandable overview, visual schematic,
minimum architecture form, technical walkthrough, model details, limitations,
implementation status, and links to the original paper. U-Net is the first
complete local implementation example.
