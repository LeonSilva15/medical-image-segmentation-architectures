# Architecture Index

This index separates documented concepts from implemented code. `reference-only`
means the architecture is included for learning and citations, but this
repository does not yet provide a tested implementation.

| Architecture | Year | Family | Status | Parent | Chapter |
| --- | ---: | --- | --- | --- | --- |
| FCN | 2015 | Dense prediction | reference-only | None | Planned |
| U-Net | 2015 | U-Net family | implemented | FCN | [Read](unet.md) |
| V-Net | 2016 | U-Net family, 3D | reference-only | U-Net | Planned |
| U-Net++ | 2018 | U-Net family, skip variants | reference-only | U-Net | Planned |
| Attention U-Net | 2018 | U-Net family, attention gates | reference-only | U-Net | Planned |
| nnU-Net | 2021 | Self-configuring pipeline | reference-only | U-Net | Planned |
| TransUNet | 2021 | Transformer hybrid | reference-only | U-Net | Planned |
| Swin-Unet | 2021 | Transformer U-shape | reference-only | TransUNet | Planned |
| UNETR | 2022 | 3D Transformer | reference-only | TransUNet | Planned |
| MedSAM | 2024 | Promptable foundation model | reference-only | None | Planned |

## How Chapters Will Grow

Each full chapter should include an understandable overview, visual schematic,
technical walkthrough, model details, limitations, implementation status, and
links to the original paper. U-Net is the first complete example.
