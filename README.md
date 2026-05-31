# Medical Image Segmentation Architectures

This repository is an interactive learning resource for medical image
segmentation architectures. It combines a readable MkDocs book, architecture
metadata, original repo-authored diagrams, citations to the original papers, and
small tested implementations.

## Read The Book

The published book is expected at:

```text
https://leonsilva15.github.io/medical-image-segmentation-architectures/
```

The source pages live under `docs/` and are configured by `mkdocs.yml`.

Start with these book pages:

- Start Here: `docs/index.md`
- What Is Segmentation?: `docs/foundations/what-is-segmentation.md`
- Architecture Lineage: `docs/evolution/lineage.md`
- U-Net Overview: `docs/architectures/unet.md`

Build the book locally with:

```sh
uv run --locked --python 3.11 --group docs mkdocs build --strict
```

Serve it locally with:

```sh
uv run --locked --python 3.11 --group docs mkdocs serve
```

## Publishing

The book is configured for GitHub Pages through GitHub Actions. After Pages is
enabled with **GitHub Actions** as the source, pushes to `main` build and deploy
the MkDocs site.

Project-site URL:

```text
https://leonsilva15.github.io/medical-image-segmentation-architectures/
```

If the repository remains private, GitHub Pages availability depends on the
GitHub account or organization plan. If the workflow succeeds but Pages is not
served, check `Settings -> Pages -> Build and deployment -> Source` and select
`GitHub Actions`.

## What This Repository Contains

- A guided architecture book using MkDocs Material.
- A metadata registry in `data/architectures.yml`.
- Original Mermaid diagrams for architecture lineage and U-Net.
- Links to original papers through DOI and arXiv metadata.
- A minimal implemented `UNet2D` baseline with tests and a synthetic demo.

## Current Implementation Status

| Model | Status | Code in `src/` | Tests | Demo | Reference-only docs? |
| --- | --- | --- | --- | --- | --- |
| Fully Convolutional Network (FCN) | reference-only | No | No | No | Yes |
| U-Net | implemented | Yes: `src/medseg_architectures/models/unet.py` | Yes: `tests/test_model_shapes.py` | Yes: `demos/demo_forward_pass.py` | No |
| DeepLabv3+ | reference-only | No | No | No | Yes |
| 3D U-Net | reference-only | No | No | No | Yes |
| V-Net | reference-only | No | No | No | Yes |
| Residual U-Net / ResUNet-style variants | reference-only | No | No | No | Yes |
| R2U-Net | reference-only | No | No | No | Yes |
| MultiResUNet | reference-only | No | No | No | Yes |
| U-Net++ | reference-only | No | No | No | Yes |
| UNet 3+ | reference-only | No | No | No | Yes |
| Attention U-Net | reference-only | No | No | No | Yes |
| U²-Net | reference-only | No | No | No | Yes |
| nnU-Net | external pipeline | No | No | No | Yes |
| SegResNet | reference-only | No | No | No | Yes |
| TransUNet | reference-only | No | No | No | Yes |
| Swin-Unet | reference-only | No | No | No | Yes |
| UNETR | reference-only | No | No | No | Yes |
| Swin UNETR | reference-only | No | No | No | Yes |
| StarDist-3D | reference-only | No | No | No | Yes |
| MedSAM | reference-only | No | No | No | Yes |
| SAM-Med2D | reference-only | No | No | No | Yes |
| SAM-Med3D | reference-only | No | No | No | Yes |
| SegVol | reference-only | No | No | No | Yes |
| MedSAM2 | reference-only | No | No | No | Yes |

Reader-facing docs use `external pipeline`; metadata stores the same status as
`external-pipeline`. Non-U-Net entries do not provide local package code, tests,
or demos until their metadata status changes. Every architecture listed in this
table has a book page; `Reference-only docs?` means the page documents the
architecture without a local implementation.

## Validate Metadata

```sh
uv run --locked --python 3.11 python scripts/validate_references.py
uv run --locked --python 3.11 python scripts/validate_architecture_metadata.py
uv run --locked --python 3.11 ruff check .
uv run --locked --python 3.11 mypy src scripts demos tests
```

## Contributing Architecture Changes

Architecture metadata is tracked in `data/architectures.yml`. Before adding or
modifying an architecture, read:

- `docs/contributing/adding-an-architecture.md`
- `docs/contributing/metadata-conventions.md`
- `docs/contributing/architecture-template.md`
- `docs/contributing/architecture-checklist.md`
- `docs/contributing/code-change-checklist.md`

Keep examples synthetic and do not add real patient data, PHI, clinical data,
DICOM headers, model weights, or clinical-readiness claims.

## Run Tests

```sh
uv run --locked --python 3.11 pytest
```

## Run The Synthetic Demo

```sh
uv run --locked --python 3.11 python demos/demo_forward_pass.py
```

The demo uses a synthetic tensor only. It does not load medical images or
clinical data.

## Safety And Limitations

This repository is for research and education, not clinical diagnosis. Tests and
demos use synthetic data by default. Do not add private medical images, PHI,
patient identifiers, DICOM headers, or clinical data.

## Citation

Use `CITATION.cff` to cite this repository. When using or discussing a specific
architecture, also cite the original paper listed in `data/architectures.yml` or
`docs/references.md`.
