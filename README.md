# Medical Image Segmentation Architectures

## Purpose

This repository is a research-to-code map of major medical image segmentation
architectures. It tracks how FCN, U-Net-style models, Transformer-based models,
self-configuring pipelines, and foundation-model approaches relate to each other.

The repository has two roles:

- a reference catalog of important architectures and papers
- a tested implementation space for selected models

## Current Status

The first implemented model is a minimal 2D U-Net baseline. Other listed
architectures are reference-only until code, tests, and demos are added.

## Architecture Metadata

Architecture facts live in `data/architectures.yml`. Each entry declares whether
it is `reference-only` or `implemented`, so the README does not imply that
reference-only architectures have working code.

## Implemented Models

| Model | Status | Code | Demo |
| --- | --- | --- | --- |
| U-Net 2D | implemented | `src/medseg_architectures/models/unet.py` | `demos/demo_forward_pass.py` |

## Quick Start

Use `uv` so the project can run with Python 3.11 even when the system Python is
older.

```sh
uv run --python 3.11 pytest
```

## Run The Demo

```sh
uv run --python 3.11 python demos/demo_forward_pass.py
```

The demo uses a synthetic tensor only. It does not load medical images or
clinical data.

## Validate References

```sh
uv run --python 3.11 python scripts/validate_references.py
```

## Run Lint

```sh
uv run --python 3.11 ruff check .
```

## Safety And Limitations

This repository is for research and education, not clinical diagnosis. Tests and
demos use synthetic data by default. Do not add private medical images, PHI,
patient identifiers, DICOM headers, or clinical data.

## Citation

When using an architecture, cite the original paper listed in
`data/architectures.yml`. A repository-level citation file can be added after the
initial implementation pattern is stable.
