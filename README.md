# Medical Image Segmentation Architectures

This repository is an interactive learning resource for medical image
segmentation architectures. It combines a readable MkDocs book, architecture
metadata, original repo-authored diagrams, citations to the original papers, and
small tested implementations.

## Read The Book

The learning material lives under `docs/` and is configured by `mkdocs.yml`.

Start here:

- `docs/index.md`
- `docs/foundations/what-is-segmentation.md`
- `docs/evolution/lineage.md`
- `docs/architectures/unet.md`

Build the book locally with:

```sh
uv run --python 3.11 --group docs mkdocs build --strict
```

Serve it locally with:

```sh
uv run --python 3.11 --group docs mkdocs serve
```

## Publishing

The book is configured for GitHub Pages through GitHub Actions. After Pages is
enabled with **GitHub Actions** as the source, pushes to `main` build and deploy
the MkDocs site.

Expected project-site URL:

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

| Model | Status | Code | Demo |
| --- | --- | --- | --- |
| U-Net 2D | implemented | `src/medseg_architectures/models/unet.py` | `demos/demo_forward_pass.py` |

Other architectures are reference-only until code, tests, demos, and complete
chapters are added.

## Validate Metadata

```sh
uv run --python 3.11 python scripts/validate_references.py
uv run --python 3.11 python scripts/validate_architecture_metadata.py
```

## Contributing Architecture Changes

Architecture metadata is tracked in `data/architectures.yml`. Before adding or
modifying an architecture, read:

- `docs/contributing/metadata-conventions.md`
- `docs/contributing/architecture-template.md`
- `docs/contributing/architecture-checklist.md`
- `docs/contributing/code-change-checklist.md`

Keep examples synthetic and do not add real patient data, PHI, clinical data,
DICOM headers, model weights, or clinical-readiness claims.

## Run Tests

```sh
uv run --python 3.11 pytest
```

## Run The Synthetic Demo

```sh
uv run --python 3.11 python demos/demo_forward_pass.py
```

The demo uses a synthetic tensor only. It does not load medical images or
clinical data.

## Safety And Limitations

This repository is for research and education, not clinical diagnosis. Tests and
demos use synthetic data by default. Do not add private medical images, PHI,
patient identifiers, DICOM headers, or clinical data.

## Citation

When using an architecture, cite the original paper listed in
`data/architectures.yml` or `docs/references.md`.
