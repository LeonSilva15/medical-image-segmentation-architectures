# Project Tracker

## Purpose

This file records what has been completed, what decisions are currently guiding the project, what work is planned, and what blockers need attention. Keep it current as the project changes so the implementation history stays visible.

## Status Legend

- `Done`: completed and verified.
- `Current`: an active decision or direction.
- `Planned`: intended work that has not started yet.
- `Blocked`: work that cannot proceed until something changes.

## Completed

- `Done`: Initialized this directory as a Git repository on `main`.
- `Done`: Created the private GitHub repository `LeonSilva15/medical-image-segmentation-architectures`.
- `Done`: Committed `project-starter.md` with the Conventional Commit message `docs: add project starter`.
- `Done`: Pushed `main` to `origin/main`.
- `Done`: Reviewed `project-starter.md` and narrowed the implementation direction.
- `Done`: Revised `project-starter.md` to reflect the lean first milestone.
- `Done`: Created `tracker.md` as the project log.
- `Done`: Scaffolded the v1 Python project structure.
- `Done`: Created `README.md`, `AGENTS.md`, `pyproject.toml`, and `data/architectures.yml`.
- `Done`: Added reference-only metadata for the initial architecture list.
- `Done`: Added `scripts/validate_references.py`.
- `Done`: Implemented a minimal `UNet2D`.
- `Done`: Added CPU-only shape tests for `UNet2D`.
- `Done`: Added one synthetic Python forward-pass demo.
- `Done`: Added the MkDocs interactive book shell.
- `Done`: Added original Mermaid lineage and U-Net architecture diagrams.
- `Done`: Added a complete U-Net learning chapter.
- `Done`: Extended architecture metadata with book fields and paper links.
- `Done`: Updated reference validation for book metadata.

## Current Decisions

- `Current`: Repository owner is `LeonSilva15`.
- `Current`: Repository visibility is private.
- `Current`: Primary branch is `main`.
- `Current`: The project is a research-to-code repository, not just a README.
- `Current`: Architecture metadata belongs in `data/architectures.yml`.
- `Current`: Architecture entries must distinguish `reference-only` from `implemented`.
- `Current`: U-Net is the first model implementation target.
- `Current`: v1 should use plain PyTorch, pytest, and PyYAML before adding heavier medical-imaging libraries.
- `Current`: Tests and demos should use synthetic data only.
- `Current`: Do not add private medical images, PHI, patient identifiers, DICOM headers, or clinical data.
- `Current`: The repository is for research and education, not clinical diagnosis.
- `Current`: The learning front door is the MkDocs book under `docs/`.
- `Current`: Architecture images should be original repo-authored Mermaid diagrams, not copied paper figures.

## Planned Work

- `Planned`: Add Mermaid lineage generation after the registry is stable.
- `Planned`: Consider GitHub Actions CI after local tests and validation are in place.
- `Planned`: Evaluate MONAI, TorchIO, notebooks, and public dataset scripts only when a later milestone needs them.
- `Planned`: Rerun full `uv` validation, demo, tests, and lint when PyPI access is stable.
- `Planned`: Add complete chapters for FCN, V-Net, U-Net++, Attention U-Net, nnU-Net, TransUNet, Swin-Unet, UNETR, and MedSAM.
- `Planned`: Add GitHub Pages deployment after the book is reviewed locally.

## Blockers

- `Blocked`: Full `uv` verification is blocked by PyPI request timeouts while resolving packages, including `mkdocs` and `mkdocs-material`.

## Change Log

- `2026-05-21`: Created private GitHub repository, committed `project-starter.md`, and pushed `main`.
- `2026-05-21`: Reviewed the starter plan and decided to narrow v1 around a lean scaffold, metadata registry, `UNet2D`, tests, and a synthetic demo.
- `2026-05-21`: Revised `project-starter.md` and created `tracker.md`.
- `2026-05-22`: Added the lean v1 scaffold, architecture registry, `UNet2D`, tests, validation script, and synthetic demo.
- `2026-05-22`: Python 3.11 syntax compilation and YAML metadata sanity checks passed; full `uv` verification was blocked by PyPI timeouts.
- `2026-05-22`: Added the MkDocs interactive book shell, lineage diagram, U-Net chapter, and book metadata validation.
- `2026-05-22`: Local syntax, YAML, and book metadata sanity checks passed; full `uv` book build remains blocked by PyPI timeouts.
