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

## Planned Work

- `Planned`: Scaffold the v1 Python project structure.
- `Planned`: Create `README.md`, `AGENTS.md`, `pyproject.toml`, and `data/architectures.yml`.
- `Planned`: Add reference-only metadata for the initial architecture list.
- `Planned`: Add `scripts/validate_references.py`.
- `Planned`: Implement a minimal `UNet2D`.
- `Planned`: Add CPU-only shape tests for `UNet2D`.
- `Planned`: Add one synthetic Python forward-pass demo.
- `Planned`: Add Mermaid lineage generation after the registry is stable.
- `Planned`: Consider GitHub Actions CI after local tests and validation are in place.
- `Planned`: Evaluate MONAI, TorchIO, notebooks, and public dataset scripts only when a later milestone needs them.

## Blockers

- None currently known.

## Change Log

- `2026-05-21`: Created private GitHub repository, committed `project-starter.md`, and pushed `main`.
- `2026-05-21`: Reviewed the starter plan and decided to narrow v1 around a lean scaffold, metadata registry, `UNet2D`, tests, and a synthetic demo.
- `2026-05-21`: Revised `project-starter.md` and created `tracker.md`.
