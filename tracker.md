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
- `Done`: Added a GitHub Actions workflow for publishing the MkDocs book to GitHub Pages.
- `Done`: Added implementation-level learning explanations for `UNet2D` in code and the U-Net book chapter.
- `Done`: Added validation requiring implemented architecture chapters to include implementation and practitioner learning sections.
- `Done`: Reran reference validation, synthetic demo, tests, lint, and strict MkDocs build after dependency resolution succeeded.
- `Done`: Added an architecture addition checklist guide with copyable metadata and chapter templates.
- `Done`: Added collapsible code excerpts to the U-Net chapter and documented the pattern for future implemented chapters.
- `Done`: Added nested U-Net resource pages for full code, cookbook notes, and a future live example.
- `Done`: Made full-code supporting pages non-collapsible and documented the convention.

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
- `Current`: Implemented architectures must include educational code docstrings plus `Implementation Walkthrough` and `Learning Notes For Practitioners` sections in their chapters.
- `Current`: Implemented architecture chapters should include curated, collapsible code excerpts for important implementation pieces.
- `Current`: Architecture chapters can use nested supporting pages for full code, cookbook recipes, and live examples when deeper material would crowd the overview.
- `Current`: Full-code supporting pages should show source directly in a normal fenced code block.
- `Current`: Agents and contributors should follow `docs/contributing/adding-an-architecture.md` before adding reference-only or implemented architectures.
- `Current`: Architecture images should be original repo-authored Mermaid diagrams, not copied paper figures.
- `Current`: GitHub Pages publishing should use GitHub Actions, not `mkdocs gh-deploy`.
- `Current`: The repository remains private unless publishing requirements force a visibility change.

## Planned Work

- `Planned`: Add Mermaid lineage generation after the registry is stable.
- `Planned`: Consider GitHub Actions test CI after local tests and validation are in place.
- `Planned`: Evaluate MONAI, TorchIO, notebooks, and public dataset scripts only when a later milestone needs them.
- `Planned`: Add complete chapters for FCN, V-Net, U-Net++, Attention U-Net, nnU-Net, TransUNet, Swin-Unet, UNETR, and MedSAM.
- `Planned`: Verify the first successful GitHub Pages deployment.

## Blockers

- `Blocked`: GitHub Pages API returned `404` before enabling or first deployment; private-repo Pages may require a paid GitHub plan.

## Change Log

- `2026-05-21`: Created private GitHub repository, committed `project-starter.md`, and pushed `main`.
- `2026-05-21`: Reviewed the starter plan and decided to narrow v1 around a lean scaffold, metadata registry, `UNet2D`, tests, and a synthetic demo.
- `2026-05-21`: Revised `project-starter.md` and created `tracker.md`.
- `2026-05-22`: Added the lean v1 scaffold, architecture registry, `UNet2D`, tests, validation script, and synthetic demo.
- `2026-05-22`: Python 3.11 syntax compilation and YAML metadata sanity checks passed; full `uv` verification was blocked by PyPI timeouts.
- `2026-05-22`: Added the MkDocs interactive book shell, lineage diagram, U-Net chapter, and book metadata validation.
- `2026-05-22`: Local syntax, YAML, and book metadata sanity checks passed; full `uv` book build remains blocked by PyPI timeouts.
- `2026-05-22`: Added the GitHub Pages deployment workflow for the MkDocs book.
- `2026-05-27`: Added implementation-level learning explanations for U-Net and validation for required implemented-chapter learning sections.
- `2026-05-27`: Added an architecture addition checklist guide and linked it from the MkDocs navigation and agent rules.
- `2026-05-27`: Added collapsible U-Net code excerpts and future guidance for architecture chapter snippets.
- `2026-05-27`: Added nested U-Net resource pages for full code, cookbook notes, and a future live example.
- `2026-05-27`: Made the U-Net full-code supporting page non-collapsible.
