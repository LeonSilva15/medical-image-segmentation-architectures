# AGENTS.md

## Project Goal

This repository documents and implements important medical image segmentation
architectures and their major modifications.

## Rules For Codex

- Do not invent paper titles, authors, DOIs, arXiv IDs, or claims.
- Every architecture in `README.md` must have an entry in `data/architectures.yml`.
- Every architecture entry must set `implementation_status` to `reference-only` or
  `implemented`.
- Every implemented architecture must have code, a CPU-friendly shape test, and a
  synthetic demo.
- Use synthetic data for tests and demos unless a public, properly licensed dataset
  is explicitly configured.
- Do not add private medical images, PHI, patient identifiers, DICOM headers, or
  clinical data.
- Keep `tracker.md` current when work is completed, plans change, or blockers are
  discovered.

## Commands

- Validate references:
  `uv run --python 3.11 python scripts/validate_references.py`
- Run demo:
  `uv run --python 3.11 python demos/demo_forward_pass.py`
- Run tests:
  `uv run --python 3.11 pytest`
- Run lint:
  `uv run --python 3.11 ruff check .`

## Definition Of Done

A task is done only when documentation, architecture metadata, tests, demos, and
the tracker are consistent.
