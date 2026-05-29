# AGENTS.md

## Project Goal

This repository documents and implements important medical image segmentation
architectures and their major modifications.

## Rules For Codex

- Do not invent paper titles, authors, DOIs, arXiv IDs, or claims.
- Every architecture in `README.md` must have an entry in `data/architectures.yml`.
- Every architecture entry must set `implementation_status` to one of
  `implemented`, `reference-only`, `planned`, `external-pipeline`, or
  `deprecated`.
- Every architecture entry must include book metadata: `slug`, `family`,
  `chapter_path`, and `paper_links`.
- Do not copy figures from papers. Use original repo-authored diagrams and link
  readers to the original papers.
- Every implemented architecture must have code, a CPU-friendly shape test, and a
  synthetic demo.
- Every implemented architecture must include code-level educational docstrings
  plus `Implementation Walkthrough` and `Learning Notes For Practitioners`
  sections in its book chapter.
- Implemented architecture chapters should include curated, collapsible code
  excerpts for important implementation pieces. Use nested supporting pages for
  full code, cookbook recipes, or live examples when they would crowd the main
  architecture chapter. Full-code supporting pages should show source directly,
  not inside collapsible blocks.
- Follow `docs/contributing/adding-an-architecture.md` before adding or
  implementing an architecture.
- Treat `data/architectures.yml` as the canonical source of truth for
  architecture identity, status, lineage, implementation state, and references.
- New architecture pages should follow
  `docs/contributing/architecture-template.md`.
- Architecture additions or modifications should follow
  `docs/contributing/architecture-checklist.md`.
- Implemented model code changes should follow
  `docs/contributing/code-change-checklist.md`.
- Use synthetic data for tests and demos unless a public, properly licensed dataset
  is explicitly configured.
- Do not add private medical images, PHI, patient identifiers, DICOM headers, or
  clinical data.
- Use `ROADMAP.md` for current project direction, planned milestones, and active
  blockers. Use git history and issues for completed work history.

## Commands

- Validate references:
  `uv run --python 3.11 python scripts/validate_references.py`
- Validate architecture metadata:
  `uv run --python 3.11 python scripts/validate_architecture_metadata.py`
- Run demo:
  `uv run --python 3.11 python demos/demo_forward_pass.py`
- Run tests:
  `uv run --python 3.11 pytest`
- Run lint:
  `uv run --python 3.11 ruff check .`
- Build book:
  `uv run --python 3.11 --group docs mkdocs build --strict`

## Definition Of Done

A task is done only when affected book pages, architecture metadata, tests, demos,
and project direction docs are consistent.
