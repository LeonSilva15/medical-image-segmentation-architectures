# Architecture Addition Or Modification Checklist

Use this checklist when adding a new architecture page or changing an existing
architecture entry.

## Metadata

- [ ] Added or updated the entry in `data/architectures.yml`.
- [ ] Used a stable architecture `id`.
- [ ] Used a URL-friendly `slug`.
- [ ] Chose one valid status label.
- [ ] Added or updated `family`, `category`, and `tags` where useful.
- [ ] Added lineage parent or child relationships where appropriate.
- [ ] Added implementation fields.
- [ ] Added documentation path when a page exists.
- [ ] Added verified paper links and reference information.
- [ ] Did not invent paper titles, authors, DOI values, arXiv IDs, or claims.

## Documentation

- [ ] Added or updated the architecture page.
- [ ] Explained the problem the architecture solved.
- [ ] Explained the core idea.
- [ ] Added a `Minimum Architecture Form` section with building blocks, shape
  flow, pseudocode, and a synthetic runnable sketch.
- [ ] Explained tensor-shape intuition when useful.
- [ ] Explained what changed relative to the parent architecture.
- [ ] Added strengths and limitations.
- [ ] Stated implementation status clearly.
- [ ] Linked related architectures.
- [ ] Added references to original sources.
- [ ] Used only original repo-authored diagrams.
- [ ] Updated `mkdocs.yml` navigation when a page should appear in the book.

## Code, Only If Implemented

- [ ] Added model code under `src/medseg_architectures/models/`.
- [ ] Added educational code docstrings.
- [ ] Exported the model from the package if appropriate.
- [ ] Registered the model in the model registry if appropriate.
- [ ] Added constructor validation.
- [ ] Added CPU-only unit tests using synthetic tensors.
- [ ] Added shape preservation tests.
- [ ] Added odd and even input size tests where relevant.
- [ ] Added a synthetic demo.
- [ ] Updated cookbook and code examples.
- [ ] Added `Implementation Walkthrough` and
  `Learning Notes For Practitioners` sections.

## Safety And Scope

- [ ] No private medical images.
- [ ] No PHI.
- [ ] No patient identifiers.
- [ ] No DICOM headers with patient metadata.
- [ ] No clinical data.
- [ ] No clinical-readiness claims.
- [ ] No unsupported benchmark claims.
- [ ] No external model weights unless explicitly intended and documented.
- [ ] Public-dataset examples document patient/case-level splits, train-only
  preprocessing fitting, patch/slice grouping, duplicate checks, and
  external-validation status.

## Validation

Run these before opening a pull request:

```sh
uv run --python 3.11 python scripts/validate_references.py
uv run --python 3.11 python scripts/validate_architecture_metadata.py
uv run --python 3.11 python scripts/validate_artifacts.py
uv run --python 3.11 ruff check .
uv run --python 3.11 pytest
uv run --python 3.11 --group docs mkdocs build --strict
```
