# Contributing Guides

This section keeps architecture additions and model changes consistent across
the project.

Use these pages before opening a pull request:

- [Adding An Architecture](adding-an-architecture.md)
- [Architecture Metadata Conventions](metadata-conventions.md)
- [Architecture Page Template](architecture-template.md)
- [Architecture Checklist](architecture-checklist.md)
- [Code Change Checklist](code-change-checklist.md)

The short rule is that `data/architectures.yml` is the source of truth for
architecture identity, status, lineage, implementation state, and references.
Architecture pages teach the ideas; metadata keeps the project index and
validation checks consistent.

This repository is for education and research. Do not add private medical
images, PHI, patient identifiers, DICOM headers, clinical data, model weights,
or clinical-readiness claims.
