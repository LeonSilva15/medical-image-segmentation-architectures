# Architecture Metadata Conventions

`data/architectures.yml` is the canonical source of truth for architecture
metadata.

Documentation pages teach the architecture. Metadata answers operational
questions:

- What is the stable architecture ID?
- Is it implemented locally?
- Where is the documentation page?
- What earlier architecture does it build on?
- Which original paper or source should be cited?
- Does it have tests and demos?
- What dimensionality, modality family, task type, output contract, prompt
  interface, and supervision style does it teach?

## Status Labels

Use only these implementation status labels.

| Status | Meaning |
| --- | --- |
| `implemented` | Local code exists under `src/`, is registered when appropriate, has tests, has a synthetic demo, and is documented. |
| `reference-only` | The architecture is tracked for learning and citation, but no local implementation is provided. |
| `planned` | The project intends to add documentation or implementation later, but it is not complete. |
| `external-pipeline` | The entry is primarily an external framework or pipeline, not a model reimplemented here. |
| `deprecated` | The entry is kept for historical context, but is no longer a project focus. |

## Current Canonical Schema

The repository currently uses this flat style:

```yaml
architectures:
  - id: unet
    slug: unet
    name: U-Net
    year: 2015
    family: U-Net family
    dimensionality: 2d
    modalities:
      - biomedical-images
    segmentation_task: semantic
    output_type: semantic-logits
    prompt_type:
      - none
    supervision_type: supervised
    parent: fcn
    chapter_path: docs/architectures/unet.md
    paper_title: "U-Net: Convolutional Networks for Biomedical Image Segmentation"
    doi: 10.1007/978-3-319-24574-4_28
    arxiv: "1505.04597"
    paper_links:
      - kind: doi
        label: DOI
        url: https://doi.org/10.1007/978-3-319-24574-4_28
      - kind: arxiv
        label: arXiv
        url: https://arxiv.org/abs/1505.04597
    modification: Adds a symmetric encoder-decoder with skip connections.
    technical_summary: >
      U-Net combines a contracting path for context with an expanding path for
      localization.
    understandable_summary: >
      U-Net sees broad context while preserving fine boundaries.
    implementation_status: implemented
    code_path: src/medseg_architectures/models/unet.py
    tests: true
    demo: true
```

Use real paper titles, DOI values, arXiv IDs, and URLs when replacing examples.
Do not invent missing bibliographic metadata.

## Required Fields

Every architecture entry should have:

- `id`
- `slug`
- `name`
- `year`
- `family`
- `dimensionality`
- `modalities`
- `segmentation_task`
- `output_type`
- `prompt_type`
- `supervision_type`
- `parent`
- `chapter_path`
- `implementation_status`
- `code_path`
- `tests`
- `demo`
- `paper_links`
- reference information from the original source

Use these controlled values where applicable:

- `dimensionality`: `2d`, `3d`, `2d-and-3d`, or `pipeline`.
- `segmentation_task`: `semantic`, `instance`, `promptable`, or
  `self-supervised-region`.
- `prompt_type`: a list using `none`, `point`, `box`, `mask`, `3d-point`,
  `text`, or `memory`. Do not combine `none` with other prompt values.

`modalities` should be a non-empty list of broad, source-supported modality
families such as `medical-images`, `volumetric-medical-images`, `microscopy`,
`ct`, or `mri`. `output_type` and `supervision_type` should use the existing
controlled values already present in `data/architectures.yml`.

## ID And Slug Rules

Architecture IDs should be stable and machine-friendly:

```text
lowercase letters, numbers, underscores, and hyphens only
```

Examples:

```text
unet
attention_unet
swin-unetr
```

Slugs should be URL-friendly:

```text
lowercase letters, numbers, and hyphens only
```

Examples:

```text
unet
attention-unet
swin-unetr
```

## Implementation Expectations

For `implemented` entries:

- `code_path` should point to an existing file.
- `tests` should be `true`.
- `demo` should be `true`.
- `chapter_path` should point to the architecture overview page.
- The chapter should include `Implementation Walkthrough` and
  `Learning Notes For Practitioners`.
- Tests and demos should use synthetic data only.

For `reference-only` entries:

- `code_path` should be `null`.
- `tests` should be `false`.
- `demo` should be `false`.
- Documentation should clearly state that the architecture is not implemented
  locally when a page exists.

For `planned` entries:

- Use the status only when the project intends to add documentation or code
  later.
- Keep incomplete implementation fields false or null.

For `external-pipeline` entries:

- Explain why the project does not reimplement the external pipeline.
- Link to the relevant paper or upstream project when appropriate.
- Do not mark local tests or demos as complete unless they exist in this repo.

For `deprecated` entries:

- Keep enough metadata for historical context.
- Explain the reason in documentation if a page exists.

## Safety And Scope Rules

Do not add:

- private medical images
- PHI
- patient identifiers
- DICOM headers with patient metadata
- clinical data
- model weights unless explicitly intended and documented
- clinical-readiness claims
- unsupported benchmark claims

Synthetic tensors and toy masks are preferred for examples and tests.

For public datasets, also document patient/case-level splits, train-only
preprocessing fitting, patch or slice grouping, duplicate-study checks, and
whether evaluation includes an external site or scanner.

## Validation

Run metadata validation before opening a pull request:

```sh
uv run --python 3.11 python scripts/validate_architecture_metadata.py
uv run --python 3.11 python scripts/validate_artifacts.py
```
