# Roadmap

This file records current project direction, planned milestones, and active
blockers. Completed work belongs in git history, release notes, or issues.

## Current Direction

- The repository is a research-to-code project for important medical image
  segmentation architectures and their major modifications.
- `data/architectures.yml` is the canonical source for architecture identity,
  status, lineage, implementation state, and references.
- Architecture entries use one of `implemented`, `reference-only`, `planned`,
  `external-pipeline`, or `deprecated`.
- U-Net is the first implemented architecture target.
- v1 uses plain PyTorch, pytest, and PyYAML before adding heavier
  medical-imaging libraries.
- Tests and demos use synthetic data only unless a public, properly licensed
  dataset is explicitly configured.
- The MkDocs book under `docs/` is the learning front door.
- Implemented architectures include educational code docstrings, implementation
  walkthroughs, practitioner notes, CPU-friendly tests, and synthetic demos.
- Architecture diagrams are original repo-authored Mermaid diagrams, not copied
  paper figures.

## Planned Milestones

- Add Mermaid lineage generation after the registry is stable.
- Evaluate MONAI, TorchIO, notebooks, and public dataset scripts only when a
  later milestone needs them.
- Add complete chapters for FCN, V-Net, U-Net++, Attention U-Net, nnU-Net,
  TransUNet, Swin-Unet, UNETR, and MedSAM.
- Verify the first successful GitHub Pages deployment.

## Active Blockers

- GitHub Pages API returned `404` before enabling or first deployment;
  private-repo Pages may require a paid GitHub plan.
- The local Codex shell cannot run `uv run` verification commands because `uv`
  is not on PATH.
