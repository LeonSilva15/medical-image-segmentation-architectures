# Model Code Change Checklist

Use this checklist when changing implemented model code.

## API Compatibility

- [ ] Existing constructor arguments still work.
- [ ] Defaults preserve previous behavior unless the change is intentionally
  breaking.
- [ ] Breaking changes are documented.
- [ ] Registry creation still works.
- [ ] README and docs examples still match the API.

## Shape Behavior

For 2D segmentation models:

- [ ] Input shape is documented as `(B, C, H, W)`.
- [ ] Output shape is documented as `(B, K, H, W)`.
- [ ] Batch dimension is preserved.
- [ ] Output channel count equals `out_channels`.
- [ ] Even input sizes are tested.
- [ ] Odd input sizes are tested.

For 3D segmentation models:

- [ ] Input shape is documented as `(B, C, D, H, W)`.
- [ ] Output shape is documented as `(B, K, D, H, W)`.
- [ ] Patch-based or memory constraints are explained.
- [ ] CPU-friendly tests use small synthetic tensors.

## Validation

- [ ] Invalid `in_channels` values are rejected.
- [ ] Invalid `out_channels` values are rejected.
- [ ] Invalid feature or channel settings are rejected.
- [ ] Invalid optional arguments are rejected.
- [ ] Error messages are specific enough to help users fix the issue.

## Tests

- [ ] Unit tests added or updated.
- [ ] Registry tests pass.
- [ ] Synthetic demo still runs on CPU.
- [ ] Tests do not require external datasets.
- [ ] Tests do not require a GPU.
- [ ] Tests and demos do not add tracked medical volumes, DICOM files, TIFF
  microscopy data, model weights, or checkpoints.

## Documentation

- [ ] Architecture page still matches implementation.
- [ ] Code page still matches implementation.
- [ ] Cookbook examples still match implementation.
- [ ] README examples still work.
- [ ] Implementation status in `data/architectures.yml` is still accurate.

## Validation Commands

```sh
uv run --python 3.11 python scripts/validate_architecture_metadata.py
uv run --python 3.11 python scripts/validate_artifacts.py
uv run --python 3.11 ruff check .
uv run --python 3.11 pytest
uv run --python 3.11 --group docs mkdocs build --strict
```
