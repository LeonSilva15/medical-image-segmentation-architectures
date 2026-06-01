import importlib.util
import sys
from pathlib import Path


def load_validator():
    script = Path(__file__).resolve().parents[1] / "scripts" / "validate_artifacts.py"
    spec = importlib.util.spec_from_file_location("validate_artifacts", script)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def messages(issues):
    return [issue.message for issue in issues]


def test_artifact_validator_errors_on_medical_volume_or_weight_files() -> None:
    validator = load_validator()
    issues = validator.validate_artifacts(
        [
            Path("data/private_scan.dcm"),
            Path("weights/model.pt"),
            Path("volumes/case.nii.gz"),
        ]
    )

    assert len([issue for issue in issues if issue.level == "error"]) == 3
    assert all("must be explicitly allowlisted" in message for message in messages(issues))


def test_artifact_validator_allows_explicit_allowlist() -> None:
    validator = load_validator()
    issues = validator.validate_artifacts(
        [Path("weights/model.pt")],
        allowlist={Path("weights/model.pt")},
    )

    assert issues == []


def test_artifact_validator_warns_on_tiff_outside_docs_assets() -> None:
    validator = load_validator()
    issues = validator.validate_artifacts([Path("data/microscopy.tiff")])

    assert len(issues) == 1
    assert issues[0].level == "warning"
    assert "TIFF files can contain microscopy data" in issues[0].message


def test_artifact_validator_allows_tiff_under_docs_assets() -> None:
    validator = load_validator()
    issues = validator.validate_artifacts([Path("docs/assets/example.tiff")])

    assert issues == []
