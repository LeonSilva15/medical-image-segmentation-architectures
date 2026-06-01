from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

pytest.importorskip("yaml")


def load_validator():
    script = Path(__file__).resolve().parents[1] / "scripts" / "validate_architecture_metadata.py"
    spec = importlib.util.spec_from_file_location("validate_architecture_metadata", script)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def write_metadata(tmp_path: Path, text: str) -> Path:
    data_dir = tmp_path / "data"
    data_dir.mkdir(exist_ok=True)
    metadata_path = data_dir / "architectures.yml"
    metadata_path.write_text(text.strip() + "\n", encoding="utf-8")
    return metadata_path


def messages(issues):
    return [issue.message for issue in issues]


def test_current_flat_schema_validates(tmp_path: Path) -> None:
    validator = load_validator()
    (tmp_path / "docs" / "architectures").mkdir(parents=True)
    (tmp_path / "docs" / "architectures" / "unet.md").write_text(
        "# U-Net\n\n## Implementation Walkthrough\n\n## Learning Notes For Practitioners\n",
        encoding="utf-8",
    )
    (tmp_path / "src" / "medseg_architectures" / "models").mkdir(parents=True)
    (tmp_path / "src" / "medseg_architectures" / "models" / "unet.py").write_text(
        "class UNet2D: pass\n",
        encoding="utf-8",
    )

    metadata_path = write_metadata(
        tmp_path,
        """
        architectures:
          - id: fcn
            slug: fcn
            name: Fully Convolutional Network
            year: 2015
            family: Dense prediction
            dimensionality: 2d
            modalities:
              - general-images
            segmentation_task: semantic
            output_type: semantic-logits
            prompt_type:
              - none
            supervision_type: supervised
            parent: null
            chapter_path: null
            paper_title: Fully Convolutional Networks for Semantic Segmentation
            doi: null
            arxiv: "1411.4038"
            paper_links:
              - kind: arxiv
                label: arXiv
                url: https://arxiv.org/abs/1411.4038
            modification: Converts classification CNNs into dense pixel prediction networks.
            technical_summary: FCN produces dense segmentation outputs.
            understandable_summary: FCN predicts one label per pixel.
            implementation_status: reference-only
            code_path: null
            tests: false
            demo: false
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
            doi: null
            arxiv: "1505.04597"
            paper_links:
              - kind: arxiv
                label: arXiv
                url: https://arxiv.org/abs/1505.04597
            modification: Adds a symmetric encoder-decoder with skip connections.
            technical_summary: U-Net fuses encoder and decoder features.
            understandable_summary: U-Net keeps context and boundary detail.
            implementation_status: implemented
            code_path: src/medseg_architectures/models/unet.py
            tests: true
            demo: true
        """,
    )

    issues = validator.validate_metadata(metadata_path, tmp_path)
    assert [issue for issue in issues if issue.level == "error"] == []


def test_required_coverage_fields_validate(tmp_path: Path) -> None:
    validator = load_validator()
    (tmp_path / "docs" / "architectures").mkdir(parents=True)
    (tmp_path / "docs" / "architectures" / "example.md").write_text("# Example\n", encoding="utf-8")

    metadata_path = write_metadata(
        tmp_path,
        """
        architectures:
          - id: fcn
            slug: fcn
            name: Fully Convolutional Network
            year: 2015
            family: Dense prediction
            dimensionality: 2d
            modalities:
              - general-images
            segmentation_task: semantic
            output_type: semantic-logits
            prompt_type:
              - none
            supervision_type: supervised
            parent: null
            chapter_path: docs/architectures/example.md
            paper_title: Fully Convolutional Networks for Semantic Segmentation
            doi: null
            arxiv: "1411.4038"
            paper_links:
              - kind: arxiv
                label: arXiv
                url: https://arxiv.org/abs/1411.4038
            modification: Converts classification CNNs into dense pixel prediction networks.
            technical_summary: FCN produces dense segmentation outputs.
            understandable_summary: FCN predicts one label per pixel.
            implementation_status: reference-only
            code_path: null
            tests: false
            demo: false
        """,
    )

    issues = validator.validate_metadata(metadata_path, tmp_path)
    assert [issue for issue in issues if issue.level == "error"] == []


def test_duplicate_id_is_error(tmp_path: Path) -> None:
    validator = load_validator()
    metadata_path = write_metadata(
        tmp_path,
        """
        architectures:
          - id: unet
            name: U-Net
            implementation_status: reference-only
          - id: unet
            name: Duplicate U-Net
            implementation_status: reference-only
        """,
    )

    issues = validator.validate_metadata(metadata_path, tmp_path)
    assert any("Duplicate architecture id" in message for message in messages(issues))


def test_invalid_id_and_slug_are_errors(tmp_path: Path) -> None:
    validator = load_validator()
    metadata_path = write_metadata(
        tmp_path,
        """
        architectures:
          - id: U-Net
            slug: U_Net
            name: U-Net
            implementation_status: reference-only
        """,
    )

    issues = validator.validate_metadata(metadata_path, tmp_path)
    assert any("id must use lowercase" in message for message in messages(issues))
    assert any("slug must use lowercase" in message for message in messages(issues))


def test_invalid_status_is_error(tmp_path: Path) -> None:
    validator = load_validator()
    metadata_path = write_metadata(
        tmp_path,
        """
        architectures:
          - id: unet
            name: U-Net
            implementation_status: complete
        """,
    )

    issues = validator.validate_metadata(metadata_path, tmp_path)
    assert any("Invalid status" in message for message in messages(issues))


def test_missing_name_is_error(tmp_path: Path) -> None:
    validator = load_validator()
    metadata_path = write_metadata(
        tmp_path,
        """
        architectures:
          - id: unet
            implementation_status: reference-only
        """,
    )

    issues = validator.validate_metadata(metadata_path, tmp_path)
    assert any("Missing required field: name" in message for message in messages(issues))


def test_missing_reference_identifier_is_error(tmp_path: Path) -> None:
    validator = load_validator()
    metadata_path = write_metadata(
        tmp_path,
        """
        architectures:
          - id: unet
            name: U-Net
            implementation_status: reference-only
            modification: Adds a symmetric encoder-decoder with skip connections.
            paper_links:
              - kind: paper
                label: Paper
                url: https://example.org/paper
        """,
    )

    issues = validator.validate_metadata(metadata_path, tmp_path)
    assert any(
        "At least one of doi or arxiv must be set" in message for message in messages(issues)
    )


def test_missing_modification_is_error(tmp_path: Path) -> None:
    validator = load_validator()
    metadata_path = write_metadata(
        tmp_path,
        """
        architectures:
          - id: unet
            name: U-Net
            implementation_status: reference-only
            arxiv: "1505.04597"
            paper_links:
              - kind: arxiv
                label: arXiv
                url: https://arxiv.org/abs/1505.04597
        """,
    )

    issues = validator.validate_metadata(metadata_path, tmp_path)
    assert any("Missing required field: modification" in message for message in messages(issues))


def test_missing_or_empty_paper_links_are_errors(tmp_path: Path) -> None:
    validator = load_validator()
    metadata_path = write_metadata(
        tmp_path,
        """
        architectures:
          - id: unet
            name: U-Net
            implementation_status: reference-only
            arxiv: "1505.04597"
            modification: Adds a symmetric encoder-decoder with skip connections.
          - id: fcn
            name: Fully Convolutional Network
            implementation_status: reference-only
            arxiv: "1411.4038"
            paper_links: []
            modification: Converts classification CNNs into dense pixel prediction networks.
        """,
    )

    issues = validator.validate_metadata(metadata_path, tmp_path)
    assert any("Missing required field: paper_links" in message for message in messages(issues))
    assert any("paper_links must be non-empty" in message for message in messages(issues))


def test_untrimmed_compact_strings_are_errors(tmp_path: Path) -> None:
    validator = load_validator()
    metadata_path = write_metadata(
        tmp_path,
        """
        architectures:
          - id: unet
            name: U-Net
            implementation_status: reference-only
            arxiv: "1505.04597"
            paper_title: >
              U-Net: Convolutional Networks for Biomedical Image Segmentation
            modification: " Adds a symmetric encoder-decoder with skip connections."
            paper_links:
              - kind: arxiv
                label: " arXiv"
                url: https://arxiv.org/abs/1505.04597
        """,
    )

    issues = validator.validate_metadata(metadata_path, tmp_path)
    assert any("paper_title must not have" in message for message in messages(issues))
    assert any("modification must not have" in message for message in messages(issues))
    assert any("paper_links[0].label must not have" in message for message in messages(issues))


def test_implemented_entry_requires_existing_code_path(tmp_path: Path) -> None:
    validator = load_validator()
    metadata_path = write_metadata(
        tmp_path,
        """
        architectures:
          - id: unet
            name: U-Net
            implementation_status: implemented
            tests: true
            demo: true
        """,
    )

    issues = validator.validate_metadata(metadata_path, tmp_path)
    assert any("Implemented entry is missing code_path" in message for message in messages(issues))

    metadata_path = write_metadata(
        tmp_path,
        """
        architectures:
          - id: unet
            name: U-Net
            implementation_status: implemented
            code_path: src/medseg_architectures/models/missing.py
            tests: true
            demo: true
        """,
    )

    issues = validator.validate_metadata(metadata_path, tmp_path)
    assert any("Code path not found" in message for message in messages(issues))


def test_implemented_entry_requires_tests_and_demo(tmp_path: Path) -> None:
    validator = load_validator()
    (tmp_path / "src" / "medseg_architectures" / "models").mkdir(parents=True)
    (tmp_path / "src" / "medseg_architectures" / "models" / "unet.py").write_text(
        "class UNet2D: pass\n",
        encoding="utf-8",
    )
    metadata_path = write_metadata(
        tmp_path,
        """
        architectures:
          - id: unet
            name: U-Net
            implementation_status: implemented
            code_path: src/medseg_architectures/models/unet.py
            tests: false
            demo: false
        """,
    )

    issues = validator.validate_metadata(metadata_path, tmp_path)
    assert any("tests to true" in message for message in messages(issues))
    assert any("demo to true" in message for message in messages(issues))


def test_missing_documentation_path_is_error_when_set(tmp_path: Path) -> None:
    validator = load_validator()
    metadata_path = write_metadata(
        tmp_path,
        """
        architectures:
          - id: unet
            name: U-Net
            implementation_status: reference-only
            chapter_path: docs/architectures/missing.md
        """,
    )

    issues = validator.validate_metadata(metadata_path, tmp_path)
    assert any("Documentation path not found" in message for message in messages(issues))


def test_unknown_parent_and_child_are_errors(tmp_path: Path) -> None:
    validator = load_validator()
    metadata_path = write_metadata(
        tmp_path,
        """
        architectures:
          - id: unet
            name: U-Net
            parent: missing-parent
            implementation_status: reference-only
            lineage:
              children:
                - missing-child
        """,
    )

    issues = validator.validate_metadata(metadata_path, tmp_path)
    assert any("Unknown parent id: missing-parent" in message for message in messages(issues))
    assert any("Unknown child id: missing-child" in message for message in messages(issues))


def test_malformed_paper_links_are_errors(tmp_path: Path) -> None:
    validator = load_validator()
    metadata_path = write_metadata(
        tmp_path,
        """
        architectures:
          - id: unet
            name: U-Net
            implementation_status: reference-only
            paper_links:
              - kind: website
                label: ""
                url: http://example.org/paper
              - not-a-mapping
        """,
    )

    issues = validator.validate_metadata(metadata_path, tmp_path)
    assert any("kind must be one of" in message for message in messages(issues))
    assert any("label is required" in message for message in messages(issues))
    assert any("url must be an https URL" in message for message in messages(issues))
    assert any("must be a mapping" in message for message in messages(issues))


def test_missing_coverage_fields_are_errors(tmp_path: Path) -> None:
    validator = load_validator()
    metadata_path = write_metadata(
        tmp_path,
        """
        architectures:
          - id: unet
            slug: unet
            name: U-Net
            year: 2015
            family: U-Net family
            parent: null
            chapter_path: null
            paper_title: "U-Net: Convolutional Networks for Biomedical Image Segmentation"
            doi: null
            arxiv: "1505.04597"
            paper_links:
              - kind: arxiv
                label: arXiv
                url: https://arxiv.org/abs/1505.04597
            modification: Adds a symmetric encoder-decoder with skip connections.
            technical_summary: U-Net fuses encoder and decoder features.
            understandable_summary: U-Net keeps context and boundary detail.
            implementation_status: reference-only
            code_path: null
            tests: false
            demo: false
        """,
    )

    issues = validator.validate_metadata(metadata_path, tmp_path)

    assert any("Missing required field: dimensionality" in message for message in messages(issues))
    assert any("Missing required field: modalities" in message for message in messages(issues))
    assert any("Missing required field: prompt_type" in message for message in messages(issues))


def test_invalid_coverage_values_are_errors(tmp_path: Path) -> None:
    validator = load_validator()
    metadata_path = write_metadata(
        tmp_path,
        """
        architectures:
          - id: unet
            slug: unet
            name: U-Net
            year: 2015
            family: U-Net family
            dimensionality: video
            modalities: []
            segmentation_task: classification
            output_type: masks
            prompt_type:
              - none
              - point
            supervision_type: weakly-supervised
            parent: null
            chapter_path: null
            paper_title: "U-Net: Convolutional Networks for Biomedical Image Segmentation"
            doi: null
            arxiv: "1505.04597"
            paper_links:
              - kind: arxiv
                label: arXiv
                url: https://arxiv.org/abs/1505.04597
            modification: Adds a symmetric encoder-decoder with skip connections.
            technical_summary: U-Net fuses encoder and decoder features.
            understandable_summary: U-Net keeps context and boundary detail.
            implementation_status: reference-only
            code_path: null
            tests: false
            demo: false
        """,
    )

    issues = validator.validate_metadata(metadata_path, tmp_path)

    assert any("dimensionality must be one of" in message for message in messages(issues))
    assert any("modalities must be a non-empty list" in message for message in messages(issues))
    assert any("segmentation_task must be one of" in message for message in messages(issues))
    assert any("output_type must be one of" in message for message in messages(issues))
    assert any("must not combine 'none'" in message for message in messages(issues))
    assert any("supervision_type must be one of" in message for message in messages(issues))


def test_readme_architecture_table_status_mismatch_is_error(tmp_path: Path) -> None:
    validator = load_validator()
    (tmp_path / "README.md").write_text(
        """
        # Example

        | Model | Status | Code in `src/` | Tests | Demo | Reference-only docs? |
        | --- | --- | --- | --- | --- | --- |
        | U-Net | reference-only | No | No | No | Yes |
        """.strip()
        + "\n",
        encoding="utf-8",
    )

    issues = validator.validate_readme_architecture_table(
        [{"name": "U-Net", "implementation_status": "implemented"}],
        tmp_path,
    )

    assert any("does not match metadata" in message for message in messages(issues))
