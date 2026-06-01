"""Validate architecture metadata consistency."""

from __future__ import annotations

import argparse
import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

VALID_STATUSES = {
    "implemented",
    "reference-only",
    "planned",
    "external-pipeline",
    "deprecated",
}
REQUIRED_FIELDS = {
    "id",
    "slug",
    "name",
    "year",
    "family",
    "dimensionality",
    "modalities",
    "segmentation_task",
    "output_type",
    "prompt_type",
    "supervision_type",
    "parent",
    "chapter_path",
    "paper_title",
    "doi",
    "arxiv",
    "paper_links",
    "modification",
    "technical_summary",
    "understandable_summary",
    "implementation_status",
    "code_path",
    "tests",
    "demo",
}
VALID_DIMENSIONALITY = {"2d", "3d", "2d-and-3d", "pipeline"}
VALID_SEGMENTATION_TASKS = {
    "semantic",
    "instance",
    "promptable",
    "self-supervised-region",
}
VALID_PROMPT_TYPES = {"none", "point", "box", "mask", "3d-point", "text", "memory"}
VALID_OUTPUT_TYPES = {
    "semantic-logits",
    "volumetric-logits",
    "pipeline-segmentation-logits",
    "instance-labels",
    "flow-field-and-instance-labels",
    "soft-region-map",
    "prompted-mask-logits",
    "volumetric-mask-logits",
    "memory-conditioned-mask-logits",
}
VALID_SUPERVISION_TYPES = {
    "supervised",
    "sparse-supervised",
    "self-configuring-supervised",
    "supervised-with-auxiliary-reconstruction",
    "self-supervised-pretraining",
    "prompt-supervised",
}
PAPER_LINK_KINDS = {"doi", "arxiv", "paper"}
COMPACT_STRING_FIELDS = (
    "id",
    "slug",
    "name",
    "family",
    "chapter_path",
    "paper_title",
    "doi",
    "arxiv",
    "modification",
    "implementation_status",
    "code_path",
    "dimensionality",
    "segmentation_task",
    "output_type",
    "supervision_type",
)
PAPER_LINK_STRING_FIELDS = ("kind", "label", "url")
IMPLEMENTED_CHAPTER_HEADINGS = (
    "## Implementation Walkthrough",
    "## Learning Notes For Practitioners",
)

ID_RE = re.compile(r"^[a-z0-9][a-z0-9_-]*$")
SLUG_RE = re.compile(r"^[a-z0-9][a-z0-9-]*$")
HTTPS_RE = re.compile(r"^https://")


@dataclass(frozen=True)
class ValidationIssue:
    """A metadata validation issue."""

    level: str
    architecture_id: str
    message: str

    def format(self) -> str:
        """Return the issue as a human-readable line."""

        return f"[{self.level}] {self.architecture_id}: {self.message}"


def repo_root() -> Path:
    """Return the repository root for this script."""

    return Path(__file__).resolve().parents[1]


def load_yaml(path: Path) -> Any:
    """Load a YAML file."""

    with path.open(encoding="utf-8") as metadata_file:
        return yaml.safe_load(metadata_file)


def architecture_entries(data: Any) -> list[dict[str, Any]]:
    """Return architecture entries from supported metadata root shapes."""

    if isinstance(data, dict) and isinstance(data.get("architectures"), list):
        return data["architectures"]
    if isinstance(data, list):
        return data
    raise ValueError("Expected YAML root to be a list or a mapping with 'architectures'.")


def nested(mapping: dict[str, Any], *keys: str) -> Any:
    """Return a nested mapping value or None."""

    current: Any = mapping
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def status(entry: dict[str, Any]) -> Any:
    """Return the flat or enriched implementation status."""

    return entry.get("implementation_status", entry.get("status"))


def documentation_path(entry: dict[str, Any]) -> Any:
    """Return the flat or enriched documentation page path."""

    return entry.get("chapter_path") or nested(entry, "documentation", "page")


def code_path(entry: dict[str, Any]) -> Any:
    """Return the flat or enriched implementation code path."""

    return entry.get("code_path") or nested(entry, "implementation", "code_path")


def tests_flag(entry: dict[str, Any]) -> bool:
    """Return whether tests are marked complete."""

    return bool(entry.get("tests", nested(entry, "implementation", "tests")))


def demo_flag(entry: dict[str, Any]) -> bool:
    """Return whether a demo is marked complete."""

    return bool(entry.get("demo", nested(entry, "implementation", "demo")))


def parent_ids(entry: dict[str, Any]) -> list[str]:
    """Return flat and enriched parent IDs."""

    parents: list[str] = []
    parent = entry.get("parent")
    if parent:
        parents.append(str(parent))

    lineage_parents = nested(entry, "lineage", "parents")
    if isinstance(lineage_parents, list):
        parents.extend(str(parent_id) for parent_id in lineage_parents if parent_id)

    return parents


def child_ids(entry: dict[str, Any]) -> list[str]:
    """Return enriched child IDs."""

    lineage_children = nested(entry, "lineage", "children")
    if isinstance(lineage_children, list):
        return [str(child_id) for child_id in lineage_children if child_id]
    return []


def path_exists(repo: Path, path_value: Any) -> bool:
    """Return whether a metadata path exists."""

    if not path_value:
        return True

    path = Path(str(path_value))
    if path.is_absolute():
        return path.exists()
    return (repo / path).exists()


def validate_paper_links(entry: dict[str, Any], architecture_id: str) -> list[ValidationIssue]:
    """Validate paper link structure."""

    paper_links = entry.get("paper_links")
    if paper_links is None:
        return [ValidationIssue("error", architecture_id, "Missing required field: paper_links.")]
    if not isinstance(paper_links, list):
        return [ValidationIssue("error", architecture_id, "paper_links must be a list.")]
    if not paper_links:
        return [ValidationIssue("error", architecture_id, "paper_links must be non-empty.")]

    issues: list[ValidationIssue] = []
    for index, link in enumerate(paper_links):
        label = f"paper_links[{index}]"
        if not isinstance(link, dict):
            issues.append(ValidationIssue("error", architecture_id, f"{label} must be a mapping."))
            continue

        kind = link.get("kind")
        if kind not in PAPER_LINK_KINDS:
            issues.append(
                ValidationIssue(
                    "error",
                    architecture_id,
                    f"{label}.kind must be one of {sorted(PAPER_LINK_KINDS)}.",
                )
            )

        if not link.get("label"):
            issues.append(ValidationIssue("error", architecture_id, f"{label}.label is required."))

        url = link.get("url")
        if not isinstance(url, str) or not HTTPS_RE.match(url):
            issues.append(
                ValidationIssue(
                    "error",
                    architecture_id,
                    f"{label}.url must be an https URL.",
                )
            )

    doi = entry.get("doi")
    if doi:
        expected = f"https://doi.org/{doi}"
        if not any(
            isinstance(link, dict) and link.get("kind") == "doi" and link.get("url") == expected
            for link in paper_links
        ):
            issues.append(
                ValidationIssue(
                    "error",
                    architecture_id,
                    f"DOI paper link must be {expected}.",
                )
            )

    arxiv = entry.get("arxiv")
    if arxiv:
        expected = f"https://arxiv.org/abs/{arxiv}"
        if not any(
            isinstance(link, dict)
            and link.get("kind") == "arxiv"
            and link.get("url") == expected
            for link in paper_links
        ):
            issues.append(
                ValidationIssue(
                    "error",
                    architecture_id,
                    f"arXiv paper link must be {expected}.",
                )
            )

    return issues


def validate_coverage_fields(entry: dict[str, Any], architecture_id: str) -> list[ValidationIssue]:
    """Validate structured educational and medical coverage metadata."""

    issues: list[ValidationIssue] = []

    dimensionality = entry.get("dimensionality")
    if dimensionality not in VALID_DIMENSIONALITY:
        issues.append(
            ValidationIssue(
                "error",
                architecture_id,
                f"dimensionality must be one of {sorted(VALID_DIMENSIONALITY)}.",
            )
        )

    modalities = entry.get("modalities")
    if not isinstance(modalities, list) or not modalities:
        issues.append(
            ValidationIssue("error", architecture_id, "modalities must be a non-empty list.")
        )
    elif not all(isinstance(modality, str) and modality.strip() for modality in modalities):
        issues.append(
            ValidationIssue(
                "error",
                architecture_id,
                "modalities entries must be non-empty strings.",
            )
        )

    segmentation_task = entry.get("segmentation_task")
    if segmentation_task not in VALID_SEGMENTATION_TASKS:
        issues.append(
            ValidationIssue(
                "error",
                architecture_id,
                f"segmentation_task must be one of {sorted(VALID_SEGMENTATION_TASKS)}.",
            )
        )

    output_type = entry.get("output_type")
    if output_type not in VALID_OUTPUT_TYPES:
        issues.append(
            ValidationIssue(
                "error",
                architecture_id,
                f"output_type must be one of {sorted(VALID_OUTPUT_TYPES)}.",
            )
        )

    prompt_type = entry.get("prompt_type")
    if not isinstance(prompt_type, list) or not prompt_type:
        issues.append(
            ValidationIssue("error", architecture_id, "prompt_type must be a non-empty list.")
        )
    elif any(prompt not in VALID_PROMPT_TYPES for prompt in prompt_type):
        issues.append(
            ValidationIssue(
                "error",
                architecture_id,
                f"prompt_type entries must be one of {sorted(VALID_PROMPT_TYPES)}.",
            )
        )
    elif "none" in prompt_type and len(prompt_type) > 1:
        issues.append(
            ValidationIssue(
                "error",
                architecture_id,
                "prompt_type must not combine 'none' with prompt values.",
            )
        )

    supervision_type = entry.get("supervision_type")
    if supervision_type not in VALID_SUPERVISION_TYPES:
        issues.append(
            ValidationIssue(
                "error",
                architecture_id,
                f"supervision_type must be one of {sorted(VALID_SUPERVISION_TYPES)}.",
            )
        )

    return issues


def validate_implemented_chapter_headings(
    architecture_id: str,
    chapter_path: Path,
) -> list[ValidationIssue]:
    """Require code-learning sections for implemented architecture chapters."""

    chapter_text = chapter_path.read_text(encoding="utf-8")
    issues: list[ValidationIssue] = []

    for heading in IMPLEMENTED_CHAPTER_HEADINGS:
        if heading not in chapter_text:
            issues.append(
                ValidationIssue(
                    "error",
                    architecture_id,
                    f"chapter is missing required heading: {heading}.",
                )
            )

    return issues


def validate_readme_architecture_table(
    entries: list[dict[str, Any]],
    repo: Path,
) -> list[ValidationIssue]:
    """Validate README implementation-status table parity with metadata."""

    readme_path = repo / "README.md"
    if not readme_path.exists():
        return []

    readme_text = readme_path.read_text(encoding="utf-8")
    table_rows = _readme_architecture_rows(readme_text)
    if not table_rows:
        return [
            ValidationIssue(
                "error",
                "README",
                "Current Implementation Status table was not found.",
            )
        ]

    metadata_by_name = {
        str(entry.get("name")): str(status(entry))
        for entry in entries
        if isinstance(entry, dict) and entry.get("name") and status(entry)
    }
    readme_by_name = {name: _metadata_status(status_text) for name, status_text in table_rows}

    issues: list[ValidationIssue] = []
    duplicate_names = {
        name for name, count in Counter(name for name, _status in table_rows).items() if count > 1
    }
    for name in sorted(duplicate_names):
        issues.append(ValidationIssue("error", "README", f"Duplicate README row: {name}."))

    for name, readme_status in sorted(readme_by_name.items()):
        metadata_status = metadata_by_name.get(name)
        if metadata_status is None:
            issues.append(
                ValidationIssue(
                    "error",
                    "README",
                    f"README architecture row has no metadata entry: {name}.",
                )
            )
        elif metadata_status != readme_status:
            issues.append(
                ValidationIssue(
                    "error",
                    name,
                    f"README status {readme_status!r} does not match metadata "
                    f"{metadata_status!r}.",
                )
            )

    for name in sorted(metadata_by_name.keys() - readme_by_name.keys()):
        issues.append(
            ValidationIssue(
                "error",
                name,
                "Metadata architecture is missing from README implementation table.",
            )
        )

    return issues


def _readme_architecture_rows(readme_text: str) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    in_table = False

    for line in readme_text.splitlines():
        stripped_line = line.strip()
        if stripped_line.startswith("| Model | Status |"):
            in_table = True
            continue
        if not in_table:
            continue
        if not stripped_line.startswith("|"):
            break
        if stripped_line.startswith("| ---"):
            continue

        cells = [cell.strip() for cell in stripped_line.strip("|").split("|")]
        if len(cells) >= 2:
            rows.append((_metadata_name(cells[0]), _metadata_status(cells[1])))

    return rows


def _metadata_name(name_text: str) -> str:
    return name_text.strip().removesuffix(" (FCN)")


def _metadata_status(status_text: str) -> str:
    return status_text.strip().replace(" ", "-")


def validate_trimmed_strings(entry: dict[str, Any], architecture_id: str) -> list[ValidationIssue]:
    """Require compact metadata strings to avoid leading or trailing whitespace."""

    issues: list[ValidationIssue] = []

    for field in COMPACT_STRING_FIELDS:
        value = entry.get(field)
        if isinstance(value, str) and value != value.strip():
            issues.append(
                ValidationIssue(
                    "error",
                    architecture_id,
                    f"{field} must not have leading or trailing whitespace.",
                )
            )

    nested_compact_values = {
        "documentation.page": nested(entry, "documentation", "page"),
        "implementation.code_path": nested(entry, "implementation", "code_path"),
    }
    for field, value in nested_compact_values.items():
        if isinstance(value, str) and value != value.strip():
            issues.append(
                ValidationIssue(
                    "error",
                    architecture_id,
                    f"{field} must not have leading or trailing whitespace.",
                )
            )

    paper_links = entry.get("paper_links")
    if isinstance(paper_links, list):
        for index, link in enumerate(paper_links):
            if not isinstance(link, dict):
                continue
            for field in PAPER_LINK_STRING_FIELDS:
                value = link.get(field)
                if isinstance(value, str) and value != value.strip():
                    issues.append(
                        ValidationIssue(
                            "error",
                            architecture_id,
                            f"paper_links[{index}].{field} must not have leading or "
                            "trailing whitespace.",
                        )
                    )

    return issues


def validate_metadata(
    metadata_path: Path,
    repo: Path | None = None,
    *,
    strict_warnings: bool = False,
) -> list[ValidationIssue]:
    """Validate architecture metadata and return issues."""

    root = repo or metadata_path.resolve().parents[1]
    entries = architecture_entries(load_yaml(metadata_path))
    issues: list[ValidationIssue] = []

    seen_ids: set[str] = set()
    seen_slugs: set[str] = set()

    for index, entry in enumerate(entries):
        fallback_id = f"entry-{index}"
        if not isinstance(entry, dict):
            issues.append(
                ValidationIssue("error", fallback_id, "Architecture entry must be a mapping.")
            )
            continue

        architecture_id = str(entry.get("id") or fallback_id)
        issues.extend(validate_trimmed_strings(entry, architecture_id))

        missing_fields = REQUIRED_FIELDS - entry.keys()
        for field in sorted(missing_fields):
            issues.append(
                ValidationIssue(
                    "error",
                    architecture_id,
                    f"Missing required field: {field}.",
                )
            )

        if not entry.get("id"):
            issues.append(ValidationIssue("error", architecture_id, "Missing required field: id."))
        elif not ID_RE.match(architecture_id):
            issues.append(
                ValidationIssue(
                    "error",
                    architecture_id,
                    "id must use lowercase letters, numbers, underscores, or hyphens.",
                )
            )

        if architecture_id in seen_ids:
            issues.append(ValidationIssue("error", architecture_id, "Duplicate architecture id."))
        seen_ids.add(architecture_id)

        if not entry.get("name"):
            issues.append(
                ValidationIssue("error", architecture_id, "Missing required field: name.")
            )

        if not entry.get("modification"):
            issues.append(
                ValidationIssue(
                    "error",
                    architecture_id,
                    "Missing required field: modification.",
                )
            )

        if not entry.get("technical_summary"):
            issues.append(
                ValidationIssue(
                    "error",
                    architecture_id,
                    "Missing required field: technical_summary.",
                )
            )

        if not entry.get("understandable_summary"):
            issues.append(
                ValidationIssue(
                    "error",
                    architecture_id,
                    "Missing required field: understandable_summary.",
                )
            )

        if not entry.get("doi") and not entry.get("arxiv"):
            issues.append(
                ValidationIssue(
                    "error",
                    architecture_id,
                    "At least one of doi or arxiv must be set.",
                )
            )

        slug = entry.get("slug")
        if slug is not None:
            slug_text = str(slug)
            if not SLUG_RE.match(slug_text):
                issues.append(
                    ValidationIssue(
                        "error",
                        architecture_id,
                        "slug must use lowercase letters, numbers, and hyphens.",
                    )
                )
            if slug_text in seen_slugs:
                issues.append(
                    ValidationIssue("error", architecture_id, "Duplicate architecture slug.")
                )
            seen_slugs.add(slug_text)

        entry_status = status(entry)
        if not entry_status:
            issues.append(
                ValidationIssue(
                    "error",
                    architecture_id,
                    "Missing status: use implementation_status or status.",
                )
            )
        elif str(entry_status) not in VALID_STATUSES:
            issues.append(
                ValidationIssue(
                    "error",
                    architecture_id,
                    f"Invalid status {entry_status!r}. Expected one of {sorted(VALID_STATUSES)}.",
                )
            )

        issues.extend(validate_coverage_fields(entry, architecture_id))

        doc_path = documentation_path(entry)
        if doc_path and not path_exists(root, doc_path):
            issues.append(
                ValidationIssue(
                    "error",
                    architecture_id,
                    f"Documentation path not found: {doc_path}",
                )
            )

        implementation_path = code_path(entry)
        if implementation_path and not path_exists(root, implementation_path):
            issues.append(
                ValidationIssue(
                    "error",
                    architecture_id,
                    f"Code path not found: {implementation_path}",
                )
            )

        if entry_status == "implemented":
            if not doc_path:
                issues.append(
                    ValidationIssue(
                        "error",
                        architecture_id,
                        "Implemented entry is missing chapter_path.",
                    )
                )
            elif path_exists(root, doc_path):
                issues.extend(
                    validate_implemented_chapter_headings(architecture_id, root / str(doc_path))
                )
            if not implementation_path:
                issues.append(
                    ValidationIssue(
                        "error",
                        architecture_id,
                        "Implemented entry is missing code_path.",
                    )
                )
            if not tests_flag(entry):
                issues.append(
                    ValidationIssue(
                        "error",
                        architecture_id,
                        "Implemented entry must set tests to true.",
                    )
                )
            if not demo_flag(entry):
                issues.append(
                    ValidationIssue(
                        "error",
                        architecture_id,
                        "Implemented entry must set demo to true.",
                    )
                )

        if entry_status in {"reference-only", "planned", "external-pipeline", "deprecated"}:
            if implementation_path:
                issues.append(
                    ValidationIssue(
                        "error",
                        architecture_id,
                        "Non-implemented entries must not set code_path.",
                    )
                )
            if tests_flag(entry):
                issues.append(
                    ValidationIssue(
                        "error",
                        architecture_id,
                        "Non-implemented entries must set tests to false.",
                    )
                )
            if demo_flag(entry):
                issues.append(
                    ValidationIssue(
                        "error",
                        architecture_id,
                        "Non-implemented entries must set demo to false.",
                    )
                )

        issues.extend(validate_paper_links(entry, architecture_id))

    all_ids = {
        str(entry.get("id"))
        for entry in entries
        if isinstance(entry, dict) and entry.get("id") is not None
    }
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        architecture_id = str(entry.get("id") or "unknown")
        for parent_id in parent_ids(entry):
            if parent_id not in all_ids:
                issues.append(
                    ValidationIssue("error", architecture_id, f"Unknown parent id: {parent_id}")
                )
        for child_id in child_ids(entry):
            if child_id not in all_ids:
                issues.append(
                    ValidationIssue("error", architecture_id, f"Unknown child id: {child_id}")
                )

    issues.extend(validate_readme_architecture_table(entries, root))

    if strict_warnings:
        return [
            ValidationIssue("error", issue.architecture_id, issue.message)
            if issue.level == "warning"
            else issue
            for issue in issues
        ]

    return issues


def main(argv: list[str] | None = None) -> int:
    """Validate architecture metadata from the command line."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "metadata_path",
        nargs="?",
        default="data/architectures.yml",
        help="Path to architecture metadata YAML file.",
    )
    parser.add_argument(
        "--strict-warnings",
        action="store_true",
        help="Treat warnings as errors.",
    )
    args = parser.parse_args(argv)

    root = repo_root()
    metadata_path = Path(args.metadata_path)
    if not metadata_path.is_absolute():
        metadata_path = root / metadata_path

    issues = validate_metadata(metadata_path, root, strict_warnings=args.strict_warnings)
    errors = [issue for issue in issues if issue.level == "error"]
    warnings = [issue for issue in issues if issue.level == "warning"]

    for issue in errors + warnings:
        print(issue.format())

    if errors:
        print(f"Architecture metadata validation failed with {len(errors)} error(s).")
        return 1

    if warnings:
        print(f"Architecture metadata validation passed with {len(warnings)} warning(s).")
    else:
        print("Architecture metadata validation passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
