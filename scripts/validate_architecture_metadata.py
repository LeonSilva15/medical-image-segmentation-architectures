"""Validate architecture metadata consistency."""

from __future__ import annotations

import argparse
import re
import sys
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
)
PAPER_LINK_STRING_FIELDS = ("kind", "label", "url")

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

    return issues


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
