"""Validate architecture references through the canonical metadata validator."""

from __future__ import annotations

from pathlib import Path

from validate_architecture_metadata import (
    architecture_entries,
    load_yaml,
    repo_root,
    validate_metadata,
)


def validate() -> None:
    """Validate reference-bearing architecture metadata."""

    root = repo_root()
    metadata_path = root / "data" / "architectures.yml"
    issues = validate_metadata(metadata_path, root)
    errors = [issue for issue in issues if issue.level == "error"]

    if errors:
        joined_errors = "\n".join(f"- {error.format()}" for error in errors)
        raise SystemExit(f"Reference validation failed:\n{joined_errors}")

    architecture_count = len(architecture_entries(load_yaml(Path(metadata_path))))
    print(f"Validated {architecture_count} architecture entries.")


if __name__ == "__main__":
    validate()
