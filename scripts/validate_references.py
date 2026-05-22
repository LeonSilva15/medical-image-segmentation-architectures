"""Validate architecture metadata."""

from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
ARCHITECTURES_PATH = ROOT / "data" / "architectures.yml"

REQUIRED_FIELDS = {
    "id",
    "name",
    "year",
    "parent",
    "paper_title",
    "doi",
    "arxiv",
    "modification",
    "technical_summary",
    "understandable_summary",
    "implementation_status",
    "code_path",
    "tests",
    "demo",
}

IMPLEMENTATION_STATUSES = {"reference-only", "implemented"}


def load_architectures() -> list[dict[str, Any]]:
    with ARCHITECTURES_PATH.open(encoding="utf-8") as metadata_file:
        data = yaml.safe_load(metadata_file)

    if not isinstance(data, dict) or "architectures" not in data:
        raise ValueError("data/architectures.yml must contain an architectures list")

    architectures = data["architectures"]
    if not isinstance(architectures, list) or not architectures:
        raise ValueError("architectures must be a non-empty list")

    return architectures


def validate_architecture(architecture: dict[str, Any], known_ids: set[str]) -> list[str]:
    errors: list[str] = []
    architecture_id = architecture.get("id", "<missing id>")

    missing_fields = REQUIRED_FIELDS - architecture.keys()
    if missing_fields:
        errors.append(f"{architecture_id}: missing fields {sorted(missing_fields)}")

    status = architecture.get("implementation_status")
    if status not in IMPLEMENTATION_STATUSES:
        errors.append(
            f"{architecture_id}: implementation_status must be one of "
            f"{sorted(IMPLEMENTATION_STATUSES)}"
        )

    parent = architecture.get("parent")
    if parent is not None and parent not in known_ids:
        errors.append(f"{architecture_id}: parent '{parent}' is not defined")

    if not architecture.get("doi") and not architecture.get("arxiv"):
        errors.append(f"{architecture_id}: at least one of doi or arxiv must be set")

    if status == "reference-only":
        if architecture.get("code_path") is not None:
            errors.append(f"{architecture_id}: reference-only entries must not set code_path")
        if architecture.get("tests") is not False:
            errors.append(f"{architecture_id}: reference-only entries must set tests to false")
        if architecture.get("demo") is not False:
            errors.append(f"{architecture_id}: reference-only entries must set demo to false")

    if status == "implemented":
        code_path = architecture.get("code_path")
        if not isinstance(code_path, str) or not code_path:
            errors.append(f"{architecture_id}: implemented entries must set code_path")
        elif not (ROOT / code_path).exists():
            errors.append(f"{architecture_id}: code_path does not exist: {code_path}")
        if architecture.get("tests") is not True:
            errors.append(f"{architecture_id}: implemented entries must set tests to true")
        if architecture.get("demo") is not True:
            errors.append(f"{architecture_id}: implemented entries must set demo to true")

    return errors


def validate() -> None:
    architectures = load_architectures()
    errors: list[str] = []

    ids = [architecture.get("id") for architecture in architectures]
    duplicate_ids = {architecture_id for architecture_id in ids if ids.count(architecture_id) > 1}
    if duplicate_ids:
        errors.append(f"duplicate architecture ids: {sorted(duplicate_ids)}")

    known_ids = {architecture_id for architecture_id in ids if isinstance(architecture_id, str)}
    for architecture in architectures:
        if not isinstance(architecture, dict):
            errors.append("architecture entries must be mappings")
            continue
        errors.extend(validate_architecture(architecture, known_ids))

    if errors:
        joined_errors = "\n".join(f"- {error}" for error in errors)
        raise SystemExit(f"Reference validation failed:\n{joined_errors}")

    print(f"Validated {len(architectures)} architecture entries.")


if __name__ == "__main__":
    validate()
