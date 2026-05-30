"""Validate architecture metadata."""

from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
ARCHITECTURES_PATH = ROOT / "data" / "architectures.yml"

REQUIRED_FIELDS = {
    "id",
    "slug",
    "name",
    "year",
    "family",
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

IMPLEMENTATION_STATUSES = {
    "implemented",
    "reference-only",
    "planned",
    "external-pipeline",
    "deprecated",
}
PAPER_LINK_KINDS = {"doi", "arxiv", "paper"}
IMPLEMENTED_CHAPTER_HEADINGS = (
    "## Implementation Walkthrough",
    "## Learning Notes For Practitioners",
)


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

    slug = architecture.get("slug")
    if not isinstance(slug, str) or not slug:
        errors.append(f"{architecture_id}: slug must be a non-empty string")

    family = architecture.get("family")
    if not isinstance(family, str) or not family:
        errors.append(f"{architecture_id}: family must be a non-empty string")

    chapter_path = architecture.get("chapter_path")
    if chapter_path is not None and not isinstance(chapter_path, str):
        errors.append(f"{architecture_id}: chapter_path must be a string or null")
    if isinstance(chapter_path, str) and not (ROOT / chapter_path).exists():
        errors.append(f"{architecture_id}: chapter_path does not exist: {chapter_path}")

    if not architecture.get("doi") and not architecture.get("arxiv"):
        errors.append(f"{architecture_id}: at least one of doi or arxiv must be set")

    errors.extend(validate_paper_links(architecture))

    if status in {"reference-only", "planned", "external-pipeline", "deprecated"}:
        if architecture.get("code_path") is not None:
            errors.append(f"{architecture_id}: non-implemented entries must not set code_path")
        if architecture.get("tests") is not False:
            errors.append(f"{architecture_id}: non-implemented entries must set tests to false")
        if architecture.get("demo") is not False:
            errors.append(f"{architecture_id}: non-implemented entries must set demo to false")

    if status == "implemented":
        if not isinstance(chapter_path, str) or not chapter_path:
            errors.append(f"{architecture_id}: implemented entries must set chapter_path")
        elif (ROOT / chapter_path).exists():
            errors.extend(
                validate_implemented_chapter_headings(architecture_id, ROOT / chapter_path)
            )
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


def validate_implemented_chapter_headings(architecture_id: str, chapter_path: Path) -> list[str]:
    """Require code-learning sections for implemented architecture chapters."""

    chapter_text = chapter_path.read_text(encoding="utf-8")
    errors: list[str] = []

    for heading in IMPLEMENTED_CHAPTER_HEADINGS:
        if heading not in chapter_text:
            errors.append(f"{architecture_id}: chapter is missing required heading: {heading}")

    return errors


def validate_paper_links(architecture: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    architecture_id = architecture.get("id", "<missing id>")
    paper_links = architecture.get("paper_links")

    if not isinstance(paper_links, list) or not paper_links:
        return [f"{architecture_id}: paper_links must be a non-empty list"]

    link_kinds: set[str] = set()
    for index, link in enumerate(paper_links):
        if not isinstance(link, dict):
            errors.append(f"{architecture_id}: paper_links[{index}] must be a mapping")
            continue

        kind = link.get("kind")
        label = link.get("label")
        url = link.get("url")

        if kind not in PAPER_LINK_KINDS:
            errors.append(
                f"{architecture_id}: paper_links[{index}].kind must be one of "
                f"{sorted(PAPER_LINK_KINDS)}"
            )
        else:
            link_kinds.add(kind)

        if not isinstance(label, str) or not label:
            errors.append(f"{architecture_id}: paper_links[{index}].label is required")
        if not isinstance(url, str) or not url.startswith("https://"):
            errors.append(f"{architecture_id}: paper_links[{index}].url must be an https URL")

    doi = architecture.get("doi")
    if doi:
        expected = f"https://doi.org/{doi}"
        if "doi" not in link_kinds:
            errors.append(f"{architecture_id}: doi is set but no DOI paper link exists")
        elif not any(link.get("url") == expected for link in paper_links if isinstance(link, dict)):
            errors.append(f"{architecture_id}: DOI paper link must be {expected}")

    arxiv = architecture.get("arxiv")
    if arxiv:
        expected = f"https://arxiv.org/abs/{arxiv}"
        if "arxiv" not in link_kinds:
            errors.append(f"{architecture_id}: arxiv is set but no arXiv paper link exists")
        elif not any(link.get("url") == expected for link in paper_links if isinstance(link, dict)):
            errors.append(f"{architecture_id}: arXiv paper link must be {expected}")

    return errors


def validate() -> None:
    architectures = load_architectures()
    errors: list[str] = []

    ids: list[str] = []
    for architecture in architectures:
        architecture_id = architecture.get("id")
        if isinstance(architecture_id, str):
            ids.append(architecture_id)
    duplicate_ids = {architecture_id for architecture_id in ids if ids.count(architecture_id) > 1}
    if duplicate_ids:
        errors.append(f"duplicate architecture ids: {sorted(duplicate_ids)}")

    slugs: list[str] = []
    for architecture in architectures:
        slug = architecture.get("slug")
        if isinstance(slug, str):
            slugs.append(slug)
    duplicate_slugs = {slug for slug in slugs if slugs.count(slug) > 1}
    if duplicate_slugs:
        errors.append(f"duplicate architecture slugs: {sorted(duplicate_slugs)}")

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
