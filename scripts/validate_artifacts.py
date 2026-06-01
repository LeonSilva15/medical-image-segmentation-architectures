"""Validate that tracked artifacts do not include unsafe medical data or weights."""

from __future__ import annotations

import argparse
import subprocess
import sys
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

ERROR_EXTENSIONS = {
    ".ckpt",
    ".dcm",
    ".dicom",
    ".h5",
    ".hdf5",
    ".mha",
    ".mhd",
    ".nii",
    ".nii.gz",
    ".nrrd",
    ".onnx",
    ".pt",
    ".pth",
    ".safetensors",
}
WARNING_EXTENSIONS = {".tif", ".tiff"}
TIFF_ALLOWED_PREFIXES = (
    Path("docs/assets"),
)


@dataclass(frozen=True)
class ArtifactIssue:
    """A potentially unsafe tracked artifact."""

    level: str
    path: Path
    message: str

    def format(self) -> str:
        """Return the issue as a human-readable line."""

        return f"[{self.level}] {self.path}: {self.message}"


def repo_root() -> Path:
    """Return the repository root for this script."""

    return Path(__file__).resolve().parents[1]


def tracked_paths(root: Path) -> list[Path]:
    """Return git-tracked paths, or an empty list when git is unavailable."""

    result = subprocess.run(
        ["git", "ls-files"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    return [Path(line) for line in result.stdout.splitlines() if line]


def load_allowlist(root: Path, allowlist_path: Path | None) -> set[Path]:
    """Load explicitly allowed artifact paths."""

    if allowlist_path is None:
        return set()

    path = allowlist_path if allowlist_path.is_absolute() else root / allowlist_path
    if not path.exists():
        raise FileNotFoundError(f"allowlist not found: {path}")

    allowed: set[Path] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            allowed.add(Path(stripped))
    return allowed


def validate_artifacts(
    paths: Iterable[Path],
    allowlist: set[Path] | None = None,
) -> list[ArtifactIssue]:
    """Validate tracked artifact paths."""

    allowed = allowlist or set()
    issues: list[ArtifactIssue] = []

    for path in paths:
        normalized = Path(path)
        if normalized in allowed:
            continue

        extension = _artifact_extension(normalized)
        if extension in ERROR_EXTENSIONS:
            issues.append(
                ArtifactIssue(
                    "error",
                    normalized,
                    "tracked medical volumes, DICOM files, model weights, and checkpoints "
                    "must be explicitly allowlisted.",
                )
            )
        elif extension in WARNING_EXTENSIONS and not _is_allowed_tiff_path(normalized):
            issues.append(
                ArtifactIssue(
                    "warning",
                    normalized,
                    "tracked TIFF files can contain microscopy data; keep them under approved "
                    "docs/assets paths or explicitly allowlist them.",
                )
            )

    return issues


def _artifact_extension(path: Path) -> str:
    lower_name = path.name.lower()
    if lower_name.endswith(".nii.gz"):
        return ".nii.gz"
    return path.suffix.lower()


def _is_allowed_tiff_path(path: Path) -> bool:
    return any(path == prefix or prefix in path.parents for prefix in TIFF_ALLOWED_PREFIXES)


def main(argv: list[str] | None = None) -> int:
    """Validate tracked artifacts from the command line."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--allowlist",
        type=Path,
        default=None,
        help="Optional newline-delimited allowlist of tracked artifact paths.",
    )
    parser.add_argument(
        "--strict-warnings",
        action="store_true",
        help="Treat warnings as errors.",
    )
    args = parser.parse_args(argv)

    root = repo_root()
    allowlist = load_allowlist(root, args.allowlist)
    issues = validate_artifacts(tracked_paths(root), allowlist)
    if args.strict_warnings:
        issues = [
            ArtifactIssue("error", issue.path, issue.message)
            if issue.level == "warning"
            else issue
            for issue in issues
        ]

    errors = [issue for issue in issues if issue.level == "error"]
    warnings = [issue for issue in issues if issue.level == "warning"]

    for issue in errors + warnings:
        print(issue.format())

    if errors:
        print(f"Artifact validation failed with {len(errors)} error(s).")
        return 1

    if warnings:
        print(f"Artifact validation passed with {len(warnings)} warning(s).")
    else:
        print("Artifact validation passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
