from pathlib import Path


def test_unet_full_code_page_matches_source() -> None:
    root = Path(__file__).resolve().parents[1]
    source = (root / "src" / "medseg_architectures" / "models" / "unet.py").read_text(
        encoding="utf-8"
    )
    code_page = (root / "docs" / "architectures" / "unet" / "code.md").read_text(
        encoding="utf-8"
    )

    assert _extract_single_python_block(code_page) == source


def _extract_single_python_block(markdown: str) -> str:
    start = markdown.index("```python\n") + len("```python\n")
    end = markdown.rindex("\n```")
    return markdown[start:end]
