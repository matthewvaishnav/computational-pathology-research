#!/usr/bin/env python3
"""Fail when loose documentation or setup scripts are added to the repo root."""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]

ALLOWED_ROOT_MARKDOWN = {
    "README.md",
    "CHANGELOG.md",
    "CLAIM_BOUNDARY.md",
    "CONTRIBUTING.md",
    "SECURITY.md",
    "CODE_OF_CONDUCT.md",
}

FORBIDDEN_ROOT_SCRIPT_SUFFIXES = {".bat", ".cmd", ".ps1", ".sh"}


def main() -> None:
    unexpected_markdown = sorted(
        path.name
        for path in ROOT.glob("*.md")
        if path.name not in ALLOWED_ROOT_MARKDOWN
    )
    unexpected_scripts = sorted(
        path.name
        for path in ROOT.iterdir()
        if path.is_file() and path.suffix.lower() in FORBIDDEN_ROOT_SCRIPT_SUFFIXES
    )

    problems: list[str] = []
    if unexpected_markdown:
        problems.append(
            "Move root Markdown into docs/: " + ", ".join(unexpected_markdown)
        )
    if unexpected_scripts:
        problems.append(
            "Move root setup/launcher scripts into scripts/: "
            + ", ".join(unexpected_scripts)
        )

    if problems:
        raise SystemExit("ROOT LAYOUT CHECK FAILED\n- " + "\n- ".join(problems))

    print("ROOT LAYOUT CHECK PASSED")


if __name__ == "__main__":
    main()
