#!/usr/bin/env python3
"""Retired placeholder for the former PMC pathology downloader.

The previous implementation guessed figure URLs and generated placeholder
captions instead of parsing each article's Open Access package and licence.
That behavior could silently create incomplete or incorrectly attributed
training data, so it is intentionally disabled during repository cleanup.
"""

from __future__ import annotations

import sys


MESSAGE = """The legacy PMC downloader has been retired.

Use the official PubMed Central Open Access package list and parse each
article's XML/archive metadata, including its per-article licence, before
adding images or captions to a research dataset. A replacement must include
manifest hashes, source URLs, licence fields, extraction tests, and failure
reporting before it is enabled in this repository.
"""


def main() -> int:
    """Explain why the unsafe legacy downloader is unavailable."""
    print(MESSAGE, file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
