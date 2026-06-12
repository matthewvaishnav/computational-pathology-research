"""Collection rules for model tests with optional historical fixtures."""

from pathlib import Path

import pytest


_ATTENTION_BACKUP = (
    Path(__file__).resolve().parents[2] / "src" / "models" / "attention_mil.py.backup"
)

# The property-equivalence suite compares against a historical backup implementation.
# Keep the suite available when that fixture is restored, but do not let its absence
# break collection of unrelated core model tests.
collect_ignore = []
if not _ATTENTION_BACKUP.exists():
    collect_ignore.append("test_attention_mil_properties.py")


def pytest_collection_modifyitems(items):
    """Classify the historical equivalence suite as property-based tests."""
    property_mark = pytest.mark.property
    for item in items:
        if item.path.name == "test_attention_mil_properties.py":
            item.add_marker(property_mark)
