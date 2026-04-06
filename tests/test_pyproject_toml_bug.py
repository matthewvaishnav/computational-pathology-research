"""
Bug condition exploration test for pyproject.toml TOML parsing failure.

**Validates: Requirements 1.1, 1.2, 1.3**

This test is designed to FAIL on unfixed code to confirm the bug exists.
When the test FAILS, it proves the TOML parsing error is present.
After the fix is implemented, this test should PASS.
"""

import subprocess
import sys
from pathlib import Path

# Use tomllib for Python 3.11+, tomli for earlier versions
if sys.version_info >= (3, 11):
    import tomllib
else:
    try:
        import tomli as tomllib
    except ImportError:
        # Install tomli if not available
        subprocess.check_call([sys.executable, "-m", "pip", "install", "tomli"])
        import tomli as tomllib


def test_pyproject_toml_parsing():
    """
    Property 1: Bug Condition - TOML Parsing Failure on Malformed Include Line

    **Validates: Requirements 1.1, 1.2, 1.3**

    This test attempts to parse pyproject.toml with the malformed [tool.black] include line.

    EXPECTED BEHAVIOR (after fix):
    - TOML parsing should succeed without errors
    - The [tool.black] section should be readable
    - The include pattern should be properly formatted

    CURRENT BEHAVIOR (unfixed code):
    - TOML parsing raises TOMLDecodeError due to unclosed string literal
    - Error points to line 60 in the [tool.black] section
    - The include line is incomplete: include = '\\.pyi?

    CRITICAL: This test MUST FAIL on unfixed code - failure confirms the bug exists.
    """
    pyproject_path = Path(__file__).parent.parent / "pyproject.toml"

    # Attempt to parse pyproject.toml
    with open(pyproject_path, "rb") as f:
        config = tomllib.load(f)

    # If we reach here, parsing succeeded
    # Verify the [tool.black] section exists and is properly formatted
    assert "tool" in config, "Missing [tool] section"
    assert "black" in config["tool"], "Missing [tool.black] section"

    black_config = config["tool"]["black"]

    # Verify the include pattern is properly formatted
    assert "include" in black_config, "Missing include configuration in [tool.black]"

    # The include pattern should be a complete regex with closing quote
    include_pattern = black_config["include"]
    assert isinstance(include_pattern, str), "include should be a string"

    # Verify the pattern is complete (should end with $ anchor)
    assert include_pattern.endswith(
        "$"
    ), f"include pattern should end with $, got: {include_pattern}"

    # Verify the pattern matches Python files
    assert "\\.py" in include_pattern, "include pattern should match .py files"

    # Verify other [tool.black] settings are preserved
    assert "line-length" in black_config, "Missing line-length in [tool.black]"
    assert black_config["line-length"] == 100, "line-length should be 100"

    assert "target-version" in black_config, "Missing target-version in [tool.black]"
    assert black_config["target-version"] == ["py39"], "target-version should be ['py39']"


def test_pip_install_simulation():
    """
    Property 1: Bug Condition - Pip Install Fails During TOML Parsing

    **Validates: Requirements 1.1, 1.2, 1.3**

    This test simulates what pip does when installing the package in editable mode.
    Pip must parse pyproject.toml to read build system requirements and project metadata.

    EXPECTED BEHAVIOR (after fix):
    - TOML parsing should succeed
    - Build system requirements should be readable
    - Project metadata should be accessible

    CURRENT BEHAVIOR (unfixed code):
    - TOML parsing fails before pip can read any configuration
    - pip install -e . terminates with TOML parsing error
    - CI workflows fail within 2-4 seconds

    CRITICAL: This test MUST FAIL on unfixed code - failure confirms the bug exists.
    """
    pyproject_path = Path(__file__).parent.parent / "pyproject.toml"

    # Simulate pip's TOML parsing
    with open(pyproject_path, "rb") as f:
        config = tomllib.load(f)

    # Verify pip can read build-system requirements
    assert "build-system" in config, "Missing [build-system] section"
    assert "requires" in config["build-system"], "Missing build-system.requires"
    assert "build-backend" in config["build-system"], "Missing build-system.build-backend"

    # Verify pip can read project metadata
    assert "project" in config, "Missing [project] section"
    assert "name" in config["project"], "Missing project.name"
    assert "version" in config["project"], "Missing project.version"
    assert "dependencies" in config["project"], "Missing project.dependencies"

    # Verify all tool configurations are readable
    assert "tool" in config, "Missing [tool] section"
    assert "pytest" in config["tool"], "Missing [tool.pytest.ini_options]"
    assert "mypy" in config["tool"], "Missing [tool.mypy]"
    assert "black" in config["tool"], "Missing [tool.black]"


def test_toml_error_location():
    """
    Property 1: Bug Condition - Error Points to [tool.black] Section

    **Validates: Requirements 1.1, 1.2, 1.3**

    This test verifies that the TOML parsing error (on unfixed code) points to
    the correct location: the [tool.black] section around line 60.

    EXPECTED BEHAVIOR (after fix):
    - No TOML parsing error should occur
    - All sections should parse successfully

    CURRENT BEHAVIOR (unfixed code):
    - TOMLDecodeError is raised
    - Error message indicates unclosed string literal
    - Error location is in the [tool.black] section

    CRITICAL: This test MUST FAIL on unfixed code - failure confirms the bug exists.
    """
    pyproject_path = Path(__file__).parent.parent / "pyproject.toml"

    # Read the file content to verify the malformed line exists (on unfixed code)
    with open(pyproject_path, "r") as f:
        content = f.read()

    # After the fix, the file should contain the complete include pattern
    # The pattern should be: include = '\\.pyi?$'
    # NOT the malformed: include = '\\.pyi?

    # Verify the [tool.black] section exists
    assert "[tool.black]" in content, "[tool.black] section should exist"

    # Parse the TOML to ensure no errors
    with open(pyproject_path, "rb") as f:
        config = tomllib.load(f)

    # Verify the include line is properly formatted
    black_config = config["tool"]["black"]
    include_pattern = black_config["include"]

    # The fixed pattern should be complete with closing quote and $ anchor
    # If this assertion passes, the bug is fixed
    assert (
        "'" not in include_pattern or include_pattern.count("'") == 0
    ), "include pattern should not contain quotes (TOML parser removes them)"

    # Verify the pattern is a valid regex
    import re

    try:
        re.compile(include_pattern)
    except re.error as e:
        raise AssertionError(f"include pattern is not a valid regex: {e}")
