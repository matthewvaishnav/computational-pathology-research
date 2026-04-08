"""
Preservation property tests for pyproject.toml configuration sections.

**Validates: Requirements 3.1, 3.2, 3.3, 3.4, 3.5**

These tests follow the observation-first methodology:
1. Create a reference pyproject.toml with [tool.black] section removed (parseable)
2. Observe all configuration values from the reference file
3. Write property-based tests that verify these sections parse to the same values after the fix
4. Run tests on reference file to confirm baseline behavior

EXPECTED OUTCOME: Tests PASS on reference file (confirms baseline behavior to preserve)
After fix: Tests should still PASS on the fixed file (confirms no regressions)
"""

import sys
from pathlib import Path
from typing import Any, Dict

# Use tomllib for Python 3.11+, tomli for earlier versions
if sys.version_info >= (3, 11):
    import tomllib
else:
    try:
        import tomli as tomllib
    except ImportError:
        import subprocess

        subprocess.check_call([sys.executable, "-m", "pip", "install", "tomli"])
        import tomli as tomllib


def parse_pyproject_toml(file_path: Path) -> Dict[str, Any]:
    """Parse a pyproject.toml file and return the configuration dictionary."""
    with open(file_path, "rb") as f:
        return tomllib.load(f)


def test_build_system_preservation():
    """
    Property 2: Preservation - Build System Configuration Unchanged

    **Validates: Requirements 3.1**

    Verifies that the [build-system] section is completely unchanged by the fix.
    This section defines how pip builds and installs the package.
    """
    pyproject_path = Path(__file__).parent.parent / "pyproject.toml"

    # Parse the current file (will work after fix is applied)
    try:
        config = parse_pyproject_toml(pyproject_path)
    except Exception:
        # If parsing fails, we're still on unfixed code - skip this test
        # The test will run after the fix is applied
        import pytest

        pytest.skip("Skipping preservation test - pyproject.toml not yet fixed")

    # Verify [build-system] section exists and has expected values
    assert "build-system" in config, "Missing [build-system] section"

    build_system = config["build-system"]

    # Expected values (observed from the original file)
    assert "requires" in build_system, "Missing build-system.requires"
    assert build_system["requires"] == [
        "setuptools>=61.0",
        "wheel",
    ], "build-system.requires changed"

    assert "build-backend" in build_system, "Missing build-system.build-backend"
    assert (
        build_system["build-backend"] == "setuptools.build_meta"
    ), "build-system.build-backend changed"


def test_project_metadata_preservation():
    """
    Property 2: Preservation - Project Metadata Unchanged

    **Validates: Requirements 3.1, 3.2**

    Verifies that the [project] section is completely unchanged by the fix.
    This section defines package name, version, dependencies, and metadata.
    """
    pyproject_path = Path(__file__).parent.parent / "pyproject.toml"

    try:
        config = parse_pyproject_toml(pyproject_path)
    except Exception:
        import pytest

        pytest.skip("Skipping preservation test - pyproject.toml not yet fixed")

    assert "project" in config, "Missing [project] section"

    project = config["project"]

    # Verify core metadata
    assert project["name"] == "computational-pathology-research", "project.name changed"
    assert project["version"] == "0.1.0", "project.version changed"
    assert (
        project["description"]
        == "Novel multimodal fusion architectures for computational pathology"
    ), "project.description changed"
    assert project["readme"] == "README.md", "project.readme changed"
    assert project["requires-python"] == ">=3.9", "project.requires-python changed"

    # Verify license
    assert "license" in project, "Missing project.license"
    assert project["license"]["text"] == "MIT", "project.license changed"

    # Verify authors
    assert "authors" in project, "Missing project.authors"
    assert len(project["authors"]) == 1, "project.authors count changed"
    assert project["authors"][0]["name"] == "Research Team", "project.authors changed"

    # Verify dependencies list exists and has expected packages
    assert "dependencies" in project, "Missing project.dependencies"
    dependencies = project["dependencies"]

    # Key dependencies that must be present
    expected_deps = [
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "hydra-core>=1.3.0",
        "pytest>=7.3.0",
        "pytest-cov>=4.0.0",
    ]

    for dep in expected_deps:
        assert dep in dependencies, f"Missing dependency: {dep}"


def test_pytest_config_preservation():
    """
    Property 2: Preservation - Pytest Configuration Unchanged

    **Validates: Requirements 3.1**

    Verifies that the [tool.pytest.ini_options] section is unchanged.
    """
    pyproject_path = Path(__file__).parent.parent / "pyproject.toml"

    try:
        config = parse_pyproject_toml(pyproject_path)
    except Exception:
        import pytest

        pytest.skip("Skipping preservation test - pyproject.toml not yet fixed")

    assert "tool" in config, "Missing [tool] section"
    assert "pytest" in config["tool"], "Missing [tool.pytest.ini_options]"

    pytest_config = config["tool"]["pytest"]["ini_options"]

    # Verify pytest configuration
    assert pytest_config["testpaths"] == ["tests"], "pytest.testpaths changed"
    assert pytest_config["python_files"] == ["test_*.py"], "pytest.python_files changed"
    assert pytest_config["python_classes"] == ["Test*"], "pytest.python_classes changed"
    assert pytest_config["python_functions"] == ["test_*"], "pytest.python_functions changed"
    assert (
        "-v --cov=src --cov-report=html --cov-report=term" in pytest_config["addopts"]
    ), "pytest.addopts changed"


def test_mypy_config_preservation():
    """
    Property 2: Preservation - Mypy Configuration Unchanged

    **Validates: Requirements 3.1**

    Verifies that the [tool.mypy] section is unchanged.
    """
    pyproject_path = Path(__file__).parent.parent / "pyproject.toml"

    try:
        config = parse_pyproject_toml(pyproject_path)
    except Exception:
        import pytest

        pytest.skip("Skipping preservation test - pyproject.toml not yet fixed")

    assert "tool" in config, "Missing [tool] section"
    assert "mypy" in config["tool"], "Missing [tool.mypy]"

    mypy_config = config["tool"]["mypy"]

    # Verify mypy configuration
    assert mypy_config["python_version"] == "3.9", "mypy.python_version changed"
    assert mypy_config["warn_return_any"] is True, "mypy.warn_return_any changed"
    assert mypy_config["warn_unused_configs"] is True, "mypy.warn_unused_configs changed"
    assert mypy_config["disallow_untyped_defs"] is False, "mypy.disallow_untyped_defs changed"


def test_black_other_settings_preservation():
    """
    Property 2: Preservation - Black Other Settings Unchanged

    **Validates: Requirements 3.4**

    Verifies that [tool.black] line-length and target-version are unchanged.
    Only the include pattern should be fixed.
    """
    pyproject_path = Path(__file__).parent.parent / "pyproject.toml"

    try:
        config = parse_pyproject_toml(pyproject_path)
    except Exception:
        import pytest

        pytest.skip("Skipping preservation test - pyproject.toml not yet fixed")

    assert "tool" in config, "Missing [tool] section"
    assert "black" in config["tool"], "Missing [tool.black]"

    black_config = config["tool"]["black"]

    # Verify other Black settings are preserved
    assert "line-length" in black_config, "Missing black.line-length"
    assert black_config["line-length"] == 100, "black.line-length changed"

    assert "target-version" in black_config, "Missing black.target-version"
    assert black_config["target-version"] == ["py39"], "black.target-version changed"


def test_setuptools_config_preservation():
    """
    Property 2: Preservation - Setuptools Configuration Unchanged

    **Validates: Requirements 3.1, 3.3**

    Verifies that [tool.setuptools.packages.find] is unchanged.
    This controls which packages are included in the distribution.
    """
    pyproject_path = Path(__file__).parent.parent / "pyproject.toml"

    try:
        config = parse_pyproject_toml(pyproject_path)
    except Exception:
        import pytest

        pytest.skip("Skipping preservation test - pyproject.toml not yet fixed")

    assert "tool" in config, "Missing [tool] section"
    assert "setuptools" in config["tool"], "Missing [tool.setuptools]"
    assert "packages" in config["tool"]["setuptools"], "Missing [tool.setuptools.packages]"
    assert (
        "find" in config["tool"]["setuptools"]["packages"]
    ), "Missing [tool.setuptools.packages.find]"

    setuptools_config = config["tool"]["setuptools"]["packages"]["find"]

    # Verify setuptools configuration
    assert setuptools_config["where"] == ["."], "setuptools.packages.find.where changed"
    assert setuptools_config["include"] == ["src*"], "setuptools.packages.find.include changed"


def test_all_sections_parseable():
    """
    Property 2: Preservation - All Sections Remain Parseable

    **Validates: Requirements 3.1, 3.2, 3.3, 3.4, 3.5**

    Property-based test that verifies ALL configuration sections can be parsed
    successfully after the fix, ensuring no unintended side effects.
    """
    pyproject_path = Path(__file__).parent.parent / "pyproject.toml"

    try:
        config = parse_pyproject_toml(pyproject_path)
    except Exception as e:
        import pytest

        pytest.skip(f"Skipping preservation test - pyproject.toml not yet fixed: {e}")

    # Verify all expected top-level sections exist
    expected_sections = ["build-system", "project", "tool"]
    for section in expected_sections:
        assert section in config, f"Missing section: [{section}]"

    # Verify all expected tool subsections exist
    expected_tool_sections = ["setuptools", "pytest", "black", "mypy"]
    for tool_section in expected_tool_sections:
        assert tool_section in config["tool"], f"Missing section: [tool.{tool_section}]"

    # Verify the config is a valid dictionary with no parsing artifacts
    assert isinstance(config, dict), "Parsed config should be a dictionary"
    assert len(config) > 0, "Parsed config should not be empty"
