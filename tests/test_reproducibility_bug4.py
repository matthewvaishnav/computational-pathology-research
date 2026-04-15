"""
Bug 4 exploration test: CITATION.cff and pyproject.toml metadata inconsistency.

CRITICAL: This test MUST FAIL on unfixed code.
Expected failure: Metadata fields (title/description, authors) are inconsistent.

**Validates: Requirements 1.7, 1.8**
"""

import sys
import unittest
from pathlib import Path

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

try:
    import yaml
except ImportError:
    import subprocess

    subprocess.check_call([sys.executable, "-m", "pip", "install", "pyyaml"])
    import yaml


class TestBug4MetadataConsistency(unittest.TestCase):
    """Property 1: Bug Condition - Inconsistent Metadata."""

    def setUp(self):
        """Load both metadata files."""
        repo_root = Path(__file__).parent.parent

        # Parse CITATION.cff
        citation_path = repo_root / "CITATION.cff"
        with open(citation_path, "r", encoding="utf-8") as f:
            self.citation_data = yaml.safe_load(f)

        # Parse pyproject.toml
        pyproject_path = repo_root / "pyproject.toml"
        with open(pyproject_path, "rb") as f:
            self.pyproject_data = tomllib.load(f)

    def test_title_description_consistency(self):
        """
        Test that CITATION.cff title/abstract matches pyproject.toml description.

        EXPECTED BEHAVIOR (after fix):
        - CITATION.cff title should semantically match pyproject.toml description
        - Both should describe the same project with consistent terminology

        CURRENT BEHAVIOR (unfixed code):
        - CITATION.cff: "Computational Pathology Research Framework"
        - pyproject.toml: "Novel multimodal fusion architectures for computational pathology"
        - These are semantically different descriptions

        CRITICAL: This test MUST FAIL on unfixed code - failure confirms the bug exists.
        """
        citation_title = self.citation_data.get("title", "")
        citation_abstract = self.citation_data.get("abstract", "")
        pyproject_description = self.pyproject_data["project"]["description"]

        # Document the counterexample (current inconsistent state)
        print("\n=== Bug 4 Counterexample: Title/Description Mismatch ===")
        print(f"CITATION.cff title: '{citation_title}'")
        print(f"CITATION.cff abstract: '{citation_abstract[:100]}...'")
        print(f"pyproject.toml description: '{pyproject_description}'")
        print("=" * 60)

        # After fix, these should be semantically consistent
        # For now, we check if they contain common key terms
        # The fix should align these to describe the same project

        # Check if both mention "computational pathology"
        self.assertIn(
            "computational pathology",
            citation_title.lower() + " " + citation_abstract.lower(),
            "CITATION.cff should mention 'computational pathology'",
        )
        self.assertIn(
            "computational pathology",
            pyproject_description.lower(),
            "pyproject.toml should mention 'computational pathology'",
        )

        # Check for semantic consistency - both should describe the framework
        # After fix, the descriptions should align
        # Current state: CITATION.cff focuses on "framework", pyproject.toml on "architectures"

        # This assertion will FAIL on unfixed code (expected)
        # After fix, both should use consistent terminology
        citation_terms = set((citation_title + " " + citation_abstract).lower().split())
        pyproject_terms = set(pyproject_description.lower().split())

        # Check for key term overlap - should be high after fix
        common_terms = citation_terms & pyproject_terms

        # Expected after fix: High overlap in key descriptive terms
        # Current unfixed: Low overlap (different focus)
        self.assertGreater(
            len(common_terms),
            5,
            f"Insufficient term overlap between metadata descriptions. "
            f"Common terms: {common_terms}. "
            f"This indicates inconsistent project descriptions.",
        )

    def test_author_consistency(self):
        """
        Test that CITATION.cff authors match pyproject.toml authors.

        EXPECTED BEHAVIOR (after fix):
        - CITATION.cff authors should match pyproject.toml authors
        - Both should attribute the project to the same person/team

        CURRENT BEHAVIOR (unfixed code):
        - CITATION.cff: "Matthew Vaishnav"
        - pyproject.toml: "Research Team"
        - These are different attributions

        CRITICAL: This test MUST FAIL on unfixed code - failure confirms the bug exists.
        """
        citation_authors = self.citation_data.get("authors", [])
        pyproject_authors = self.pyproject_data["project"]["authors"]

        # Extract author names
        citation_author_names = []
        for author in citation_authors:
            if "family-names" in author and "given-names" in author:
                full_name = f"{author['given-names']} {author['family-names']}"
                citation_author_names.append(full_name)
            elif "name" in author:
                citation_author_names.append(author["name"])

        pyproject_author_names = [author.get("name", "") for author in pyproject_authors]

        # Document the counterexample (current inconsistent state)
        print("\n=== Bug 4 Counterexample: Author Mismatch ===")
        print(f"CITATION.cff authors: {citation_author_names}")
        print(f"pyproject.toml authors: {pyproject_author_names}")
        print("=" * 60)

        # After fix, authors should match
        # This assertion will FAIL on unfixed code (expected)
        self.assertEqual(
            len(citation_author_names),
            len(pyproject_author_names),
            f"Author count mismatch: CITATION.cff has {len(citation_author_names)} authors, "
            f"pyproject.toml has {len(pyproject_author_names)} authors",
        )

        # Check if author names match
        for citation_name in citation_author_names:
            self.assertIn(
                citation_name,
                pyproject_author_names,
                f"Author '{citation_name}' from CITATION.cff not found in pyproject.toml authors",
            )

    def test_metadata_semantic_identity(self):
        """
        Test that both files represent the same project identity.

        EXPECTED BEHAVIOR (after fix):
        - Both files should describe the same project
        - Key metadata should be consistent (name, version, license)

        CURRENT BEHAVIOR (unfixed code):
        - Inconsistent descriptions create ambiguity about project identity
        - Different author attributions create trust issues

        CRITICAL: This test MUST FAIL on unfixed code - failure confirms the bug exists.
        """
        # Check version consistency
        citation_version = self.citation_data.get("version", "")
        pyproject_version = self.pyproject_data["project"]["version"]

        self.assertEqual(
            citation_version,
            pyproject_version,
            f"Version mismatch: CITATION.cff={citation_version}, pyproject.toml={pyproject_version}",
        )

        # Check license consistency
        citation_license = self.citation_data.get("license", "")
        pyproject_license = self.pyproject_data["project"]["license"]["text"]

        self.assertEqual(
            citation_license,
            pyproject_license,
            f"License mismatch: CITATION.cff={citation_license}, pyproject.toml={pyproject_license}",
        )

        # Document the overall inconsistency
        print("\n=== Bug 4 Summary: Metadata Inconsistency ===")
        print("CITATION.cff represents:")
        print(f"  Title: {self.citation_data.get('title', '')}")
        citation_authors_list = [
            f"{a.get('given-names', '')} {a.get('family-names', '')}"
            for a in self.citation_data.get("authors", [])
        ]
        print(f"  Authors: {citation_authors_list}")
        print(f"  Version: {citation_version}")
        print(f"  License: {citation_license}")
        print("\npyproject.toml represents:")
        print(f"  Description: {self.pyproject_data['project']['description']}")
        pyproject_authors_list = [
            a.get("name", "") for a in self.pyproject_data["project"]["authors"]
        ]
        print(f"  Authors: {pyproject_authors_list}")
        print(f"  Version: {pyproject_version}")
        print(f"  License: {pyproject_license}")
        print("\nINCONSISTENCY: Title/description and authors differ between files.")
        print("This creates trust issues for citation and package management.")
        print("=" * 60)


if __name__ == "__main__":
    unittest.main()
