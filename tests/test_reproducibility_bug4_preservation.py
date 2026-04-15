"""
Bug 4 preservation tests: Verify non-buggy metadata fields remain valid.

IMPORTANT: These tests MUST PASS on unfixed code.
Captures baseline behavior to preserve after fix.

**Validates: Requirements 3.10, 3.11, 3.12**
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


class TestBug4Preservation(unittest.TestCase):
    """Property 2: Preservation - Other Metadata Fields."""

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

    def test_citation_cff_format_validity(self):
        """
        Test that CITATION.cff is valid CFF format for bibliographic tools.

        PRESERVATION: CITATION.cff must remain valid CFF 1.2.0 format.
        This ensures citation tools can parse the file correctly.

        **Validates: Requirements 3.10**
        """
        # Verify required CFF fields
        self.assertIn("cff-version", self.citation_data, "Missing cff-version field")
        self.assertEqual(self.citation_data["cff-version"], "1.2.0", "CFF version should be 1.2.0")

        self.assertIn("message", self.citation_data, "Missing message field")
        self.assertIsInstance(self.citation_data["message"], str, "Message should be a string")

        self.assertIn("type", self.citation_data, "Missing type field")
        self.assertEqual(self.citation_data["type"], "software", "Type should be 'software'")

        self.assertIn("title", self.citation_data, "Missing title field")
        self.assertIsInstance(self.citation_data["title"], str, "Title should be a string")

        self.assertIn("authors", self.citation_data, "Missing authors field")
        self.assertIsInstance(self.citation_data["authors"], list, "Authors should be a list")
        self.assertGreater(
            len(self.citation_data["authors"]), 0, "Authors list should not be empty"
        )

        # Verify author structure
        for author in self.citation_data["authors"]:
            self.assertIsInstance(author, dict, "Each author should be a dictionary")
            # CFF allows either family-names/given-names OR name
            has_family_given = "family-names" in author and "given-names" in author
            has_name = "name" in author
            self.assertTrue(
                has_family_given or has_name,
                "Author must have either family-names/given-names or name",
            )

    def test_pyproject_toml_format_validity(self):
        """
        Test that pyproject.toml is valid TOML format for package managers.

        PRESERVATION: pyproject.toml must remain valid TOML format.
        This ensures pip, build tools, and package managers can parse it.

        **Validates: Requirements 3.11**
        """
        # Verify required pyproject.toml sections
        self.assertIn("build-system", self.pyproject_data, "Missing [build-system] section")
        self.assertIn("project", self.pyproject_data, "Missing [project] section")

        # Verify build-system structure
        build_system = self.pyproject_data["build-system"]
        self.assertIn("requires", build_system, "Missing build-system.requires")
        self.assertIn("build-backend", build_system, "Missing build-system.build-backend")
        self.assertIsInstance(build_system["requires"], list, "requires should be a list")
        self.assertIsInstance(
            build_system["build-backend"], str, "build-backend should be a string"
        )

        # Verify project structure
        project = self.pyproject_data["project"]
        required_fields = ["name", "version", "description", "authors"]
        for field in required_fields:
            self.assertIn(field, project, f"Missing project.{field}")

        self.assertIsInstance(project["name"], str, "project.name should be a string")
        self.assertIsInstance(project["version"], str, "project.version should be a string")
        self.assertIsInstance(project["description"], str, "project.description should be a string")
        self.assertIsInstance(project["authors"], list, "project.authors should be a list")
        self.assertGreater(len(project["authors"]), 0, "project.authors should not be empty")

    def test_camelyon_dataset_citations_preserved(self):
        """
        Test that CAMELYON dataset citations are preserved and correctly formatted.

        PRESERVATION: Dataset citations in CITATION.cff must remain intact.
        These citations credit the original CAMELYON dataset authors.

        **Validates: Requirements 3.12**
        """
        self.assertIn("references", self.citation_data, "Missing references field")
        references = self.citation_data["references"]
        self.assertIsInstance(references, list, "References should be a list")
        self.assertGreaterEqual(
            len(references), 2, "Should have at least 2 CAMELYON dataset citations"
        )

        # Verify first CAMELYON citation (GigaScience paper)
        gigascience_ref = references[0]
        self.assertEqual(gigascience_ref["type"], "article", "First reference should be article")
        self.assertIn(
            "CAMELYON", gigascience_ref["title"], "First reference should mention CAMELYON"
        )
        self.assertIn("authors", gigascience_ref, "Reference should have authors")
        self.assertIn("journal", gigascience_ref, "Reference should have journal")
        self.assertEqual(
            gigascience_ref["journal"], "GigaScience", "First reference should be from GigaScience"
        )
        self.assertIn("doi", gigascience_ref, "Reference should have DOI")
        self.assertEqual(
            gigascience_ref["doi"],
            "10.1093/gigascience/giy065",
            "First reference DOI should be correct",
        )

        # Verify second CAMELYON citation (JAMA paper)
        jama_ref = references[1]
        self.assertEqual(jama_ref["type"], "article", "Second reference should be article")
        self.assertIn("authors", jama_ref, "Reference should have authors")
        self.assertIn("journal", jama_ref, "Reference should have journal")
        self.assertEqual(jama_ref["journal"], "JAMA", "Second reference should be from JAMA")
        self.assertIn("doi", jama_ref, "Reference should have DOI")
        self.assertEqual(
            jama_ref["doi"], "10.1001/jama.2017.14585", "Second reference DOI should be correct"
        )

        # Verify author structure in references
        for ref in references:
            self.assertIn("authors", ref, "Each reference should have authors")
            authors = ref["authors"]
            self.assertIsInstance(authors, list, "Reference authors should be a list")
            self.assertGreater(len(authors), 0, "Reference authors should not be empty")

            # Verify first author is Ehteshami Bejnordi
            first_author = authors[0]
            self.assertIn("family-names", first_author, "Author should have family-names")
            self.assertEqual(
                first_author["family-names"],
                "Ehteshami Bejnordi",
                "First author should be Ehteshami Bejnordi",
            )

    def test_license_field_preserved(self):
        """
        Test that license field is preserved in both files.

        PRESERVATION: License information must remain valid.

        **Validates: Requirements 3.10, 3.11**
        """
        # Verify CITATION.cff license
        self.assertIn("license", self.citation_data, "CITATION.cff missing license")
        self.assertEqual(self.citation_data["license"], "MIT", "CITATION.cff license should be MIT")

        # Verify pyproject.toml license
        project = self.pyproject_data["project"]
        self.assertIn("license", project, "pyproject.toml missing license")
        self.assertIn("text", project["license"], "pyproject.toml license missing text")
        self.assertEqual(project["license"]["text"], "MIT", "pyproject.toml license should be MIT")

    def test_version_field_preserved(self):
        """
        Test that version field is preserved in both files.

        PRESERVATION: Version information must remain valid.

        **Validates: Requirements 3.10, 3.11**
        """
        # Verify CITATION.cff version
        self.assertIn("version", self.citation_data, "CITATION.cff missing version")
        self.assertEqual(
            self.citation_data["version"], "0.1.0", "CITATION.cff version should be 0.1.0"
        )

        # Verify pyproject.toml version
        project = self.pyproject_data["project"]
        self.assertIn("version", project, "pyproject.toml missing version")
        self.assertEqual(project["version"], "0.1.0", "pyproject.toml version should be 0.1.0")

    def test_keywords_field_preserved(self):
        """
        Test that keywords field is preserved in both files.

        PRESERVATION: Keywords must remain valid for discoverability.

        **Validates: Requirements 3.10, 3.11**
        """
        # Verify CITATION.cff keywords
        self.assertIn("keywords", self.citation_data, "CITATION.cff missing keywords")
        citation_keywords = self.citation_data["keywords"]
        self.assertIsInstance(citation_keywords, list, "CITATION.cff keywords should be a list")
        self.assertGreater(len(citation_keywords), 0, "CITATION.cff keywords should not be empty")

        # Verify expected keywords are present
        expected_keywords = [
            "computational pathology",
            "whole-slide imaging",
            "multiple instance learning",
            "deep learning",
            "histopathology",
            "pytorch",
        ]
        for keyword in expected_keywords:
            self.assertIn(keyword, citation_keywords, f"CITATION.cff missing keyword: {keyword}")

        # Verify pyproject.toml keywords
        project = self.pyproject_data["project"]
        self.assertIn("keywords", project, "pyproject.toml missing keywords")
        pyproject_keywords = project["keywords"]
        self.assertIsInstance(pyproject_keywords, list, "pyproject.toml keywords should be a list")
        self.assertGreater(
            len(pyproject_keywords), 0, "pyproject.toml keywords should not be empty"
        )

        # Verify computational pathology is mentioned
        self.assertIn(
            "computational pathology",
            pyproject_keywords,
            "pyproject.toml should have 'computational pathology' keyword",
        )

    def test_repository_urls_preserved(self):
        """
        Test that repository URLs are preserved in CITATION.cff.

        PRESERVATION: Repository URLs must remain valid.

        **Validates: Requirements 3.10**
        """
        self.assertIn("repository-code", self.citation_data, "Missing repository-code")
        self.assertIn("url", self.citation_data, "Missing url")

        expected_repo = "https://github.com/matthewvaishnav/computational-pathology-research"
        self.assertEqual(
            self.citation_data["repository-code"],
            expected_repo,
            "repository-code should be correct",
        )
        self.assertEqual(self.citation_data["url"], expected_repo, "url should be correct")

    def test_preferred_citation_preserved(self):
        """
        Test that preferred-citation is preserved in CITATION.cff.

        PRESERVATION: Preferred citation structure must remain valid.

        **Validates: Requirements 3.10**
        """
        self.assertIn("preferred-citation", self.citation_data, "Missing preferred-citation")
        preferred = self.citation_data["preferred-citation"]

        self.assertIsInstance(preferred, dict, "preferred-citation should be a dictionary")
        self.assertEqual(
            preferred["type"], "software", "preferred-citation type should be software"
        )
        self.assertIn("title", preferred, "preferred-citation missing title")
        self.assertIn("authors", preferred, "preferred-citation missing authors")
        self.assertIn("year", preferred, "preferred-citation missing year")
        self.assertIn("version", preferred, "preferred-citation missing version")
        self.assertIn("url", preferred, "preferred-citation missing url")

    def test_date_released_preserved(self):
        """
        Test that date-released is preserved in CITATION.cff.

        PRESERVATION: Release date must remain valid.

        **Validates: Requirements 3.10**
        """
        self.assertIn("date-released", self.citation_data, "Missing date-released")
        # Date format should be YYYY-M-D or YYYY-MM-DD
        date_released = str(self.citation_data["date-released"])
        self.assertRegex(
            date_released, r"^\d{4}-\d{1,2}-\d{1,2}$", "date-released should be in YYYY-M-D format"
        )

    def test_abstract_field_preserved(self):
        """
        Test that abstract field is preserved in CITATION.cff.

        PRESERVATION: Abstract must remain valid and descriptive.

        **Validates: Requirements 3.10**
        """
        self.assertIn("abstract", self.citation_data, "Missing abstract")
        abstract = self.citation_data["abstract"]
        self.assertIsInstance(abstract, str, "abstract should be a string")
        self.assertGreater(len(abstract), 50, "abstract should be descriptive (>50 chars)")

        # Verify abstract mentions key framework features
        abstract_lower = abstract.lower()
        self.assertIn("pytorch", abstract_lower, "abstract should mention PyTorch")
        self.assertIn(
            "computational pathology",
            abstract_lower,
            "abstract should mention computational pathology",
        )

    def test_pyproject_dependencies_preserved(self):
        """
        Test that dependencies are preserved in pyproject.toml.

        PRESERVATION: Dependencies list must remain valid.

        **Validates: Requirements 3.11**
        """
        project = self.pyproject_data["project"]
        self.assertIn("dependencies", project, "Missing dependencies")
        dependencies = project["dependencies"]

        self.assertIsInstance(dependencies, list, "dependencies should be a list")
        self.assertGreater(len(dependencies), 0, "dependencies should not be empty")

        # Verify key dependencies are present
        key_deps = ["torch", "torchvision", "pytest", "numpy", "pandas"]
        for dep in key_deps:
            dep_found = any(dep in d for d in dependencies)
            self.assertTrue(dep_found, f"Missing key dependency: {dep}")

    def test_pyproject_classifiers_preserved(self):
        """
        Test that classifiers are preserved in pyproject.toml.

        PRESERVATION: PyPI classifiers must remain valid.

        **Validates: Requirements 3.11**
        """
        project = self.pyproject_data["project"]
        self.assertIn("classifiers", project, "Missing classifiers")
        classifiers = project["classifiers"]

        self.assertIsInstance(classifiers, list, "classifiers should be a list")
        self.assertGreater(len(classifiers), 0, "classifiers should not be empty")

        # Verify expected classifiers
        expected_classifiers = [
            "Development Status :: 3 - Alpha",
            "License :: OSI Approved :: MIT License",
            "Programming Language :: Python :: 3",
        ]
        for classifier in expected_classifiers:
            self.assertIn(classifier, classifiers, f"Missing classifier: {classifier}")


if __name__ == "__main__":
    unittest.main()
