"""
Unit tests for disease taxonomy configuration system.
"""

import json
import tempfile
from pathlib import Path

import pytest
import yaml

from src.clinical.taxonomy import DiseaseTaxonomy


class TestDiseaseTaxonomy:
    """Test cases for DiseaseTaxonomy class."""

    @pytest.fixture
    def simple_taxonomy_dict(self):
        """Simple taxonomy for testing."""
        return {
            "name": "Test Taxonomy",
            "version": "1.0",
            "description": "Simple test taxonomy",
            "diseases": [
                {
                    "id": "root1",
                    "name": "Root Disease 1",
                    "description": "First root disease",
                    "parent": None,
                    "children": ["child1", "child2"],
                },
                {
                    "id": "root2",
                    "name": "Root Disease 2",
                    "description": "Second root disease",
                    "parent": None,
                    "children": [],
                },
                {
                    "id": "child1",
                    "name": "Child Disease 1",
                    "description": "First child disease",
                    "parent": "root1",
                    "children": ["grandchild1"],
                },
                {
                    "id": "child2",
                    "name": "Child Disease 2",
                    "description": "Second child disease",
                    "parent": "root1",
                    "children": [],
                },
                {
                    "id": "grandchild1",
                    "name": "Grandchild Disease 1",
                    "description": "First grandchild disease",
                    "parent": "child1",
                    "children": [],
                },
            ],
        }

    @pytest.fixture
    def invalid_taxonomy_dict(self):
        """Invalid taxonomy for testing validation."""
        return {
            "name": "Invalid Taxonomy",
            "version": "1.0",
            "diseases": [
                {
                    "id": "disease1",
                    "name": "Disease 1",
                    "parent": "nonexistent",  # Invalid parent
                    "children": [],
                }
            ],
        }

    def test_init_with_dict(self, simple_taxonomy_dict):
        """Test initialization with dictionary."""
        taxonomy = DiseaseTaxonomy(config_dict=simple_taxonomy_dict)

        assert taxonomy.name == "Test Taxonomy"
        assert taxonomy.version == "1.0"
        assert taxonomy.description == "Simple test taxonomy"
        assert len(taxonomy.diseases) == 5
        assert len(taxonomy.root_diseases) == 2
        assert len(taxonomy.leaf_diseases) == 3

    def test_init_with_yaml_file(self, simple_taxonomy_dict):
        """Test initialization with YAML file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(simple_taxonomy_dict, f)
            yaml_path = f.name

        try:
            taxonomy = DiseaseTaxonomy(config_path=yaml_path)
            assert taxonomy.name == "Test Taxonomy"
            assert len(taxonomy.diseases) == 5
        finally:
            Path(yaml_path).unlink()

    def test_init_with_json_file(self, simple_taxonomy_dict):
        """Test initialization with JSON file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(simple_taxonomy_dict, f)
            json_path = f.name

        try:
            taxonomy = DiseaseTaxonomy(config_path=json_path)
            assert taxonomy.name == "Test Taxonomy"
            assert len(taxonomy.diseases) == 5
        finally:
            Path(json_path).unlink()

    def test_init_no_config(self):
        """Test initialization without configuration."""
        with pytest.raises(ValueError, match="Either config_path or config_dict must be provided"):
            DiseaseTaxonomy()

    def test_init_both_configs(self, simple_taxonomy_dict):
        """Test initialization with both configurations."""
        with pytest.raises(
            ValueError, match="Only one of config_path or config_dict should be provided"
        ):
            DiseaseTaxonomy(config_path="dummy.yaml", config_dict=simple_taxonomy_dict)

    def test_init_nonexistent_file(self):
        """Test initialization with nonexistent file."""
        with pytest.raises(FileNotFoundError):
            DiseaseTaxonomy(config_path="nonexistent.yaml")

    def test_init_unsupported_format(self, simple_taxonomy_dict):
        """Test initialization with unsupported file format."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("dummy content")
            txt_path = f.name

        try:
            with pytest.raises(ValueError, match="Unsupported file format"):
                DiseaseTaxonomy(config_path=txt_path)
        finally:
            Path(txt_path).unlink()

    def test_validation_invalid_parent(self, invalid_taxonomy_dict):
        """Test validation with invalid parent reference."""
        with pytest.raises(ValueError, match="references non-existent parent"):
            DiseaseTaxonomy(config_dict=invalid_taxonomy_dict)

    def test_validation_invalid_child(self):
        """Test validation with invalid child reference."""
        invalid_dict = {
            "name": "Invalid Taxonomy",
            "diseases": [
                {
                    "id": "disease1",
                    "name": "Disease 1",
                    "parent": None,
                    "children": ["nonexistent"],  # Invalid child
                }
            ],
        }

        with pytest.raises(ValueError, match="references non-existent child"):
            DiseaseTaxonomy(config_dict=invalid_dict)

    def test_validation_inconsistent_relationship(self):
        """Test validation with inconsistent parent-child relationship."""
        invalid_dict = {
            "name": "Invalid Taxonomy",
            "diseases": [
                {"id": "parent", "name": "Parent", "parent": None, "children": ["child"]},
                {
                    "id": "child",
                    "name": "Child",
                    "parent": "other_parent",  # Inconsistent parent
                    "children": [],
                },
            ],
        }

        with pytest.raises(ValueError, match="Inconsistent parent-child relationship"):
            DiseaseTaxonomy(config_dict=invalid_dict)

    def test_validation_cycle(self):
        """Test validation with inconsistent parent-child relationship that would create a cycle."""
        # Note: The validation catches inconsistent relationships before cycle detection
        # This is correct behavior - a child listing a parent as its child is inconsistent
        invalid_dict = {
            "name": "Invalid Taxonomy",
            "diseases": [
                {"id": "disease1", "name": "Disease 1", "parent": None, "children": ["disease2"]},
                {
                    "id": "disease2",
                    "name": "Disease 2",
                    "parent": "disease1",
                    "children": ["disease3"],
                },
                {
                    "id": "disease3",
                    "name": "Disease 3",
                    "parent": "disease2",
                    "children": [
                        "disease1"
                    ],  # Inconsistent: disease1 doesn't have disease3 as parent
                },
            ],
        }

        with pytest.raises(ValueError, match="Inconsistent parent-child relationship"):
            DiseaseTaxonomy(config_dict=invalid_dict)

    def test_validation_empty_taxonomy(self):
        """Test validation with empty taxonomy."""
        invalid_dict = {"name": "Empty Taxonomy", "diseases": []}

        with pytest.raises(ValueError, match="contains no diseases"):
            DiseaseTaxonomy(config_dict=invalid_dict)

    def test_get_disease(self, simple_taxonomy_dict):
        """Test getting disease information."""
        taxonomy = DiseaseTaxonomy(config_dict=simple_taxonomy_dict)

        disease = taxonomy.get_disease("root1")
        assert disease is not None
        assert disease["name"] == "Root Disease 1"
        assert disease["parent"] is None
        assert disease["children"] == ["child1", "child2"]

        # Test nonexistent disease
        assert taxonomy.get_disease("nonexistent") is None

    def test_get_children(self, simple_taxonomy_dict):
        """Test getting children of a disease."""
        taxonomy = DiseaseTaxonomy(config_dict=simple_taxonomy_dict)

        children = taxonomy.get_children("root1")
        assert children == ["child1", "child2"]

        children = taxonomy.get_children("child2")
        assert children == []

        children = taxonomy.get_children("nonexistent")
        assert children == []

    def test_get_parent(self, simple_taxonomy_dict):
        """Test getting parent of a disease."""
        taxonomy = DiseaseTaxonomy(config_dict=simple_taxonomy_dict)

        parent = taxonomy.get_parent("child1")
        assert parent == "root1"

        parent = taxonomy.get_parent("root1")
        assert parent is None

        parent = taxonomy.get_parent("nonexistent")
        assert parent is None

    def test_get_ancestors(self, simple_taxonomy_dict):
        """Test getting ancestors of a disease."""
        taxonomy = DiseaseTaxonomy(config_dict=simple_taxonomy_dict)

        ancestors = taxonomy.get_ancestors("grandchild1")
        assert ancestors == ["child1", "root1"]

        ancestors = taxonomy.get_ancestors("child1")
        assert ancestors == ["root1"]

        ancestors = taxonomy.get_ancestors("root1")
        assert ancestors == []

    def test_get_descendants(self, simple_taxonomy_dict):
        """Test getting descendants of a disease."""
        taxonomy = DiseaseTaxonomy(config_dict=simple_taxonomy_dict)

        descendants = taxonomy.get_descendants("root1")
        assert set(descendants) == {"child1", "child2", "grandchild1"}

        descendants = taxonomy.get_descendants("child1")
        assert descendants == ["grandchild1"]

        descendants = taxonomy.get_descendants("child2")
        assert descendants == []

    def test_is_ancestor(self, simple_taxonomy_dict):
        """Test checking ancestor relationships."""
        taxonomy = DiseaseTaxonomy(config_dict=simple_taxonomy_dict)

        assert taxonomy.is_ancestor("root1", "grandchild1")
        assert taxonomy.is_ancestor("child1", "grandchild1")
        assert not taxonomy.is_ancestor("child2", "grandchild1")
        assert not taxonomy.is_ancestor("grandchild1", "root1")

    def test_is_descendant(self, simple_taxonomy_dict):
        """Test checking descendant relationships."""
        taxonomy = DiseaseTaxonomy(config_dict=simple_taxonomy_dict)

        assert taxonomy.is_descendant("grandchild1", "root1")
        assert taxonomy.is_descendant("grandchild1", "child1")
        assert not taxonomy.is_descendant("grandchild1", "child2")
        assert not taxonomy.is_descendant("root1", "grandchild1")

    def test_get_level(self, simple_taxonomy_dict):
        """Test getting disease level in hierarchy."""
        taxonomy = DiseaseTaxonomy(config_dict=simple_taxonomy_dict)

        assert taxonomy.get_level("root1") == 0
        assert taxonomy.get_level("root2") == 0
        assert taxonomy.get_level("child1") == 1
        assert taxonomy.get_level("child2") == 1
        assert taxonomy.get_level("grandchild1") == 2

    def test_get_diseases_at_level(self, simple_taxonomy_dict):
        """Test getting diseases at specific level."""
        taxonomy = DiseaseTaxonomy(config_dict=simple_taxonomy_dict)

        level_0 = taxonomy.get_diseases_at_level(0)
        assert set(level_0) == {"root1", "root2"}

        level_1 = taxonomy.get_diseases_at_level(1)
        assert set(level_1) == {"child1", "child2"}

        level_2 = taxonomy.get_diseases_at_level(2)
        assert level_2 == ["grandchild1"]

        level_3 = taxonomy.get_diseases_at_level(3)
        assert level_3 == []

    def test_get_leaf_diseases(self, simple_taxonomy_dict):
        """Test getting leaf diseases."""
        taxonomy = DiseaseTaxonomy(config_dict=simple_taxonomy_dict)

        leaves = taxonomy.get_leaf_diseases()
        assert set(leaves) == {"root2", "child2", "grandchild1"}

    def test_get_root_diseases(self, simple_taxonomy_dict):
        """Test getting root diseases."""
        taxonomy = DiseaseTaxonomy(config_dict=simple_taxonomy_dict)

        roots = taxonomy.get_root_diseases()
        assert set(roots) == {"root1", "root2"}

    def test_get_num_classes(self, simple_taxonomy_dict):
        """Test getting number of classes."""
        taxonomy = DiseaseTaxonomy(config_dict=simple_taxonomy_dict)

        assert taxonomy.get_num_classes() == 5

    def test_to_dict(self, simple_taxonomy_dict):
        """Test exporting taxonomy to dictionary."""
        taxonomy = DiseaseTaxonomy(config_dict=simple_taxonomy_dict)

        exported = taxonomy.to_dict()
        assert exported["name"] == "Test Taxonomy"
        assert exported["version"] == "1.0"
        assert len(exported["diseases"]) == 5

    def test_save_yaml(self, simple_taxonomy_dict):
        """Test saving taxonomy to YAML file."""
        taxonomy = DiseaseTaxonomy(config_dict=simple_taxonomy_dict)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            output_path = f.name

        try:
            taxonomy.save(output_path, format="yaml")

            # Load and verify
            with open(output_path, "r") as f:
                loaded = yaml.safe_load(f)

            assert loaded["name"] == "Test Taxonomy"
            assert len(loaded["diseases"]) == 5
        finally:
            Path(output_path).unlink()

    def test_save_json(self, simple_taxonomy_dict):
        """Test saving taxonomy to JSON file."""
        taxonomy = DiseaseTaxonomy(config_dict=simple_taxonomy_dict)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            output_path = f.name

        try:
            taxonomy.save(output_path, format="json")

            # Load and verify
            with open(output_path, "r") as f:
                loaded = json.load(f)

            assert loaded["name"] == "Test Taxonomy"
            assert len(loaded["diseases"]) == 5
        finally:
            Path(output_path).unlink()

    def test_save_unsupported_format(self, simple_taxonomy_dict):
        """Test saving with unsupported format."""
        taxonomy = DiseaseTaxonomy(config_dict=simple_taxonomy_dict)

        with pytest.raises(ValueError, match="Unsupported format"):
            taxonomy.save("output.txt", format="txt")

    def test_str_repr(self, simple_taxonomy_dict):
        """Test string representations."""
        taxonomy = DiseaseTaxonomy(config_dict=simple_taxonomy_dict)

        str_repr = str(taxonomy)
        assert "Test Taxonomy" in str_repr
        assert "1.0" in str_repr
        assert "5" in str_repr

        repr_str = repr(taxonomy)
        assert "DiseaseTaxonomy" in repr_str
        assert "Test Taxonomy" in repr_str
        assert "roots=2" in repr_str
        assert "leaves=3" in repr_str


class TestRealTaxonomies:
    """Test cases for real taxonomy configurations."""

    def test_cancer_grading_taxonomy(self):
        """Test cancer grading taxonomy."""
        taxonomy = DiseaseTaxonomy(config_path="configs/taxonomies/cancer_grading.yaml")

        assert taxonomy.name == "Cancer Grading System"
        assert taxonomy.version == "1.0"

        # Check structure
        roots = taxonomy.get_root_diseases()
        assert set(roots) == {"normal", "benign", "malignant"}

        # Check malignant grades
        malignant_children = taxonomy.get_children("malignant")
        assert set(malignant_children) == {"grade_1", "grade_2", "grade_3", "grade_4"}

        # Check benign subtypes
        benign_children = taxonomy.get_children("benign")
        assert set(benign_children) == {"benign_adenoma", "benign_hyperplasia", "benign_cyst"}

        # Check leaf diseases
        leaves = taxonomy.get_leaf_diseases()
        expected_leaves = {
            "normal",
            "benign_adenoma",
            "benign_hyperplasia",
            "benign_cyst",
            "grade_1",
            "grade_2",
            "grade_3",
            "grade_4",
        }
        assert set(leaves) == expected_leaves

    def test_cardiac_pathology_taxonomy(self):
        """Test cardiac pathology taxonomy."""
        taxonomy = DiseaseTaxonomy(config_path="configs/taxonomies/cardiac_pathology.yaml")

        assert taxonomy.name == "Cardiac Pathology Classification"
        assert taxonomy.version == "1.0"

        # Check structure
        roots = taxonomy.get_root_diseases()
        assert set(roots) == {"normal_cardiac", "ischemic", "inflammatory", "structural"}

        # Check ischemic subtypes
        ischemic_children = taxonomy.get_children("ischemic")
        assert set(ischemic_children) == {"acute_mi", "chronic_ischemia", "ischemic_cardiomyopathy"}

        # Check acute MI subtypes
        mi_children = taxonomy.get_children("acute_mi")
        assert set(mi_children) == {"stemi", "nstemi"}

        # Check metadata
        stemi = taxonomy.get_disease("stemi")
        assert stemi["metadata"]["risk_level"] == "critical"
        assert stemi["metadata"]["urgency"] == "emergency"

    def test_tissue_types_taxonomy(self):
        """Test tissue types taxonomy."""
        taxonomy = DiseaseTaxonomy(config_path="configs/taxonomies/tissue_types.yaml")

        assert taxonomy.name == "Tissue Type Classification"
        assert taxonomy.version == "1.0"

        # Check structure
        roots = taxonomy.get_root_diseases()
        assert set(roots) == {"epithelial", "connective", "muscle", "nervous"}

        # Check epithelial subtypes
        epithelial_children = taxonomy.get_children("epithelial")
        assert set(epithelial_children) == {
            "squamous_epithelium",
            "glandular_epithelium",
            "transitional_epithelium",
        }

        # Check muscle subtypes
        muscle_children = taxonomy.get_children("muscle")
        assert set(muscle_children) == {"skeletal_muscle", "cardiac_muscle", "smooth_muscle"}

        # Check metadata
        cardiac_muscle = taxonomy.get_disease("cardiac_muscle")
        assert cardiac_muscle["metadata"]["control"] == "involuntary"
        assert cardiac_muscle["metadata"]["striation"] == "striated"
