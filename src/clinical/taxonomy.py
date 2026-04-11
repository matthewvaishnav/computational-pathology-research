"""
Disease taxonomy configuration system for clinical workflow integration.

This module provides structured disease classification schemes that can be loaded
from configuration files and used throughout the clinical workflow system.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union

import yaml

logger = logging.getLogger(__name__)


class DiseaseTaxonomy:
    """
    Disease taxonomy configuration system supporting hierarchical disease classification.
    
    Supports loading disease taxonomies from YAML/JSON configuration files with
    hierarchical parent-child relationships, validation, and querying capabilities.
    
    Example taxonomy structure:
    ```yaml
    name: "Cancer Grading System"
    version: "1.0"
    diseases:
      - id: "benign"
        name: "Benign"
        description: "Non-cancerous tissue"
        parent: null
        children: ["benign_adenoma", "benign_hyperplasia"]
      - id: "malignant"
        name: "Malignant"
        description: "Cancerous tissue"
        parent: null
        children: ["grade_1", "grade_2", "grade_3"]
      - id: "grade_1"
        name: "Grade 1 Cancer"
        description: "Well-differentiated cancer"
        parent: "malignant"
        children: []
    ```
    """
    
    def __init__(self, config_path: Optional[Union[str, Path]] = None, config_dict: Optional[Dict[str, Any]] = None):
        """
        Initialize disease taxonomy from configuration file or dictionary.
        
        Args:
            config_path: Path to YAML/JSON configuration file
            config_dict: Dictionary containing taxonomy configuration
            
        Raises:
            ValueError: If neither config_path nor config_dict provided
            FileNotFoundError: If config_path doesn't exist
            ValidationError: If taxonomy configuration is invalid
        """
        if config_path is None and config_dict is None:
            raise ValueError("Either config_path or config_dict must be provided")
        
        if config_path is not None and config_dict is not None:
            raise ValueError("Only one of config_path or config_dict should be provided")
        
        if config_path is not None:
            self.config_path = Path(config_path)
            if not self.config_path.exists():
                raise FileNotFoundError(f"Taxonomy configuration file not found: {config_path}")
            
            # Load configuration from file
            with open(self.config_path, 'r', encoding='utf-8') as f:
                if self.config_path.suffix.lower() in ['.yaml', '.yml']:
                    self.config = yaml.safe_load(f)
                elif self.config_path.suffix.lower() == '.json':
                    self.config = json.load(f)
                else:
                    raise ValueError(f"Unsupported file format: {self.config_path.suffix}")
        else:
            self.config = config_dict
            self.config_path = None
        
        # Parse and validate taxonomy
        self._parse_taxonomy()
        self._validate_taxonomy()
        
        logger.info(f"Loaded disease taxonomy '{self.name}' with {len(self.diseases)} disease states")
    
    def _parse_taxonomy(self) -> None:
        """Parse taxonomy configuration into internal data structures."""
        # Extract metadata
        self.name = self.config.get('name', 'Unnamed Taxonomy')
        self.version = self.config.get('version', '1.0')
        self.description = self.config.get('description', '')
        
        # Parse diseases
        self.diseases: Dict[str, Dict[str, Any]] = {}
        self.disease_ids: List[str] = []
        self.root_diseases: List[str] = []
        self.leaf_diseases: List[str] = []
        
        diseases_config = self.config.get('diseases', [])
        if not isinstance(diseases_config, list):
            raise ValueError("'diseases' must be a list")
        
        for disease_config in diseases_config:
            if not isinstance(disease_config, dict):
                raise ValueError("Each disease must be a dictionary")
            
            disease_id = disease_config.get('id')
            if not disease_id:
                raise ValueError("Each disease must have an 'id' field")
            
            if disease_id in self.diseases:
                raise ValueError(f"Duplicate disease ID: {disease_id}")
            
            self.diseases[disease_id] = {
                'id': disease_id,
                'name': disease_config.get('name', disease_id),
                'description': disease_config.get('description', ''),
                'parent': disease_config.get('parent'),
                'children': disease_config.get('children', []),
                'metadata': disease_config.get('metadata', {})
            }
            
            self.disease_ids.append(disease_id)
    
    def _validate_taxonomy(self) -> None:
        """Validate taxonomy for completeness and consistency."""
        errors = []
        
        # Check for empty taxonomy
        if not self.diseases:
            errors.append("Taxonomy contains no diseases")
        
        # Validate parent-child relationships
        for disease_id, disease in self.diseases.items():
            parent_id = disease['parent']
            children_ids = disease['children']
            
            # Validate parent exists
            if parent_id is not None and parent_id not in self.diseases:
                errors.append(f"Disease '{disease_id}' references non-existent parent '{parent_id}'")
            
            # Validate children exist
            for child_id in children_ids:
                if child_id not in self.diseases:
                    errors.append(f"Disease '{disease_id}' references non-existent child '{child_id}'")
                else:
                    # Check bidirectional consistency
                    child_parent = self.diseases[child_id]['parent']
                    if child_parent != disease_id:
                        errors.append(f"Inconsistent parent-child relationship: '{disease_id}' lists '{child_id}' as child, but '{child_id}' has parent '{child_parent}'")
        
        # Check for cycles (only after validating all references exist)
        if not errors:  # Only check cycles if no reference errors
            visited = set()
            rec_stack = set()
            
            def has_cycle(disease_id: str) -> bool:
                if disease_id in rec_stack:
                    return True
                if disease_id in visited:
                    return False
                
                visited.add(disease_id)
                rec_stack.add(disease_id)
                
                for child_id in self.diseases[disease_id]['children']:
                    if has_cycle(child_id):
                        return True
                
                rec_stack.remove(disease_id)
                return False
            
            for disease_id in self.disease_ids:
                if disease_id not in visited and has_cycle(disease_id):
                    errors.append(f"Cycle detected in taxonomy involving disease '{disease_id}'")
        
        # Identify root and leaf diseases (only if no errors so far)
        if not errors:
            for disease_id, disease in self.diseases.items():
                if disease['parent'] is None:
                    self.root_diseases.append(disease_id)
                if not disease['children']:
                    self.leaf_diseases.append(disease_id)
            
            # Check for orphaned diseases (no path to root)
            reachable = set()
            
            def mark_reachable(disease_id: str):
                if disease_id in reachable:
                    return
                reachable.add(disease_id)
                for child_id in self.diseases[disease_id]['children']:
                    mark_reachable(child_id)
            
            for root_id in self.root_diseases:
                mark_reachable(root_id)
            
            orphaned = set(self.disease_ids) - reachable
            if orphaned:
                errors.append(f"Orphaned diseases (no path to root): {sorted(orphaned)}")
        
        if errors:
            error_msg = "Taxonomy validation failed:\n" + "\n".join(f"  - {error}" for error in errors)
            raise ValueError(error_msg)
    
    def get_disease(self, disease_id: str) -> Optional[Dict[str, Any]]:
        """
        Get disease information by ID.
        
        Args:
            disease_id: Disease identifier
            
        Returns:
            Disease information dictionary or None if not found
        """
        return self.diseases.get(disease_id)
    
    def get_children(self, disease_id: str) -> List[str]:
        """
        Get direct children of a disease.
        
        Args:
            disease_id: Disease identifier
            
        Returns:
            List of child disease IDs
        """
        disease = self.get_disease(disease_id)
        return disease['children'] if disease else []
    
    def get_parent(self, disease_id: str) -> Optional[str]:
        """
        Get parent of a disease.
        
        Args:
            disease_id: Disease identifier
            
        Returns:
            Parent disease ID or None if root disease
        """
        disease = self.get_disease(disease_id)
        return disease['parent'] if disease else None
    
    def get_ancestors(self, disease_id: str) -> List[str]:
        """
        Get all ancestors of a disease (path to root).
        
        Args:
            disease_id: Disease identifier
            
        Returns:
            List of ancestor disease IDs from immediate parent to root
        """
        ancestors = []
        current_id = disease_id
        
        while True:
            parent_id = self.get_parent(current_id)
            if parent_id is None:
                break
            ancestors.append(parent_id)
            current_id = parent_id
        
        return ancestors
    
    def get_descendants(self, disease_id: str) -> List[str]:
        """
        Get all descendants of a disease (entire subtree).
        
        Args:
            disease_id: Disease identifier
            
        Returns:
            List of descendant disease IDs
        """
        descendants = []
        
        def collect_descendants(current_id: str):
            for child_id in self.get_children(current_id):
                descendants.append(child_id)
                collect_descendants(child_id)
        
        collect_descendants(disease_id)
        return descendants
    
    def is_ancestor(self, ancestor_id: str, descendant_id: str) -> bool:
        """
        Check if one disease is an ancestor of another.
        
        Args:
            ancestor_id: Potential ancestor disease ID
            descendant_id: Potential descendant disease ID
            
        Returns:
            True if ancestor_id is an ancestor of descendant_id
        """
        return ancestor_id in self.get_ancestors(descendant_id)
    
    def is_descendant(self, descendant_id: str, ancestor_id: str) -> bool:
        """
        Check if one disease is a descendant of another.
        
        Args:
            descendant_id: Potential descendant disease ID
            ancestor_id: Potential ancestor disease ID
            
        Returns:
            True if descendant_id is a descendant of ancestor_id
        """
        return descendant_id in self.get_descendants(ancestor_id)
    
    def get_level(self, disease_id: str) -> int:
        """
        Get the level/depth of a disease in the hierarchy.
        
        Args:
            disease_id: Disease identifier
            
        Returns:
            Level (0 for root diseases, 1 for their children, etc.)
        """
        return len(self.get_ancestors(disease_id))
    
    def get_diseases_at_level(self, level: int) -> List[str]:
        """
        Get all diseases at a specific level in the hierarchy.
        
        Args:
            level: Hierarchy level (0 for root, 1 for children of root, etc.)
            
        Returns:
            List of disease IDs at the specified level
        """
        return [disease_id for disease_id in self.disease_ids if self.get_level(disease_id) == level]
    
    def get_leaf_diseases(self) -> List[str]:
        """
        Get all leaf diseases (diseases with no children).
        
        Returns:
            List of leaf disease IDs
        """
        return self.leaf_diseases.copy()
    
    def get_root_diseases(self) -> List[str]:
        """
        Get all root diseases (diseases with no parent).
        
        Returns:
            List of root disease IDs
        """
        return self.root_diseases.copy()
    
    def get_num_classes(self) -> int:
        """
        Get total number of disease classes in taxonomy.
        
        Returns:
            Number of disease classes
        """
        return len(self.diseases)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Export taxonomy to dictionary format.
        
        Returns:
            Dictionary representation of taxonomy
        """
        return {
            'name': self.name,
            'version': self.version,
            'description': self.description,
            'diseases': [self.diseases[disease_id] for disease_id in self.disease_ids]
        }
    
    def save(self, output_path: Union[str, Path], format: str = 'yaml') -> None:
        """
        Save taxonomy to file.
        
        Args:
            output_path: Output file path
            format: Output format ('yaml' or 'json')
        """
        output_path = Path(output_path)
        taxonomy_dict = self.to_dict()
        
        with open(output_path, 'w', encoding='utf-8') as f:
            if format.lower() == 'yaml':
                yaml.dump(taxonomy_dict, f, default_flow_style=False, indent=2)
            elif format.lower() == 'json':
                json.dump(taxonomy_dict, f, indent=2)
            else:
                raise ValueError(f"Unsupported format: {format}")
        
        logger.info(f"Saved taxonomy to {output_path}")
    
    def __str__(self) -> str:
        """String representation of taxonomy."""
        return f"DiseaseTaxonomy(name='{self.name}', version='{self.version}', diseases={len(self.diseases)})"
    
    def __repr__(self) -> str:
        """Detailed string representation of taxonomy."""
        return (f"DiseaseTaxonomy(name='{self.name}', version='{self.version}', "
                f"diseases={len(self.diseases)}, roots={len(self.root_diseases)}, "
                f"leaves={len(self.leaf_diseases)})")