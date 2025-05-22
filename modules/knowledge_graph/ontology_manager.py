import yaml
import logging
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)

class OntologyManager:
    def __init__(self, schema_config_path: str = "config/schema.yaml"):
        self.schema_config_path = schema_config_path
        self.schema = self._load_schema()

    def _load_schema(self) -> Dict:
        try:
            with open(self.schema_config_path, 'r') as f:
                schema_data = yaml.safe_load(f)
                logger.info(f"Schema loaded successfully from {self.schema_config_path}")
                # Basic validation of schema structure
                if not isinstance(schema_data, dict) or \
                   not 'entity_types' in schema_data or \
                   not 'relationship_types' in schema_data:
                    logger.error("Invalid schema format. 'entity_types' and 'relationship_types' are required.")
                    return self._get_default_schema()
                return schema_data
        except FileNotFoundError:
            logger.warning(f"Schema file not found at {self.schema_config_path}. Using default schema.")
            return self._get_default_schema()
        except Exception as e:
            logger.error(f"Error loading schema from {self.schema_config_path}: {e}. Using default schema.")
            return self._get_default_schema()

    def _get_default_schema(self) -> Dict:
        # Provides a very basic fallback schema if loading fails
        return {
            'entity_types': ['Thing'],
            'relationship_types': {
                'relatedTo': {'domain': 'Thing', 'range': 'Thing'}
            }
        }

    def get_entity_types(self) -> List[str]:
        return self.schema.get('entity_types', [])

    def get_relationship_types(self) -> List[str]:
        return list(self.schema.get('relationship_types', {}).keys())

    def get_relationship_schema(self, rel_type: str) -> Optional[Dict[str, str]]:
        return self.schema.get('relationship_types', {}).get(rel_type)

    def is_valid_entity_type(self, entity_type: str) -> bool:
        return entity_type in self.get_entity_types()

    def is_valid_relationship(self, domain_type: str, rel_type: str, range_type: str) -> bool:
        if not self.is_valid_entity_type(domain_type):
            logger.debug(f"Invalid domain type: {domain_type}")
            return False
        if not self.is_valid_entity_type(range_type):
            logger.debug(f"Invalid range type: {range_type}")
            return False
        
        rel_schema = self.get_relationship_schema(rel_type)
        if not rel_schema:
            logger.debug(f"Unknown relationship type: {rel_type}")
            return False

        # Check domain constraint
        # For simplicity, direct match or if schema domain is 'Thing' (generic)
        # A more advanced check might involve traversing subclassOf relationships if they exist in entity types
        expected_domain = rel_schema.get('domain')
        if not (domain_type == expected_domain or expected_domain == 'Thing' or domain_type == 'Thing'):
            # Allow if the provided domain_type is a subclass of expected_domain (not implemented here yet)
            # For now, simple check. If ontology includes subclassOf, this needs hierarchy check.
            logger.debug(f"Domain type {domain_type} not valid for relationship {rel_type} (expected {expected_domain})")
            return False

        # Check range constraint
        expected_range = rel_schema.get('range')
        if not (range_type == expected_range or expected_range == 'Thing' or range_type == 'Thing'):
            logger.debug(f"Range type {range_type} not valid for relationship {rel_type} (expected {expected_range})")
            return False
            
        return True

if __name__ == '__main__':
    # Basic test usage
    # Adjust path if running this script directly from its directory for ad-hoc testing
    # Assuming project root is two levels up from modules/knowledge_graph/
    import os
    script_dir = os.path.dirname(__file__)
    project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))
    schema_path_for_direct_run = os.path.join(project_root, "config/schema.yaml")
    
    print(f"Attempting to load schema from: {schema_path_for_direct_run}")
    # Configure logging for the __main__ block to see output
    logging.basicConfig(level=logging.INFO)
    
    manager = OntologyManager(schema_config_path=schema_path_for_direct_run)
                                
    print("Entity Types:", manager.get_entity_types())
    print("Relationship Types:", manager.get_relationship_types())
    print("Schema for 'knows':", manager.get_relationship_schema('knows'))
    print("Is Person a valid entity type?", manager.is_valid_entity_type('Person'))
    print("Is 'worksFor(Person, Organization)' valid?", manager.is_valid_relationship('Person', 'worksFor', 'Organization'))
    print("Is 'worksFor(Location, Organization)' valid?", manager.is_valid_relationship('Location', 'worksFor', 'Organization')) # Expected False
    print("Is 'madeUpRel(Person, Organization)' valid?", manager.is_valid_relationship('Person', 'madeUpRel', 'Organization')) # Expected False

    # Test with default schema if actual schema file is missing
    print("\n--- Testing with a non-existent schema path to trigger default schema ---")
    manager_default = OntologyManager(schema_config_path="non_existent_schema.yaml")
    print("Entity Types (default):", manager_default.get_entity_types())
    print("Relationship Types (default):", manager_default.get_relationship_types())
    print("Is 'Thing' a valid entity type (default)?", manager_default.is_valid_entity_type('Thing'))
    print("Is 'relatedTo(Thing, Thing)' valid (default)?", manager_default.is_valid_relationship('Thing', 'relatedTo', 'Thing'))
    print("Is 'knows(Thing, Thing)' valid (default)?", manager_default.is_valid_relationship('Thing', 'knows', 'Thing')) # Expected False
```
