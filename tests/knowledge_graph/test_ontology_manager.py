import pytest
import yaml
import os
import logging
from modules.knowledge_graph.ontology_manager import OntologyManager

# Configure basic logging for tests to see OntologyManager logs if needed
# logging.basicConfig(level=logging.DEBUG) # Use DEBUG to see detailed logs from OntologyManager
logger = logging.getLogger(__name__)

# --- Test Schema Definitions ---

@pytest.fixture(scope="session")
def valid_schema_content():
    return {
        'entity_types': ['Person', 'Organization', 'Location', 'Event', 'Thing', 'Software'],
        'relationship_types': {
            'worksFor': {'domain': 'Person', 'range': 'Organization'},
            'locatedIn': {'domain': 'Organization', 'range': 'Location'},
            'participantIn': {'domain': 'Person', 'range': 'Event'},
            'uses': {'domain': 'Person', 'range': 'Software'},
            'genericRel': {'domain': 'Thing', 'range': 'Thing'},
            'develops': {'domain': 'Organization', 'range': 'Software'}
        }
    }

@pytest.fixture(scope="session")
def malformed_schema_content_missing_entities():
    return {
        # 'entity_types': ['Person', 'Organization'], # Missing entity_types
        'relationship_types': {
            'worksFor': {'domain': 'Person', 'range': 'Organization'}
        }
    }

@pytest.fixture(scope="session")
def malformed_schema_content_bad_yaml():
    return "this is not valid yaml: { "

@pytest.fixture(scope="session")
def default_schema_content():
    return {
        'entity_types': ['Thing'],
        'relationship_types': {
            'relatedTo': {'domain': 'Thing', 'range': 'Thing'}
        }
    }

# --- File Fixtures ---

@pytest.fixture
def valid_schema_file(tmp_path, valid_schema_content):
    schema_file = tmp_path / "valid_test_schema.yaml"
    with open(schema_file, 'w') as f:
        yaml.dump(valid_schema_content, f)
    return str(schema_file)

@pytest.fixture
def malformed_file_missing_keys(tmp_path, malformed_schema_content_missing_entities):
    schema_file = tmp_path / "malformed_test_schema_keys.yaml"
    with open(schema_file, 'w') as f:
        yaml.dump(malformed_schema_content_missing_entities, f)
    return str(schema_file)

@pytest.fixture
def malformed_file_bad_yaml(tmp_path, malformed_schema_content_bad_yaml):
    schema_file = tmp_path / "malformed_test_schema_bad.yaml"
    with open(schema_file, 'w') as f:
        f.write(malformed_schema_content_bad_yaml)
    return str(schema_file)

# --- OntologyManager Instance Fixtures ---

@pytest.fixture
def manager_with_valid_schema(valid_schema_file):
    return OntologyManager(schema_config_path=valid_schema_file)

@pytest.fixture
def manager_for_default_schema_on_missing_file():
    # Using a path that definitely won't exist
    return OntologyManager(schema_config_path="non_existent_schema_for_test.yaml")

# --- Schema Loading Tests ---

def test_load_valid_schema(manager_with_valid_schema, valid_schema_content):
    logger.debug(f"Loaded schema in test_load_valid_schema: {manager_with_valid_schema.schema}")
    assert manager_with_valid_schema.schema == valid_schema_content
    assert manager_with_valid_schema.get_entity_types() == valid_schema_content['entity_types']

def test_load_schema_file_not_found(manager_for_default_schema_on_missing_file, default_schema_content, caplog):
    caplog.set_level(logging.WARNING)
    # OntologyManager already initialized by fixture, which should trigger loading
    assert manager_for_default_schema_on_missing_file.schema == default_schema_content
    assert "Schema file not found" in caplog.text
    assert "Using default schema" in caplog.text

def test_load_malformed_schema_missing_keys(malformed_file_missing_keys, default_schema_content, caplog):
    caplog.set_level(logging.ERROR)
    manager = OntologyManager(schema_config_path=malformed_file_missing_keys)
    assert manager.schema == default_schema_content
    assert "Invalid schema format" in caplog.text
    assert "Using default schema" in caplog.text

def test_load_malformed_schema_bad_yaml(malformed_file_bad_yaml, default_schema_content, caplog):
    caplog.set_level(logging.ERROR)
    manager = OntologyManager(schema_config_path=malformed_file_bad_yaml)
    assert manager.schema == default_schema_content
    assert "Error loading schema" in caplog.text
    assert "Using default schema" in caplog.text

# --- Getter Method Tests ---

def test_get_entity_types(manager_with_valid_schema, valid_schema_content):
    assert sorted(manager_with_valid_schema.get_entity_types()) == sorted(valid_schema_content['entity_types'])

def test_get_relationship_types(manager_with_valid_schema, valid_schema_content):
    assert sorted(manager_with_valid_schema.get_relationship_types()) == sorted(list(valid_schema_content['relationship_types'].keys()))

def test_get_relationship_schema(manager_with_valid_schema, valid_schema_content):
    # Test for a valid relationship type
    rel_type = 'worksFor'
    expected_schema = valid_schema_content['relationship_types'][rel_type]
    assert manager_with_valid_schema.get_relationship_schema(rel_type) == expected_schema
    
    # Test for a non-existent relationship type
    assert manager_with_valid_schema.get_relationship_schema('nonExistentRel') is None

# --- Validation Method Tests ---

@pytest.mark.parametrize("entity_type, expected", [
    ('Person', True),
    ('Organization', True),
    ('Location', True),
    ('Event', True),
    ('Thing', True),
    ('Software', True),
    ('UnknownType', False),
    ('', False)
])
def test_is_valid_entity_type(manager_with_valid_schema, entity_type, expected):
    assert manager_with_valid_schema.is_valid_entity_type(entity_type) == expected

@pytest.mark.parametrize("domain_type, rel_type, range_type, expected_validity", [
    # Valid cases based on valid_schema_content
    ('Person', 'worksFor', 'Organization', True),
    ('Organization', 'locatedIn', 'Location', True),
    ('Person', 'participantIn', 'Event', True),
    ('Person', 'uses', 'Software', True),
    ('Organization', 'develops', 'Software', True),
    
    # Valid cases using 'Thing' as domain or range if relationship is generic
    ('Person', 'genericRel', 'Location', True), # Person is a Thing, Location is a Thing
    ('Thing', 'genericRel', 'Thing', True),
    ('Software', 'genericRel', 'Event', True), # Software is a Thing, Event is a Thing

    # Invalid relationship type
    ('Person', 'nonExistentRel', 'Organization', False),

    # Valid relationship type, but invalid domain
    ('Location', 'worksFor', 'Organization', False), # worksFor domain is Person
    ('Event', 'develops', 'Software', False), # develops domain is Organization

    # Valid relationship type, but invalid range
    ('Person', 'worksFor', 'Location', False), # worksFor range is Organization
    ('Organization', 'develops', 'Person', False), # develops range is Software

    # Entity types themselves are not in the schema
    ('UnknownDomain', 'worksFor', 'Organization', False),
    ('Person', 'worksFor', 'UnknownRange', False),
    ('UnknownDomain', 'nonExistentRel', 'UnknownRange', False),
    
    # Implicit 'Thing' compatibility for domain/range with specific schema
    # The current implementation of is_valid_relationship in OntologyManager:
    # expected_domain = rel_schema.get('domain')
    # if not (domain_type == expected_domain or expected_domain == 'Thing' or domain_type == 'Thing'):
    # This means if schema expects 'Person', and you give 'Thing', it's FALSE.
    # If schema expects 'Thing', and you give 'Person', it's TRUE.
    # Let's test these specific cases based on the current logic.
    ('Thing', 'worksFor', 'Organization', False), # Schema domain is 'Person', 'Thing' is not 'Person'.
    ('Person', 'genericRel', 'Organization', True), # Schema domain is 'Thing', 'Person' is compatible.
    ('Organization', 'genericRel', 'Person', True), # Schema range is 'Thing', 'Person' is compatible.
    
    # Empty types
    ('', 'worksFor', 'Organization', False),
    ('Person', '', 'Organization', False),
    ('Person', 'worksFor', '', False),
])
def test_is_valid_relationship(manager_with_valid_schema, domain_type, rel_type, range_type, expected_validity):
    assert manager_with_valid_schema.is_valid_relationship(domain_type, rel_type, range_type) == expected_validity

# Test for default schema validation logic
def test_default_schema_validations(manager_for_default_schema_on_missing_file, default_schema_content):
    manager = manager_for_default_schema_on_missing_file
    assert manager.is_valid_entity_type('Thing') == True
    assert manager.is_valid_entity_type('Person') == False # Default schema only has 'Thing'
    
    assert manager.is_valid_relationship('Thing', 'relatedTo', 'Thing') == True
    assert manager.is_valid_relationship('Person', 'relatedTo', 'Thing') == False # Domain 'Person' not in default schema
    assert manager.is_valid_relationship('Thing', 'relatedTo', 'Organization') == False # Range 'Organization' not in default schema
    assert manager.is_valid_relationship('Thing', 'knows', 'Thing') == False # 'knows' not in default schema relationships

print("") # Ensures newline at end of file
