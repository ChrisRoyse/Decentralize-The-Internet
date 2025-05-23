import pytest
from unittest.mock import MagicMock, patch, call, ANY
import asyncio # Required for patching create_task

from modules.knowledge_graph.graph_manager import KnowledgeGraphManager
# OntologyManager is patched where it's used, but import for type hinting if needed by KGM's __init__
# from modules.knowledge_graph.ontology_manager import OntologyManager


# --- Mocks for Neo4j Driver/Session/Transaction ---
@pytest.fixture
def mock_neo4j_tx():
    mock_tx = MagicMock()
    # Configure tx.run().single() and tx.run().data() as needed per test
    mock_tx.run.return_value.single.return_value = None # Default for "no existing relationship"
    mock_tx.run.return_value.data.return_value = []
    return mock_tx

@pytest.fixture
def mock_neo4j_session(mock_neo4j_tx):
    mock_session = MagicMock()
    # Simulate 'with session.begin_transaction() as tx:'
    mock_session.begin_transaction.return_value.__enter__.return_value = mock_neo4j_tx
    # Simulate 'with self.driver.session() as session:' for direct session usage if any
    mock_session.__enter__.return_value = mock_session 
    return mock_session

@pytest.fixture
def mock_neo4j_driver(mock_neo4j_session):
    mock_driver = MagicMock()
    # Simulate 'with self.driver.session() as session:'
    mock_driver.session.return_value.__enter__.return_value = mock_neo4j_session
    return mock_driver

# --- Mock for OntologyManager ---
@pytest.fixture
def mock_ontology_manager_instance():
    mock_om = MagicMock(spec=True) # spec=True helps catch calls to non-existent methods
    
    def is_valid_side_effect(entity_type):
        # Define valid types for tests explicitly
        return entity_type in ["Person", "Organization", "Location", "Event", "Thing", "Software"]
        
    mock_om.is_valid_entity_type.side_effect = is_valid_side_effect
    # _get_node_labels_cypher is internal to KGM, so we test its effects on Cypher, not the mock_om directly for it.
    # KGM's _get_node_labels_cypher calls mock_om.is_valid_entity_type.
    return mock_om

# --- Dummy KGM Config ---
@pytest.fixture
def kgm_config():
    return {
        "knowledge_graph": {"neo4j_uri": "bolt://dummy:7687", "neo4j_user": "neo4j", "neo4j_password": "password"},
        "node": {"id": "test_node_1"} # For ShardManager, if its __init__ is called
    }

# --- Fixture for KnowledgeGraphManager Instance ---
@pytest.fixture
@patch('modules.knowledge_graph.graph_manager.GraphDatabase')
@patch('modules.knowledge_graph.graph_manager.OntologyManager')
@patch('modules.knowledge_graph.graph_manager.ShardManager')
@patch('modules.knowledge_graph.graph_manager.MessageBus')
@patch('modules.knowledge_graph.graph_manager.ReasoningEngine')
@patch('asyncio.create_task') # Patch asyncio.create_task
def kg_manager_instance(
    mock_asyncio_create_task, # Order matters for mock args
    MockReasoningEngine, MockMessageBus, MockShardManager, 
    MockOntologyManagerClass, MockGraphDatabase,
    kgm_config, mock_neo4j_driver, mock_ontology_manager_instance
):
    MockGraphDatabase.driver.return_value = mock_neo4j_driver
    # Ensure KGM's __init__ uses our pre-configured mock_ontology_manager_instance
    MockOntologyManagerClass.return_value = mock_ontology_manager_instance 
    
    # Provide mocks for other dependencies if their __init__ is complex or has side effects
    MockShardManager.return_value = MagicMock()
    MockMessageBus.return_value = MagicMock()
    MockReasoningEngine.return_value = MagicMock()

    kgm = KnowledgeGraphManager(kgm_config)
    # kgm.ontology_manager is already set by patching the class constructor's return value
    return kgm

# --- Test Scenarios ---

def test_ontology_manager_instantiated(kg_manager_instance, mock_ontology_manager_instance):
    # The fixture kg_manager_instance patches OntologyManager constructor.
    # We check if the instance used by kgm is the one returned by the patched constructor.
    assert kg_manager_instance.ontology_manager == mock_ontology_manager_instance
    # Check if the constructor was called (via the patched class)
    # This is implicitly tested by mock_ontology_manager_instance being used.
    # To be more explicit on constructor call:
    from modules.knowledge_graph.graph_manager import OntologyManager as PatchedOntologyManager
    PatchedOntologyManager.assert_called_once()


def test_create_relationship_specific_types(kg_manager_instance, mock_neo4j_session, mock_neo4j_tx, mock_ontology_manager_instance):
    kgm = kg_manager_instance
    
    # Ensure OntologyManager allows these types
    mock_ontology_manager_instance.is_valid_entity_type.side_effect = lambda x: x in ["Person", "Organization", "Thing"]

    # Simulate _process_fact leading to _create_relationship
    # Initial check for existing relationship returns None
    mock_neo4j_tx.run.return_value.single.return_value = None 
    
    test_fact = {
        "entity1": "Alice", "entity1_type": "Person",
        "relation": "WORKS_FOR",
        "entity2": "AcmeCorp", "entity2_type": "Organization",
        "confidence": 0.9, "source": "test_src"
    }
    mock_conflict_resolver = MagicMock() # Not used if no existing relationship

    kgm._process_fact(mock_neo4j_session, test_fact, mock_conflict_resolver)

    # Check the Cypher queries run by _create_relationship
    # mock_tx.run.call_args_list gives list of all calls to tx.run
    # Each call is a tuple: ( (pos_arg1, pos_arg2, ...), {kwarg1: val1, ...} )
    
    # First call to tx.run is the check for existing, second is _create_relationship's MERGE/SET/CREATE
    # This depends on the exact sequence of calls. If _process_fact has one tx.run, and _create_relationship has one.
    # Let's assume _create_relationship has one major call.
    # This might be fragile if multiple tx.run calls are in _create_relationship.
    # A better way is to capture all calls and check for specific patterns.
    
    # Capture all Cypher queries
    all_cypher_queries = [args[0] for args, kwargs in mock_neo4j_tx.run.call_args_list]
    
    # Look for the MERGE/SET/CREATE query from _create_relationship
    # It should be the one that contains the actual relationship creation.
    create_query = None
    for q in all_cypher_queries:
        if f"`{test_fact['relation']}`" in q and "CREATE (a)-[r:" in q:
            create_query = q
            break
    
    assert create_query is not None, "Create relationship query not found"

    # Normalize whitespace for robust comparison
    normalized_query = " ".join(create_query.split())
    
    assert "MERGE (a {name: $e1})" in normalized_query
    assert "SET a:`Person`" in normalized_query # Check for specific label for Alice
    assert "MERGE (b {name: $e2})" in normalized_query
    assert "SET b:`Organization`" in normalized_query # Check for specific label for AcmeCorp
    assert f"CREATE (a)-[r:`{test_fact['relation']}` {{" in normalized_query


def test_create_relationship_missing_type(kg_manager_instance, mock_neo4j_session, mock_neo4j_tx, mock_ontology_manager_instance):
    kgm = kg_manager_instance
    mock_ontology_manager_instance.is_valid_entity_type.side_effect = lambda x: x in ["Organization", "Thing"] # Person is not valid here
    
    mock_neo4j_tx.run.return_value.single.return_value = None
    test_fact = {
        "entity1": "Charlie", # No entity1_type, should default to Thing
        "relation": "FOUNDED",
        "entity2": "StartupY", "entity2_type": "Organization",
        "confidence": 0.85, "source": "test_src_missing"
    }
    mock_conflict_resolver = MagicMock()
    kgm._process_fact(mock_neo4j_session, test_fact, mock_conflict_resolver)
    
    all_cypher_queries = [args[0] for args, kwargs in mock_neo4j_tx.run.call_args_list]
    create_query = next(q for q in all_cypher_queries if f"`{test_fact['relation']}`" in q and "CREATE (a)-[r:" in q)
    normalized_query = " ".join(create_query.split())

    assert "SET a:`Thing`" in normalized_query # Default for Charlie
    assert "SET b:`Organization`" in normalized_query # Specific for StartupY


def test_create_relationship_invalid_type(kg_manager_instance, mock_neo4j_session, mock_neo4j_tx, mock_ontology_manager_instance):
    kgm = kg_manager_instance
    # Make "InvalidType" not valid, but "Organization" and "Thing" valid
    mock_ontology_manager_instance.is_valid_entity_type.side_effect = lambda x: x in ["Organization", "Thing"]
    
    mock_neo4j_tx.run.return_value.single.return_value = None
    test_fact = {
        "entity1": "David", "entity1_type": "InvalidType", # Invalid, should default to Thing
        "relation": "INVESTED_IN",
        "entity2": "CompanyZ", "entity2_type": "Organization",
        "confidence": 0.7, "source": "test_src_invalid"
    }
    mock_conflict_resolver = MagicMock()
    kgm._process_fact(mock_neo4j_session, test_fact, mock_conflict_resolver)
    
    all_cypher_queries = [args[0] for args, kwargs in mock_neo4j_tx.run.call_args_list]
    create_query = next(q for q in all_cypher_queries if f"`{test_fact['relation']}`" in q and "CREATE (a)-[r:" in q)
    normalized_query = " ".join(create_query.split())

    assert "SET a:`Thing`" in normalized_query # Default for David due to invalid type
    assert "SET b:`Organization`" in normalized_query


def test_store_disputed_relationship_specific_types(kg_manager_instance, mock_neo4j_session, mock_neo4j_tx, mock_ontology_manager_instance):
    kgm = kg_manager_instance
    mock_ontology_manager_instance.is_valid_entity_type.side_effect = lambda x: x in ["Person", "Event", "Thing"]

    # Simulate existing relationship
    mock_neo4j_tx.run.return_value.single.return_value = {"r": {"confidence": 0.5, "source": "old_src"}} 
    
    test_fact = {
        "entity1": "Eve", "entity1_type": "Person",
        "relation": "ATTENDED",
        "entity2": "ConferenceX", "entity2_type": "Event",
        "confidence": 0.9, "source": "new_src_disputed"
    }
    mock_conflict_resolver = MagicMock()
    mock_conflict_resolver.resolve_conflict.return_value = MagicMock(action="store_disputed", final_confidence=0.9, reason="test dispute")
    
    kgm._process_fact(mock_neo4j_session, test_fact, mock_conflict_resolver)
    
    # The first call to tx.run is the check, the second is _store_disputed_relationship
    # This is assuming _process_fact itself doesn't make other tx.run calls before calling the helper.
    dispute_query_call = mock_neo4j_tx.run.call_args_list[1] # Get the second call
    dispute_query = dispute_query_call[0][0] # The Cypher string
    normalized_query = " ".join(dispute_query.split())

    assert "MERGE (a {name: $e1})" in normalized_query
    assert "SET a:`Person`" in normalized_query
    assert "MERGE (b {name: $e2})" in normalized_query
    assert "SET b:`Event`" in normalized_query
    assert f"CREATE (a)-[r:`{test_fact['relation']}` {{" in normalized_query
    assert "disputed: true" in normalized_query


def test_apply_inference_with_types(kg_manager_instance, mock_neo4j_session, mock_neo4j_tx, mock_ontology_manager_instance):
    kgm = kg_manager_instance
    mock_ontology_manager_instance.is_valid_entity_type.side_effect = lambda x: x in ["Software", "Organization", "Thing"]
    
    inferred_fact = {
        "entity1": "ProductA", "entity1_type": "Software",
        "relation": "DEVELOPED_BY",
        "entity2": "DevHouse", "entity2_type": "Organization"
    }
    confidence = 0.88
    reason = "Logical deduction"
    
    kgm._apply_inference(mock_neo4j_session, inferred_fact, confidence, reason)
    
    # _apply_inference makes one call to tx.run
    inference_query = mock_neo4j_tx.run.call_args[0][0]
    normalized_query = " ".join(inference_query.split())
    
    assert "MERGE (a {name: $e1})" in normalized_query
    assert "SET a:`Software`" in normalized_query
    assert "MERGE (b {name: $e2})" in normalized_query
    assert "SET b:`Organization`" in normalized_query
    assert f"CREATE (a)-[r:`{inferred_fact['relation']}` {{" in normalized_query
    assert "source: \"inference\"" in normalized_query


def test_apply_inference_without_types(kg_manager_instance, mock_neo4j_session, mock_neo4j_tx, mock_ontology_manager_instance):
    kgm = kg_manager_instance
    # Ensure 'Thing' is the only valid type for this test's scope if specific types aren't provided.
    mock_ontology_manager_instance.is_valid_entity_type.side_effect = lambda x: x == "Thing"

    inferred_fact = { # No entity types provided
        "entity1": "ConceptX",
        "relation": "RELATED_TO",
        "entity2": "ConceptY"
    }
    confidence = 0.75
    reason = "General inference"
    
    kgm._apply_inference(mock_neo4j_session, inferred_fact, confidence, reason)
    
    inference_query = mock_neo4j_tx.run.call_args[0][0]
    normalized_query = " ".join(inference_query.split())

    assert "SET a:`Thing`" in normalized_query # Default label
    assert "SET b:`Thing`" in normalized_query # Default label


# --- Tests for MATCH Clause Modifications ---

def test_match_process_fact_existing_rel(kg_manager_instance, mock_neo4j_session, mock_neo4j_tx):
    kgm = kg_manager_instance
    test_fact = {"entity1": "E1", "relation": "REL", "entity2": "E2"}
    mock_conflict_resolver = MagicMock() # Needed for _process_fact signature

    # We are interested in the first tx.run call inside _process_fact
    kgm._process_fact(mock_neo4j_session, test_fact, mock_conflict_resolver)
    
    # The first call to tx.run is the check for existing relationship
    check_query = mock_neo4j_tx.run.call_args_list[0][0][0] # Get Cypher string of first call
    normalized_query = " ".join(check_query.split())

    assert "MATCH (a {name: $e1})-[r:`REL`]->(b {name: $e2})" in normalized_query
    assert ":Entity" not in normalized_query # Crucial check


def test_match_update_relationship(kg_manager_instance, mock_neo4j_session, mock_neo4j_tx):
    kgm = kg_manager_instance
    # Call _update_relationship directly or trigger via _process_fact
    # Direct call is easier for isolating the query
    kgm._update_relationship(mock_neo4j_session, "E1_update", "REL_UPDATE", "E2_update", 0.99, "test_update")
    
    update_query = mock_neo4j_tx.run.call_args[0][0]
    normalized_query = " ".join(update_query.split())
    
    assert "MATCH (a {name: $e1})-[r:`REL_UPDATE`]->(b {name: $e2})" in normalized_query
    assert ":Entity" not in normalized_query


def test_match_mark_as_official_truth(kg_manager_instance, mock_neo4j_session, mock_neo4j_tx):
    kgm = kg_manager_instance
    kgm.mark_as_official_truth("E1_official", "REL_OFFICIAL", "E2_official", "admin_user")
    
    official_query = mock_neo4j_tx.run.call_args[0][0] # Cypher is first arg to tx.run
    # The query in mark_as_official_truth uses f-string for relation, so it will be part of the string directly
    normalized_query = " ".join(official_query.split())
    
    assert "MATCH (a {name: $e1})-[r:`REL_OFFICIAL`]->(b {name: $e2})" in normalized_query
    assert ":Entity" not in normalized_query

print("") # Ensures newline at end of file
```
