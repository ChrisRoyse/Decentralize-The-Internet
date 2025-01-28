import pytest
from unittest.mock import Mock, patch
from cli import KnowledgeGraphCLI

@pytest.fixture
def mock_config():
    return {
        "knowledge_graph": {
            "neo4j_uri": "bolt://localhost:7687",
            "neo4j_user": "neo4j",
            "neo4j_password": "password"
        },
        "security": {
            "allowed_roles": ["admin", "user"]
        }
    }

@pytest.fixture
def cli(mock_config):
    with patch('cli.KnowledgeGraphCLI._load_config', return_value=mock_config):
        return KnowledgeGraphCLI()

def test_set_official_truth_with_permission(cli):
    # Mock access control to allow the operation
    cli.access_control.check_permission = Mock(return_value=True)
    cli.kg_manager.mark_as_official_truth = Mock(return_value=True)

    result = cli.set_official_truth(
        entity1="CompanyA",
        relation="acquired",
        entity2="StartupB",
        admin_id="admin1"
    )

    assert result == True
    cli.kg_manager.mark_as_official_truth.assert_called_once()

def test_set_official_truth_without_permission(cli):
    # Mock access control to deny the operation
    cli.access_control.check_permission = Mock(return_value=False)

    result = cli.set_official_truth(
        entity1="CompanyA",
        relation="acquired",
        entity2="StartupB",
        admin_id="user1"
    )

    assert result == False
    assert not hasattr(cli.kg_manager, 'mark_as_official_truth.called')

def test_list_relationships(cli):
    mock_relationships = [
        {
            "entity1": "CompanyA",
            "relation": "acquired",
            "entity2": "StartupB",
            "confidence": 0.9,
            "source": "news.com",
            "is_official": True,
            "is_disputed": False
        }
    ]
    
    # Mock the session and run methods
    mock_session = Mock()
    mock_session.run.return_value = [Mock(**{"values.return_value": list(r.values())}) 
                                    for r in mock_relationships]
    
    with patch.object(cli.kg_manager, 'driver') as mock_driver:
        mock_driver.session.return_value.__enter__.return_value = mock_session
        
        results = cli.list_relationships(filter_type="official")
        
        assert len(results) == 1
        assert results[0]["entity1"] == "CompanyA"
        assert results[0]["is_official"] == True 