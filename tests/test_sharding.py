import pytest
from modules.knowledge_graph.shard_manager import ShardManager

@pytest.fixture
def shard_config():
    return {
        "node": {
            "id": "node1"
        },
        "sharding": {
            "enabled": True,
            "nodes": {
                "node1": {
                    "host": "host1",
                    "port": 7687,
                    "role": "full"
                },
                "node2": {
                    "host": "host2",
                    "port": 7687,
                    "role": "full"
                }
            }
        }
    }

def test_shard_distribution(shard_config):
    manager = ShardManager(shard_config, "node1")
    
    # Test consistent hashing
    entity1 = "Company_A"
    entity2 = "Company_B"
    
    shard1 = manager.get_shard_for_entity(entity1)
    shard2 = manager.get_shard_for_entity(entity2)
    
    # Same entity should always map to same shard
    assert shard1 == manager.get_shard_for_entity(entity1)
    
    # Entities should be distributed
    total_entities = 1000
    shard_counts = {}
    
    for i in range(total_entities):
        entity = f"Entity_{i}"
        shard = manager.get_shard_for_entity(entity)
        shard_counts[shard.node_id] = shard_counts.get(shard.node_id, 0) + 1
    
    # Check reasonable distribution (roughly equal)
    expected_per_shard = total_entities / len(shard_config["sharding"]["nodes"])
    for count in shard_counts.values():
        assert abs(count - expected_per_shard) < expected_per_shard * 0.2

def test_local_entity_handling(shard_config):
    manager = ShardManager(shard_config, "node1")
    
    # Test if node correctly identifies its entities
    local_entity = None
    remote_entity = None
    
    # Find an entity that hashes to our shard
    for i in range(1000):
        entity = f"Entity_{i}"
        if manager.should_handle_entity(entity):
            local_entity = entity
        else:
            remote_entity = entity
        if local_entity and remote_entity:
            break
    
    assert manager.should_handle_entity(local_entity)
    assert not manager.should_handle_entity(remote_entity) 