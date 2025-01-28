import hashlib
from typing import List, Dict, Optional
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class ShardInfo:
    node_id: str
    shard_range: tuple[int, int]  # (start, end) of hash range
    host: str
    port: int

class ShardManager:
    """Manages the distribution of entities across shards"""
    
    def __init__(self, config, node_id: str):
        self.config = config
        self.node_id = node_id
        self.shard_map: Dict[str, ShardInfo] = {}
        self.hash_range = 2**32  # Use 32-bit hash space
        
        # Load shard configuration
        self._load_shard_config()
        
        # Get our shard range
        self.our_shard = self.shard_map.get(node_id)
        if not self.our_shard:
            logger.error(f"No shard configuration found for node {node_id}")
            raise ValueError(f"Missing shard config for node {node_id}")

    def _load_shard_config(self):
        """Load shard configuration from config file"""
        shard_config = self.config.get("sharding", {}).get("nodes", {})
        total_nodes = len(shard_config)
        if total_nodes == 0:
            logger.warning("No shard configuration found, defaulting to single node")
            # Default to single node handling full range
            self.shard_map[self.node_id] = ShardInfo(
                node_id=self.node_id,
                shard_range=(0, self.hash_range),
                host="localhost",
                port=7687
            )
            return

        # Calculate shard ranges
        range_size = self.hash_range // total_nodes
        for i, (node_id, node_config) in enumerate(shard_config.items()):
            start = i * range_size
            end = start + range_size if i < total_nodes - 1 else self.hash_range
            
            self.shard_map[node_id] = ShardInfo(
                node_id=node_id,
                shard_range=(start, end),
                host=node_config["host"],
                port=node_config["port"]
            )

    def get_shard_for_entity(self, entity_id: str) -> ShardInfo:
        """Determine which shard should store this entity"""
        hash_value = self._hash_entity(entity_id)
        
        for shard in self.shard_map.values():
            if shard.shard_range[0] <= hash_value < shard.shard_range[1]:
                return shard
                
        # Should never happen with proper ranges
        raise ValueError(f"No shard found for hash value {hash_value}")

    def should_handle_entity(self, entity_id: str) -> bool:
        """Check if this node should handle the given entity"""
        shard = self.get_shard_for_entity(entity_id)
        return shard.node_id == self.node_id

    def _hash_entity(self, entity_id: str) -> int:
        """Create consistent hash for entity ID"""
        hash_obj = hashlib.md5(entity_id.encode())
        # Use first 4 bytes as 32-bit integer
        return int.from_bytes(hash_obj.digest()[:4], byteorder='big')

    def get_all_shards(self) -> List[ShardInfo]:
        """Get information about all shards"""
        return list(self.shard_map.values())

    def get_neighbor_shards(self) -> List[ShardInfo]:
        """Get information about other shards"""
        return [shard for shard in self.shard_map.values() 
                if shard.node_id != self.node_id] 