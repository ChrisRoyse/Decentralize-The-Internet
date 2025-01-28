from neo4j import GraphDatabase
from typing import List, Dict, Any
import logging
from ..monitoring.metrics import (
    RELATIONSHIPS_CREATED, CONFLICTS_RESOLVED, 
    TRUTH_CONFIDENCE
)
from .shard_manager import ShardManager
from ..messaging.message_bus import MessageBus

logger = logging.getLogger(__name__)

class KnowledgeGraphManager:
    """Manages interactions with the Neo4j knowledge graph database"""
    
    def __init__(self, config):
        self.config = config
        self.uri = config["knowledge_graph"]["neo4j_uri"]
        self.user = config["knowledge_graph"]["neo4j_user"]
        self.password = config["knowledge_graph"]["neo4j_password"]
        
        try:
            self.driver = GraphDatabase.driver(
                self.uri, 
                auth=(self.user, self.password)
            )
            logger.info("Successfully connected to Neo4j database")
        except Exception as e:
            logger.error(f"Failed to connect to Neo4j: {e}")
            raise

        self.shard_manager = ShardManager(config, config["node"]["id"])
        self.message_bus = MessageBus(config)
        
        # Subscribe to cross-shard updates
        self.message_bus.subscribe("kg_updates", self._handle_remote_update)

    def insert_relationships(self, facts: List[Dict[str, Any]], conflict_resolver) -> None:
        """Insert relationships, handling sharding"""
        local_facts = []
        remote_facts = {}
        
        # Sort facts by shard
        for fact in facts:
            entity1 = fact["entity1"]
            entity2 = fact["entity2"]
            
            # If either entity belongs to another shard, queue for forwarding
            shard1 = self.shard_manager.get_shard_for_entity(entity1)
            shard2 = self.shard_manager.get_shard_for_entity(entity2)
            
            if shard1.node_id != self.shard_manager.node_id:
                remote_facts.setdefault(shard1.node_id, []).append(fact)
            elif shard2.node_id != self.shard_manager.node_id:
                remote_facts.setdefault(shard2.node_id, []).append(fact)
            else:
                local_facts.append(fact)
        
        # Process local facts
        if local_facts:
            self._process_local_facts(local_facts, conflict_resolver)
        
        # Forward remote facts
        for node_id, node_facts in remote_facts.items():
            self.message_bus.publish("kg_updates", {
                "target_node": node_id,
                "facts": node_facts
            })

    def _process_local_facts(self, facts: List[Dict], conflict_resolver) -> None:
        """Process facts that belong to this shard"""
        with self.driver.session() as session:
            for fact in facts:
                try:
                    self._process_fact(session, fact, conflict_resolver)
                except Exception as e:
                    logger.error(f"Error processing fact {fact}: {e}")

    def _process_fact(self, session, fact: Dict[str, Any], conflict_resolver) -> None:
        """Process a single fact, handling conflicts"""
        entity1 = fact["entity1"]
        relation = fact["relation"]
        entity2 = fact["entity2"]
        confidence = fact.get("confidence", 1.0)
        source = fact.get("source", "unknown")

        # Check for existing relationship
        existing = session.run(
            """
            MATCH (a:Entity {name: $e1})-[r:`%s`]->(b:Entity {name: $e2})
            RETURN r
            """ % relation,
            e1=entity1, e2=entity2
        ).single()

        if existing and existing["r"]:
            # Handle conflict
            resolution = conflict_resolver.resolve_conflict(
                existing_edge={
                    "confidence": existing["r"]["confidence"],
                    "source": existing["r"].get("source", "unknown"),
                    "official_truth": existing["r"].get("official_truth", False)
                },
                new_edge={
                    "confidence": confidence,
                    "source": source
                }
            )

            if resolution.action == "overwrite":
                self._update_relationship(
                    session, 
                    entity1, 
                    relation, 
                    entity2, 
                    resolution.final_confidence, 
                    source,
                    resolution.reason
                )
            elif resolution.action == "store_disputed":
                self._store_disputed_relationship(
                    session,
                    entity1,
                    relation,
                    entity2,
                    resolution.final_confidence,
                    source,
                    resolution.reason
                )
        else:
            # Create new relationship
            self._create_relationship(session, entity1, relation, entity2, confidence, source)

        CONFLICTS_RESOLVED.labels(
            resolution_type=resolution.action
        ).inc()
        
        if resolution.action in ["overwrite", "store_disputed"]:
            RELATIONSHIPS_CREATED.inc()
            TRUTH_CONFIDENCE.observe(resolution.final_confidence)

    def _create_relationship(self, session, entity1: str, relation: str, entity2: str, 
                           confidence: float, source: str) -> None:
        """Create a new relationship between entities"""
        session.run(
            """
            MERGE (a:Entity {name: $e1})
            MERGE (b:Entity {name: $e2})
            CREATE (a)-[r:`%s` {
                confidence: $conf,
                source: $source,
                timestamp: timestamp()
            }]->(b)
            """ % relation,
            e1=entity1, e2=entity2, conf=confidence, source=source
        )

    def _update_relationship(self, session, entity1: str, relation: str, entity2: str,
                           confidence: float, source: str, reason: str = "") -> None:
        """Update an existing relationship with new confidence and metadata"""
        session.run(
            """
            MATCH (a:Entity {name: $e1})-[r:`%s`]->(b:Entity {name: $e2})
            SET r.confidence = $conf,
                r.truth_confidence = $conf,
                r.source = $source,
                r.update_reason = $reason,
                r.updated_at = timestamp()
            """ % relation,
            e1=entity1, e2=entity2, conf=confidence, source=source, reason=reason
        )

    def _store_disputed_relationship(self, session, entity1: str, relation: str, entity2: str,
                                   confidence: float, source: str, reason: str = "") -> None:
        """Store a disputed version of a relationship with metadata"""
        session.run(
            """
            MATCH (a:Entity {name: $e1}), (b:Entity {name: $e2})
            CREATE (a)-[r:`%s` {
                confidence: $conf,
                source: $source,
                disputed: true,
                update_reason: $reason,
                timestamp: timestamp()
            }]->(b)
            """ % relation,
            e1=entity1, e2=entity2, conf=confidence, source=source, reason=reason
        )

    def mark_as_official_truth(self, entity1: str, relation: str, entity2: str, admin_id: str) -> bool:
        """Mark a relationship as official truth (admin only)"""
        try:
            with self.driver.session() as session:
                result = session.run(
                    """
                    MATCH (a:Entity {name: $e1})-[r:`%s`]->(b:Entity {name: $e2})
                    SET r.official_truth = true,
                        r.official_truth_by = $admin_id,
                        r.official_truth_at = timestamp()
                    RETURN r
                    """ % relation,
                    e1=entity1, e2=entity2, admin_id=admin_id
                )
                return bool(result.single())
        except Exception as e:
            logger.error(f"Error marking relationship as official truth: {e}")
            return False

    def close(self):
        """Close the database connection"""
        self.driver.close()

    def _handle_remote_update(self, message: Dict):
        """Handle updates forwarded from other shards"""
        if message.get("target_node") != self.shard_manager.node_id:
            return
            
        facts = message.get("facts", [])
        if facts:
            self.insert_relationships(facts, self.conflict_resolver) 