from neo4j import GraphDatabase
from typing import List, Dict, Any
import logging
from ..monitoring.metrics import (
    RELATIONSHIPS_CREATED, CONFLICTS_RESOLVED, 
    TRUTH_CONFIDENCE
)
from .shard_manager import ShardManager
from ..messaging.message_bus import MessageBus
from ..agents.llm_agent import BaseLLMAgent
from .reasoning_engine import ReasoningEngine

logger = logging.getLogger(__name__)

class KnowledgeGraphManager:
    """Manages Neo4j knowledge graph with LLM-based conflict resolution."""
    
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
            logger.info("Connected to Neo4j")
        except Exception as e:
            logger.error(f"Failed to connect to Neo4j: {e}")
            raise

        self.shard_manager = ShardManager(config, config["node"]["id"])
        self.message_bus = MessageBus(config)
        
        # Subscribe to cross-shard updates
        self.message_bus.subscribe("kg_updates", self._handle_remote_update)

        # Add reasoning engine
        self.reasoning_engine = ReasoningEngine()
        
        # Start inference tasks
        self._start_inference_tasks()

    def insert_relationships(self, facts: List[Dict[str, Any]], conflict_resolver):
        """Insert facts into graph, using LLM for conflict resolution."""
        with self.driver.session() as session:
            for fact in facts:
                try:
                    self._process_fact(session, fact, conflict_resolver)
                except Exception as e:
                    logger.error(f"Error processing fact {fact}: {e}")

    def _process_fact(self, session, fact: Dict[str, Any], conflict_resolver):
        """Process a single fact, handling conflicts with LLM."""
        entity1 = fact["entity1"]
        relation = fact["relation"]
        entity2 = fact["entity2"]
        confidence = fact.get("confidence", 1.0)
        source = fact.get("source", "unknown")

        # Check for existing relationship
        existing = session.run(
            f"""
            MATCH (a:Entity {{name: $e1}})-[r:`{relation}`]->(b:Entity {{name: $e2}})
            RETURN r
            """,
            e1=entity1, e2=entity2
        ).single()

        if existing and existing["r"]:
            # Handle conflict with LLM
            existing_edge = {
                "confidence": existing["r"]["confidence"],
                "source": existing["r"].get("source", "unknown"),
                "official_truth": existing["r"].get("official_truth", False)
            }
            new_edge = {
                "confidence": confidence,
                "source": source
            }

            resolution = conflict_resolver.resolve_conflict(existing_edge, new_edge)
            
            if resolution.action == "overwrite":
                self._update_relationship(
                    session, entity1, relation, entity2,
                    resolution.final_confidence, source, resolution.reason
                )
                RELATIONSHIPS_CREATED.inc()
                
            elif resolution.action == "store_disputed":
                self._store_disputed_relationship(
                    session, entity1, relation, entity2,
                    resolution.final_confidence, source, resolution.reason
                )
                RELATIONSHIPS_CREATED.inc()
                
            CONFLICTS_RESOLVED.labels(resolution_type=resolution.action).inc()
            TRUTH_CONFIDENCE.observe(resolution.final_confidence)
            
        else:
            # Create new relationship
            self._create_relationship(session, entity1, relation, entity2, confidence, source)
            RELATIONSHIPS_CREATED.inc()

    def _create_relationship(self, session, e1: str, rel: str, e2: str, conf: float, source: str):
        """Create a new relationship."""
        session.run(
            f"""
            MERGE (a:Entity {{name: $e1}})
            MERGE (b:Entity {{name: $e2}})
            CREATE (a)-[r:`{rel}` {{
                confidence: $conf,
                source: $source,
                created_at: timestamp()
            }}]->(b)
            """,
            e1=e1, e2=e2, conf=conf, source=source
        )

    def _update_relationship(self, session, e1: str, rel: str, e2: str, conf: float, 
                           source: str, reason: str = ""):
        """Update an existing relationship."""
        session.run(
            f"""
            MATCH (a:Entity {{name: $e1}})-[r:`{rel}`]->(b:Entity {{name: $e2}})
            SET r.confidence = $conf,
                r.source = $source,
                r.update_reason = $reason,
                r.updated_at = timestamp()
            """,
            e1=e1, e2=e2, conf=conf, source=source, reason=reason
        )

    def _store_disputed_relationship(self, session, e1: str, rel: str, e2: str, conf: float,
                                   source: str, reason: str = ""):
        """Store a disputed version of a relationship."""
        session.run(
            f"""
            MATCH (a:Entity {{name: $e1}}), (b:Entity {{name: $e2}})
            CREATE (a)-[r:`{rel}` {{
                confidence: $conf,
                source: $source,
                disputed: true,
                dispute_reason: $reason,
                created_at: timestamp()
            }}]->(b)
            """,
            e1=e1, e2=e2, conf=conf, source=source, reason=reason
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
        """Close the database connection."""
        self.driver.close()

    def _handle_remote_update(self, message: Dict):
        """Handle updates forwarded from other shards"""
        if message.get("target_node") != self.shard_manager.node_id:
            return
            
        facts = message.get("facts", [])
        if facts:
            self.insert_relationships(facts, self.conflict_resolver) 

    def _start_inference_tasks(self):
        """Start background task for updates."""
        import asyncio
        asyncio.create_task(self._inference_loop())

    async def _inference_loop(self):
        """Periodically check for and apply updates."""
        while True:
            try:
                # Get relevant facts
                with self.driver.session() as session:
                    facts = session.run("""
                        MATCH (a:Entity)-[r]->(b:Entity)
                        WHERE r.metadata IS NOT NULL
                        RETURN a.name as entity1, type(r) as relation,
                               b.name as entity2, r.metadata as metadata
                    """)
                    
                    fact_list = list(facts)
                    
                    # Get new inferences
                    inferences = self.reasoning_engine.infer_new_knowledge(
                        fact_list,
                        reasoning_types=["temporal", "causal", "logical", "taxonomic"]
                    )
                    
                    for inference in inferences:
                        # Validate inference
                        if self.reasoning_engine.validate_inference(inference):
                            # Apply the inference
                            self._apply_inference(
                                session,
                                inference["inferred_fact"],
                                inference["confidence"],
                                f"Inferred via {inference['reasoning_type']} reasoning: {inference['proof']}"
                            )
                
                await asyncio.sleep(3600)  # Check hourly
                
            except Exception as e:
                logger.error(f"Error in inference loop: {e}")
                await asyncio.sleep(3600)

    def _apply_inference(self, session, fact, confidence, reason):
        """Apply an inferred update to the graph."""
        try:
            session.run(
                f"""
                MERGE (a:Entity {{name: $e1}})
                MERGE (b:Entity {{name: $e2}})
                CREATE (a)-[r:`{fact['relation']}` {{
                    confidence: $conf,
                    source: "inference",
                    inference_reason: $reason,
                    inferred_at: timestamp()
                }}]->(b)
                """,
                e1=fact["entity1"],
                e2=fact["entity2"],
                conf=confidence,
                reason=reason
            )
            logger.info(f"Applied inference: {fact['entity1']}->{fact['entity2']}")
        except Exception as e:
            logger.error(f"Error applying inference: {e}")

class KnowledgeGraphAgent(BaseLLMAgent):
    """Agent for managing knowledge graph operations using ASF framework."""
    
    def validate_update(self, operation: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and enrich graph updates using Agent-Specific Framework."""
        prompt = self._format_agent_prompt(
            role="""You are the Chief Knowledge Graph Validator responsible for 
maintaining data quality and consistency.""",
            objective="""Validate and enrich proposed knowledge graph updates before 
they are committed to ensure data quality and consistency.""",
            context="""Our knowledge graph represents real-world facts with high confidence.
Updates must be carefully validated to maintain trust and accuracy.
Ground truth takes precedence over new information unless compelling evidence exists.""",
            sop=f"""Analyze this proposed graph update:
{operation}

Follow these steps:
1. Verify entity normalization
   - Check entity names are standardized
   - Resolve any ambiguous references
   
2. Validate relationship semantics
   - Confirm relationship type is valid
   - Check for logical consistency
   
3. Assess confidence scoring
   - Review confidence calculation
   - Adjust if needed based on evidence
   
4. Check for conflicts
   - Identify potential contradictions
   - Flag for human review if critical

5. Enrich metadata
   - Add relevant timestamps
   - Include validation notes""",
            tools_available="""Available Tools:
- Entity Resolution API
- Temporal Reasoning Engine
- Confidence Scoring Model
- Conflict Detection System""",
            constraints="""Constraints:
- Must maintain backwards compatibility
- Cannot override admin-marked ground truth
- Must preserve audit trail
- Must handle edge cases gracefully"""
        )
        
        response = self._call_llm(prompt, temperature=0.2)
        
        try:
            import json
            result = json.loads(response)
            return {
                "valid": result.get("valid", False),
                "enriched_operation": result.get("enriched_operation", operation),
                "validation_notes": result.get("validation_notes", []),
                "confidence_adjustment": result.get("confidence_adjustment", 0.0)
            }
        except Exception as e:
            logger.error(f"Error in graph validation: {e}")
            return {
                "valid": False,
                "enriched_operation": operation,
                "validation_notes": ["Failed to validate due to error"],
                "confidence_adjustment": -0.2
            }

    def _format_agent_prompt(self, role: str, objective: str, context: str, 
                           sop: str, tools_available: str, constraints: str) -> str:
        """Format prompt using Agent-Specific Framework from paper."""
        return f"""# Role
{role}

# Objective
{objective}

# Context
{context}

# Standard Operating Procedure
{sop}

# Available Tools
{tools_available}

# Constraints
{constraints}

Response (in JSON):""" 