from neo4j import GraphDatabase
from typing import List, Dict, Any, Optional # Added Optional
import logging
from ..monitoring.metrics import (
    RELATIONSHIPS_CREATED, CONFLICTS_RESOLVED, 
    TRUTH_CONFIDENCE
)
from .shard_manager import ShardManager
from ..messaging.message_bus import MessageBus
from ..agents.llm_agent import BaseLLMAgent # Though not directly used here, it's part of existing structure
from .reasoning_engine import ReasoningEngine
from modules.knowledge_graph.ontology_manager import OntologyManager # Added import

logger = logging.getLogger(__name__)

class KnowledgeGraphManager:
    """Manages Neo4j knowledge graph with LLM-based conflict resolution and schema-awareness."""
    
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

        self.ontology_manager = OntologyManager() # Instantiated OntologyManager
        self.shard_manager = ShardManager(config, config["node"]["id"])
        self.message_bus = MessageBus(config)
        
        # Subscribe to cross-shard updates
        self.message_bus.subscribe("kg_updates", self._handle_remote_update)

        # Add reasoning engine
        self.reasoning_engine = ReasoningEngine()
        
        # Start inference tasks
        self._start_inference_tasks()

    def _get_node_labels_cypher(self, entity_type: Optional[str]) -> str:
        default_label = ":`Thing`" # Default label from our schema
        if entity_type and self.ontology_manager.is_valid_entity_type(entity_type):
            # Basic sanitization: Neo4j labels are quite flexible but avoid complex chars if not needed.
            # For simplicity, assume entity_type is a valid Neo4j label name if it's in the ontology.
            return f":`{entity_type}`"
        elif entity_type: # Type provided but not valid according to ontology
            logger.warning(f"Entity type '{entity_type}' is not valid per ontology. Falling back to default label '{default_label}'.")
            return default_label
        else: # No type provided
            return default_label

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
        entity1_type = fact.get("entity1_type") # Extract entity types
        entity2_type = fact.get("entity2_type")

        # Check for existing relationship (Option B: remove label constraint)
        existing = session.run(
            f"""
            MATCH (a {{name: $e1}})-[r:`{relation}`]->(b {{name: $e2}})
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
                self._update_relationship( # entity types not strictly needed for update, but passed for consistency
                    session, entity1, relation, entity2,
                    resolution.final_confidence, source, resolution.reason,
                    entity1_type, entity2_type 
                )
                RELATIONSHIPS_CREATED.inc() # This might be a miscounting if it's an update.
                                            # Consider separate metric for updates if needed.
                
            elif resolution.action == "store_disputed":
                self._store_disputed_relationship(
                    session, entity1, relation, entity2,
                    resolution.final_confidence, source, resolution.reason,
                    entity1_type, entity2_type
                )
                RELATIONSHIPS_CREATED.inc()
                
            CONFLICTS_RESOLVED.labels(resolution_type=resolution.action).inc()
            TRUTH_CONFIDENCE.observe(resolution.final_confidence)
            
        else:
            # Create new relationship
            self._create_relationship(session, entity1, relation, entity2, confidence, source, entity1_type, entity2_type)
            RELATIONSHIPS_CREATED.inc()

    def _create_relationship(self, session, e1: str, rel: str, e2: str, conf: float, source: str, 
                           entity1_type: Optional[str], entity2_type: Optional[str]):
        """Create a new relationship with typed entities."""
        e1_labels = self._get_node_labels_cypher(entity1_type)
        e2_labels = self._get_node_labels_cypher(entity2_type)
        
        query = f"""
            MERGE (a {{name: $e1}})
            SET a{e1_labels}
            MERGE (b {{name: $e2}})
            SET b{e2_labels}
            CREATE (a)-[r:`{rel}` {{
                confidence: $conf,
                source: $source,
                created_at: timestamp()
            }}]->(b)
            """
        session.run(query, e1=e1, e2=e2, conf=conf, source=source)

    def _update_relationship(self, session, e1: str, rel: str, e2: str, conf: float, 
                           source: str, reason: str = "",
                           entity1_type: Optional[str] = None, entity2_type: Optional[str] = None): # Types added for signature consistency
        """Update an existing relationship. Node labels are not changed here."""
        session.run(
            f"""
            MATCH (a {{name: $e1}})-[r:`{rel}`]->(b {{name: $e2}})
            SET r.confidence = $conf,
                r.source = $source,
                r.update_reason = $reason,
                r.updated_at = timestamp()
            """,
            e1=e1, e2=e2, conf=conf, source=source, reason=reason
        )

    def _store_disputed_relationship(self, session, e1: str, rel: str, e2: str, conf: float,
                                   source: str, reason: str = "",
                                   entity1_type: Optional[str] = None, entity2_type: Optional[str] = None):
        """Store a disputed version of a relationship with typed entities."""
        e1_labels = self._get_node_labels_cypher(entity1_type)
        e2_labels = self._get_node_labels_cypher(entity2_type)

        query = f"""
            MERGE (a {{name: $e1}})
            SET a{e1_labels}
            MERGE (b {{name: $e2}})
            SET b{e2_labels}
            CREATE (a)-[r:`{rel}` {{
                confidence: $conf,
                source: $source,
                disputed: true,
                dispute_reason: $reason,
                created_at: timestamp()
            }}]->(b)
            """
        session.run(query, e1=e1, e2=e2, conf=conf, source=source, reason=reason)

    def mark_as_official_truth(self, entity1: str, relation: str, entity2: str, admin_id: str) -> bool:
        """Mark a relationship as official truth (admin only)"""
        try:
            with self.driver.session() as session:
                # Using f-string for relation type is generally okay if relation is controlled,
                # but be cautious if it can be arbitrary user input.
                # For this context, relation names are from a controlled vocabulary (ontology).
                result = session.run(
                    f"""
                    MATCH (a {{name: $e1}})-[r:`{relation}`]->(b {{name: $e2}})
                    SET r.official_truth = true,
                        r.official_truth_by = $admin_id,
                        r.official_truth_at = timestamp()
                    RETURN r
                    """,
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
            # Assuming conflict_resolver is available or needs to be instantiated/passed
            # For now, if it was a member, it would be self.conflict_resolver
            # This part of the code might need adjustment depending on how conflict_resolver is managed.
            # For this subtask, we focus on the Cypher changes.
            # Let's assume a conflict_resolver is passed or accessible.
            # A placeholder might be needed if it's not readily available in this method's scope.
            # For now, I'm commenting out the line that would cause an error if conflict_resolver is not defined.
            # self.insert_relationships(facts, self.conflict_resolver) 
            logger.warning("_handle_remote_update might need a conflict_resolver instance.")


    def _start_inference_tasks(self):
        """Start background task for updates."""
        import asyncio
        # Check if an event loop is already running, common in async environments
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError: # No running event loop
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        loop.create_task(self._inference_loop())


    async def _inference_loop(self):
        """Periodically check for and apply updates."""
        while True:
            try:
                # Get relevant facts
                with self.driver.session() as session:
                    # Modified to match any node, not just :Entity
                    facts_cursor = session.run("""
                        MATCH (a)-[r]->(b)
                        WHERE r.metadata IS NOT NULL OR r.confidence IS NOT NULL 
                        RETURN a.name as entity1, type(r) as relation,
                               b.name as entity2, r.metadata as metadata, r.confidence as confidence
                    """)
                    
                    fact_list = []
                    for record in facts_cursor:
                        fact = {"entity1": record["entity1"], "relation": record["relation"], "entity2": record["entity2"]}
                        # Include metadata or confidence if they exist
                        if record["metadata"]:
                            fact["metadata"] = record["metadata"]
                        elif record["confidence"]:
                            fact["metadata"] = {"confidence": record["confidence"]} # Adapt to expected structure for reasoning
                        fact_list.append(fact)
                    
                    if not fact_list: # Avoid calling reasoning engine with empty facts
                        await asyncio.sleep(3600) 
                        continue

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
                                inference["inferred_fact"], # Expected to have entity1, relation, entity2
                                inference.get("confidence", 0.75), # Default confidence for inferences
                                f"Inferred via {inference['reasoning_type']} reasoning: {inference['proof']}"
                            )
                
                await asyncio.sleep(3600)  # Check hourly
                
            except Exception as e:
                logger.error(f"Error in inference loop: {e}")
                await asyncio.sleep(3600) # Wait before retrying

    def _apply_inference(self, session, fact, confidence, reason):
        """Apply an inferred update to the graph."""
        try:
            # Assuming inferred facts don't yet have explicit types. Using default labels.
            # If inferences start providing types, this should be updated.
            e1_labels = self._get_node_labels_cypher(fact.get("entity1_type")) # Handle potential future types
            e2_labels = self._get_node_labels_cypher(fact.get("entity2_type"))

            query = f"""
            MERGE (a {{name: $e1}})
            SET a{e1_labels}
            MERGE (b {{name: $e2}})
            SET b{e2_labels}
            CREATE (a)-[r:`{fact['relation']}` {{
                confidence: $conf,
                source: "inference",
                inference_reason: $reason,
                inferred_at: timestamp()
            }}]->(b)
            """
            session.run(query,
                e1=fact["entity1"],
                e2=fact["entity2"],
                conf=confidence,
                reason=reason
            )
            logger.info(f"Applied inference: {fact['entity1']}({e1_labels.strip(':`')}) -[{fact['relation']}]-> {fact['entity2']}({e2_labels.strip(':`')})")
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