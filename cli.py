#!/usr/bin/env python3
import argparse
import sys
import logging
from pathlib import Path
from typing import List, Dict, Optional
from modules.knowledge_graph.graph_manager import KnowledgeGraphManager
from modules.security.access_control import AccessControl
import yaml

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class KnowledgeGraphCLI:
    def __init__(self, config_path: str = "config/config.yaml"):
        self.config = self._load_config(config_path)
        self.kg_manager = KnowledgeGraphManager(self.config)
        self.access_control = AccessControl(self.config)

    def _load_config(self, config_path: str) -> dict:
        """Load configuration from YAML file"""
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Failed to load config from {config_path}: {e}")
            sys.exit(1)

    def set_official_truth(self, entity1: str, relation: str, entity2: str, admin_id: str) -> bool:
        """Mark a relationship as official truth"""
        if not self.access_control.check_permission(admin_id, "manage:truth"):
            logger.error(f"User {admin_id} does not have permission to set official truth")
            return False

        success = self.kg_manager.mark_as_official_truth(entity1, relation, entity2, admin_id)
        if success:
            logger.info(f"Marked ({entity1})-[{relation}]->({entity2}) as official truth")
        else:
            logger.error(f"Failed to mark relationship as official truth")
        return success

    def remove_official_truth(self, entity1: str, relation: str, entity2: str, admin_id: str) -> bool:
        """Remove official truth status from a relationship"""
        if not self.access_control.check_permission(admin_id, "manage:truth"):
            logger.error(f"User {admin_id} does not have permission to remove official truth")
            return False

        try:
            with self.kg_manager.driver.session() as session:
                result = session.run(
                    f"""
                    MATCH (a:Entity {{name: $e1}})-[r:`{relation}`]->(b:Entity {{name: $e2}})
                    REMOVE r.official_truth
                    REMOVE r.official_truth_by
                    REMOVE r.official_truth_at
                    SET r.updated_at = timestamp()
                    RETURN r
                    """,
                    e1=entity1, e2=entity2
                )
                success = bool(result.single())
                if success:
                    logger.info(f"Removed official truth status from ({entity1})-[{relation}]->({entity2})")
                return success
        except Exception as e:
            logger.error(f"Error removing official truth: {e}")
            return False

    def list_relationships(self, filter_type: str = "all", limit: int = 100) -> List[Dict]:
        """List relationships with optional filtering"""
        try:
            with self.kg_manager.driver.session() as session:
                query = """
                MATCH (a:Entity)-[r]->(b:Entity)
                WHERE 1=1
                """
                
                if filter_type == "disputed":
                    query += " AND r.disputed = true"
                elif filter_type == "official":
                    query += " AND r.official_truth = true"
                
                query += """
                RETURN a.name AS entity1, 
                       type(r) AS relation, 
                       b.name AS entity2, 
                       r.confidence AS confidence,
                       r.source AS source,
                       r.official_truth AS is_official,
                       r.disputed AS is_disputed
                LIMIT $limit
                """
                
                result = session.run(query, limit=limit)
                relationships = []
                
                for record in result:
                    rel = {
                        "entity1": record["entity1"],
                        "relation": record["relation"],
                        "entity2": record["entity2"],
                        "confidence": record["confidence"],
                        "source": record["source"],
                        "is_official": record["is_official"],
                        "is_disputed": record["is_disputed"]
                    }
                    relationships.append(rel)
                
                return relationships
        except Exception as e:
            logger.error(f"Error listing relationships: {e}")
            return []

def main():
    parser = argparse.ArgumentParser(description="Knowledge Graph Management CLI")
    parser.add_argument("--config", default="config/config.yaml", help="Path to config file")

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Set official truth command
    set_parser = subparsers.add_parser("set-official", help="Mark a relationship as official truth")
    set_parser.add_argument("--entity1", required=True, help="Source entity name")
    set_parser.add_argument("--relation", required=True, help="Relationship type")
    set_parser.add_argument("--entity2", required=True, help="Target entity name")
    set_parser.add_argument("--admin-id", required=True, help="Admin user ID")

    # Remove official truth command
    unset_parser = subparsers.add_parser("unset-official", help="Remove official truth status")
    unset_parser.add_argument("--entity1", required=True)
    unset_parser.add_argument("--relation", required=True)
    unset_parser.add_argument("--entity2", required=True)
    unset_parser.add_argument("--admin-id", required=True)

    # List relationships command
    list_parser = subparsers.add_parser("list", help="List relationships")
    list_parser.add_argument("--filter", choices=["all", "disputed", "official"], 
                            default="all", help="Filter relationships")
    list_parser.add_argument("--limit", type=int, default=100, 
                            help="Maximum number of relationships to show")

    args = parser.parse_args()

    # Initialize CLI
    cli = KnowledgeGraphCLI(args.config)

    if args.command == "set-official":
        success = cli.set_official_truth(
            args.entity1, args.relation, args.entity2, args.admin_id
        )
        sys.exit(0 if success else 1)

    elif args.command == "unset-official":
        success = cli.remove_official_truth(
            args.entity1, args.relation, args.entity2, args.admin_id
        )
        sys.exit(0 if success else 1)

    elif args.command == "list":
        relationships = cli.list_relationships(args.filter, args.limit)
        if relationships:
            print("\nKnowledge Graph Relationships:")
            print("=" * 80)
            for rel in relationships:
                status = []
                if rel["is_official"]:
                    status.append("OFFICIAL")
                if rel["is_disputed"]:
                    status.append("DISPUTED")
                status_str = f" [{', '.join(status)}]" if status else ""
                
                print(f"{rel['entity1']} -[{rel['relation']}]-> {rel['entity2']}"
                      f"{status_str} (conf: {rel['confidence']:.2f})")
            print(f"\nTotal: {len(relationships)} relationships")
        else:
            print("No relationships found")

    else:
        parser.print_help()
        sys.exit(1)

if __name__ == "__main__":
    main() 