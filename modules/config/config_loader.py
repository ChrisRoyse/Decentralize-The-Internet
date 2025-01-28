import yaml
from .secret_loader import get_neo4j_password, get_encryption_key

class ConfigLoader:
    @staticmethod
    def load_config(config_path: str = "config/config.yaml") -> dict:
        """Load configuration with secrets"""
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Inject secrets
        if 'knowledge_graph' in config:
            config['knowledge_graph']['neo4j_password'] = get_neo4j_password()
        
        if 'security' in config:
            config['security']['encryption_key'] = get_encryption_key()
        
        return config 