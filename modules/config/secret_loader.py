import os
from typing import Optional

def read_secret(secret_name: str) -> Optional[str]:
    """Read a secret from Docker secrets or environment variable"""
    # Check for Docker secret file
    secret_file = f"/run/secrets/{secret_name}"
    if os.path.exists(secret_file):
        with open(secret_file, 'r') as f:
            return f.read().strip()
    
    # Fall back to environment variable
    env_var = f"{secret_name.upper()}"
    return os.environ.get(env_var)

def get_neo4j_password() -> str:
    """Get Neo4j password from secrets or environment"""
    password = None
    
    # Try Docker secret file
    if os.environ.get('NEO4J_PASSWORD_FILE'):
        with open(os.environ['NEO4J_PASSWORD_FILE'], 'r') as f:
            password = f.read().strip()
    
    # Fall back to environment variable
    if not password:
        password = os.environ.get('NEO4J_PASSWORD', 'neo4j')
    
    return password

def get_encryption_key() -> bytes:
    """Get encryption key from secrets or environment"""
    key = None
    
    # Try Docker secret file
    if os.environ.get('ENCRYPTION_KEY_FILE'):
        with open(os.environ['ENCRYPTION_KEY_FILE'], 'r') as f:
            key = f.read().strip().encode()
    
    # Fall back to environment variable
    if not key:
        key = os.environ.get('ENCRYPTION_KEY', 'default_key').encode()
    
    return key 