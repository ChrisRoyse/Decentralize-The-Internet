from typing import Set, Dict, Optional
import logging
from dataclasses import dataclass
import time
import jwt
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

@dataclass
class UserRole:
    name: str
    permissions: Set[str]

@dataclass
class User:
    id: str
    username: str
    roles: Set[str]
    active: bool = True

class AccessControl:
    """
    Manages role-based access control and permissions
    """
    
    def __init__(self, config):
        self.config = config
        
        # Initialize roles and permissions
        self.roles: Dict[str, UserRole] = self._init_roles()
        
        # User storage
        self.users: Dict[str, User] = {}
        
        # JWT settings
        self.jwt_secret = config["security"].get("jwt_secret", "your-secret-key")
        self.token_expiry = timedelta(hours=24)
        
    def _init_roles(self) -> Dict[str, UserRole]:
        """Initialize role definitions"""
        return {
            "admin": UserRole(
                name="admin",
                permissions={
                    "read:all", "write:all", "delete:all", "manage:users"
                }
            ),
            "crawler": UserRole(
                name="crawler",
                permissions={
                    "read:urls", "write:urls", "read:content", "write:content"
                }
            ),
            "analyst": UserRole(
                name="analyst",
                permissions={
                    "read:content", "read:analytics", "write:analytics"
                }
            )
        }

    def add_user(self, user_id: str, username: str, roles: Set[str]) -> bool:
        """Add a new user"""
        try:
            # Validate roles
            for role in roles:
                if role not in self.roles:
                    logger.error(f"Invalid role: {role}")
                    return False
            
            self.users[user_id] = User(
                id=user_id,
                username=username,
                roles=roles
            )
            return True
            
        except Exception as e:
            logger.error(f"Error adding user: {e}")
            return False

    def check_permission(self, user_id: str, permission: str) -> bool:
        """Check if user has a specific permission"""
        try:
            user = self.users.get(user_id)
            if not user or not user.active:
                return False
                
            # Get all permissions from user's roles
            user_permissions = set()
            for role_name in user.roles:
                role = self.roles.get(role_name)
                if role:
                    user_permissions.update(role.permissions)
                    
            return permission in user_permissions
            
        except Exception as e:
            logger.error(f"Error checking permission: {e}")
            return False

    def generate_token(self, user_id: str) -> Optional[str]:
        """Generate JWT token for user"""
        try:
            user = self.users.get(user_id)
            if not user or not user.active:
                return None
                
            payload = {
                "user_id": user_id,
                "username": user.username,
                "roles": list(user.roles),
                "exp": datetime.utcnow() + self.token_expiry
            }
            
            return jwt.encode(payload, self.jwt_secret, algorithm="HS256")
            
        except Exception as e:
            logger.error(f"Error generating token: {e}")
            return None

    def validate_token(self, token: str) -> Optional[Dict]:
        """Validate JWT token and return payload"""
        try:
            payload = jwt.decode(token, self.jwt_secret, algorithms=["HS256"])
            
            # Check if user still exists and is active
            user_id = payload.get("user_id")
            user = self.users.get(user_id)
            if not user or not user.active:
                return None
                
            return payload
            
        except jwt.ExpiredSignatureError:
            logger.warning("Token has expired")
            return None
        except jwt.InvalidTokenError as e:
            logger.warning(f"Invalid token: {e}")
            return None
        except Exception as e:
            logger.error(f"Error validating token: {e}")
            return None

    def deactivate_user(self, user_id: str) -> bool:
        """Deactivate a user"""
        try:
            if user_id in self.users:
                self.users[user_id].active = False
                return True
            return False
        except Exception as e:
            logger.error(f"Error deactivating user: {e}")
            return False 