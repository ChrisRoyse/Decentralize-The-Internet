from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives import serialization
import base64
import os
import logging
from typing import Tuple, Optional

logger = logging.getLogger(__name__)

class EncryptionManager:
    """
    Handles symmetric and asymmetric encryption for secure communication
    """
    
    def __init__(self, config):
        self.config = config
        
        # Generate or load encryption keys
        self._init_keys()
        
        # Create Fernet instance for symmetric encryption
        self.fernet = Fernet(self.symmetric_key)
        
    def _init_keys(self):
        """Initialize encryption keys"""
        try:
            # Generate symmetric key if not exists
            self.symmetric_key = self._generate_symmetric_key()
            
            # Generate asymmetric keypair if not exists
            self.private_key = rsa.generate_private_key(
                public_exponent=65537,
                key_size=2048
            )
            self.public_key = self.private_key.public_key()
            
            logger.info("Encryption keys initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize encryption keys: {e}")
            raise

    def _generate_symmetric_key(self) -> bytes:
        """Generate a symmetric key using a password and salt"""
        password = self.config["security"].get("encryption_password", "default_password").encode()
        salt = os.urandom(16)
        
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        
        key = base64.urlsafe_b64encode(kdf.derive(password))
        return key

    def encrypt_symmetric(self, data: bytes) -> bytes:
        """Encrypt data using symmetric encryption"""
        try:
            return self.fernet.encrypt(data)
        except Exception as e:
            logger.error(f"Symmetric encryption failed: {e}")
            raise

    def decrypt_symmetric(self, encrypted_data: bytes) -> bytes:
        """Decrypt symmetrically encrypted data"""
        try:
            return self.fernet.decrypt(encrypted_data)
        except Exception as e:
            logger.error(f"Symmetric decryption failed: {e}")
            raise

    def encrypt_asymmetric(self, data: bytes, recipient_public_key) -> bytes:
        """Encrypt data using recipient's public key"""
        try:
            encrypted = recipient_public_key.encrypt(
                data,
                padding.OAEP(
                    mgf=padding.MGF1(algorithm=hashes.SHA256()),
                    algorithm=hashes.SHA256(),
                    label=None
                )
            )
            return encrypted
        except Exception as e:
            logger.error(f"Asymmetric encryption failed: {e}")
            raise

    def decrypt_asymmetric(self, encrypted_data: bytes) -> bytes:
        """Decrypt data using our private key"""
        try:
            decrypted = self.private_key.decrypt(
                encrypted_data,
                padding.OAEP(
                    mgf=padding.MGF1(algorithm=hashes.SHA256()),
                    algorithm=hashes.SHA256(),
                    label=None
                )
            )
            return decrypted
        except Exception as e:
            logger.error(f"Asymmetric decryption failed: {e}")
            raise

    def get_public_key_bytes(self) -> bytes:
        """Export public key in PEM format"""
        return self.public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        )

    def load_public_key(self, key_bytes: bytes):
        """Load a public key from bytes"""
        return serialization.load_pem_public_key(key_bytes)

class SecureMessage:
    """Helper class for encrypting/decrypting messages"""
    
    def __init__(self, encryption_manager: EncryptionManager):
        self.encryption_manager = encryption_manager

    def encrypt_message(self, message: dict, recipient_public_key=None) -> dict:
        """
        Encrypt a message, optionally using recipient's public key
        Returns dict with encrypted data and metadata
        """
        try:
            # Convert message to bytes
            message_bytes = str(message).encode()
            
            if recipient_public_key:
                # Asymmetric encryption
                encrypted = self.encryption_manager.encrypt_asymmetric(
                    message_bytes, 
                    recipient_public_key
                )
                method = "asymmetric"
            else:
                # Symmetric encryption
                encrypted = self.encryption_manager.encrypt_symmetric(message_bytes)
                method = "symmetric"
            
            return {
                "encrypted_data": base64.b64encode(encrypted).decode(),
                "encryption_method": method
            }
            
        except Exception as e:
            logger.error(f"Message encryption failed: {e}")
            raise

    def decrypt_message(self, encrypted_message: dict) -> Optional[dict]:
        """Decrypt a message"""
        try:
            encrypted_data = base64.b64decode(encrypted_message["encrypted_data"])
            method = encrypted_message["encryption_method"]
            
            if method == "asymmetric":
                decrypted = self.encryption_manager.decrypt_asymmetric(encrypted_data)
            else:
                decrypted = self.encryption_manager.decrypt_symmetric(encrypted_data)
                
            # Convert back to dict
            return eval(decrypted.decode())  # Note: In production, use proper serialization
            
        except Exception as e:
            logger.error(f"Message decryption failed: {e}")
            return None 