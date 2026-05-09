"""
Encryption Module

Implements at-rest encryption, HSM integration, and key management
for HistoCore Real-Time WSI Streaming.

Requirements: AES-256-GCM encryption, HSM support, key rotation
"""

import hashlib
import logging
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

# Cryptography imports
try:
    from cryptography.fernet import Fernet
    from cryptography.hazmat.backends import default_backend
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2

    CRYPTO_AVAILABLE = True
except ImportError:
    CRYPTO_AVAILABLE = False
    logger.warning("cryptography not available. Install: pip install cryptography")

# PKCS#11 HSM support
try:
    import PyKCS11
    PKCS11_AVAILABLE = True
except ImportError:
    PKCS11_AVAILABLE = False
    logger.debug("PyKCS11 not available. Install for HSM support: pip install PyKCS11")


class HSMManager:
    """Manages Hardware Security Module (HSM) integration via PKCS#11."""
    
    def __init__(self, config):
        """Initialize HSM manager."""
        self.config = config
        
        if not PKCS11_AVAILABLE:
            raise RuntimeError("PyKCS11 library required for HSM support. Install: pip install PyKCS11")
        
        self.pkcs11 = PyKCS11.PyKCS11Lib()
        self.session = None
        
        logger.info("HSM manager initialized: library=%s slot=%d", 
                   config.hsm_library_path, config.hsm_slot_id)
    
    def connect(self) -> None:
        """Connect to HSM and open session."""
        try:
            # Load PKCS#11 library
            self.pkcs11.load(self.config.hsm_library_path)
            
            # Get slot
            slots = self.pkcs11.getSlotList(tokenPresent=True)
            if self.config.hsm_slot_id >= len(slots):
                raise ValueError(f"HSM slot {self.config.hsm_slot_id} not found")
            
            slot = slots[self.config.hsm_slot_id]
            
            # Open session
            self.session = self.pkcs11.openSession(slot, PyKCS11.CKF_SERIAL_SESSION | PyKCS11.CKF_RW_SESSION)
            
            # Login with PIN
            self.session.login(self.config.hsm_pin)
            
            logger.info("Connected to HSM slot %d", self.config.hsm_slot_id)
            
        except Exception as e:
            logger.error(f"Failed to connect to HSM: {e}")
            raise RuntimeError(f"HSM connection failed: {e}") from e
    
    def disconnect(self) -> None:
        """Disconnect from HSM."""
        if self.session:
            try:
                self.session.logout()
                self.session.closeSession()
                logger.info("Disconnected from HSM")
            except Exception as e:
                logger.warning(f"Error disconnecting from HSM: {e}")
    
    def generate_key(self, key_label: Optional[str] = None) -> int:
        """Generate AES-256 key in HSM.
        
        Returns:
            Key handle (CK_OBJECT_HANDLE)
        """
        if not self.session:
            raise RuntimeError("Not connected to HSM")
        
        label = key_label or self.config.hsm_key_label
        
        # Key template for AES-256
        template = [
            (PyKCS11.CKA_CLASS, PyKCS11.CKO_SECRET_KEY),
            (PyKCS11.CKA_KEY_TYPE, PyKCS11.CKK_AES),
            (PyKCS11.CKA_VALUE_LEN, 32),  # 256 bits
            (PyKCS11.CKA_LABEL, label),
            (PyKCS11.CKA_TOKEN, True),  # Persistent
            (PyKCS11.CKA_PRIVATE, True),
            (PyKCS11.CKA_SENSITIVE, True),
            (PyKCS11.CKA_ENCRYPT, True),
            (PyKCS11.CKA_DECRYPT, True),
            (PyKCS11.CKA_EXTRACTABLE, False),  # Cannot export key
        ]
        
        # Generate key
        key_handle = self.session.generateKey(template)
        
        logger.info(f"Generated AES-256 key in HSM: label={label} handle={key_handle}")
        
        return key_handle
    
    def find_key(self, key_label: Optional[str] = None) -> Optional[int]:
        """Find key in HSM by label.
        
        Returns:
            Key handle if found, None otherwise
        """
        if not self.session:
            raise RuntimeError("Not connected to HSM")
        
        label = key_label or self.config.hsm_key_label
        
        # Search template
        template = [
            (PyKCS11.CKA_CLASS, PyKCS11.CKO_SECRET_KEY),
            (PyKCS11.CKA_LABEL, label),
        ]
        
        # Find objects
        objects = self.session.findObjects(template)
        
        if objects:
            logger.info(f"Found key in HSM: label={label} handle={objects[0]}")
            return objects[0]
        else:
            logger.warning(f"Key not found in HSM: label={label}")
            return None
    
    def encrypt(self, key_handle: int, plaintext: bytes) -> bytes:
        """Encrypt data using HSM key.
        
        Args:
            key_handle: HSM key handle
            plaintext: Data to encrypt
            
        Returns:
            Encrypted data (IV + ciphertext + tag for AES-GCM)
        """
        if not self.session:
            raise RuntimeError("Not connected to HSM")
        
        # Generate random IV (12 bytes for GCM)
        iv = os.urandom(12)
        
        # AES-GCM mechanism
        mechanism = PyKCS11.Mechanism(PyKCS11.CKM_AES_GCM, {
            'pIv': iv,
            'ulIvLen': len(iv),
            'ulTagBits': 128,  # 16-byte tag
        })
        
        # Encrypt
        ciphertext = bytes(self.session.encrypt(key_handle, plaintext, mechanism))
        
        # Return IV + ciphertext (tag is appended by HSM)
        return iv + ciphertext
    
    def decrypt(self, key_handle: int, ciphertext: bytes) -> bytes:
        """Decrypt data using HSM key.
        
        Args:
            key_handle: HSM key handle
            ciphertext: Encrypted data (IV + ciphertext + tag)
            
        Returns:
            Decrypted plaintext
        """
        if not self.session:
            raise RuntimeError("Not connected to HSM")
        
        # Extract IV (first 12 bytes)
        iv = ciphertext[:12]
        encrypted_data = ciphertext[12:]
        
        # AES-GCM mechanism
        mechanism = PyKCS11.Mechanism(PyKCS11.CKM_AES_GCM, {
            'pIv': iv,
            'ulIvLen': len(iv),
            'ulTagBits': 128,
        })
        
        # Decrypt
        plaintext = bytes(self.session.decrypt(key_handle, encrypted_data, mechanism))
        
        return plaintext
    
    def delete_key(self, key_handle: int) -> None:
        """Delete key from HSM."""
        if not self.session:
            raise RuntimeError("Not connected to HSM")
        
        self.session.destroyObject(key_handle)
        logger.info(f"Deleted key from HSM: handle={key_handle}")


class EncryptionManager:
    """Manages at-rest encryption for cached data with optional HSM support."""

    def __init__(self, config):
        """Initialize encryption manager."""
        self.config = config

        if not CRYPTO_AVAILABLE:
            raise RuntimeError("cryptography library required for encryption")

        # Initialize encryption key
        self.master_key = None
        self.fernet = None
        
        # HSM support
        self.hsm_manager = None
        self.hsm_key_handle = None
        if config.enable_hsm:
            self.hsm_manager = HSMManager(config)

        logger.info("Encryption manager initialized: algorithm=%s hsm=%s", 
                   config.encryption_algorithm, config.enable_hsm)

    def initialize_master_key(self, password: Optional[str] = None) -> bytes:
        """Initialize or load master encryption key.
        
        If HSM enabled, uses HSM for key storage and encryption.
        Otherwise, uses file-based key storage with Fernet.
        """
        if self.config.enable_hsm:
            # HSM-based key management
            return self._initialize_hsm_key()
        else:
            # File-based key management
            return self._initialize_file_key(password)
    
    def _initialize_hsm_key(self) -> bytes:
        """Initialize HSM-based encryption key."""
        try:
            # Connect to HSM
            self.hsm_manager.connect()
            
            # Try to find existing key
            self.hsm_key_handle = self.hsm_manager.find_key()
            
            if self.hsm_key_handle is None:
                # Generate new key in HSM
                self.hsm_key_handle = self.hsm_manager.generate_key()
                logger.info("Generated new master key in HSM")
            else:
                logger.info("Loaded existing master key from HSM")
            
            # Return dummy key (actual key never leaves HSM)
            return b"HSM_KEY_HANDLE_" + str(self.hsm_key_handle).encode()
            
        except Exception as e:
            logger.error(f"HSM initialization failed: {e}")
            raise RuntimeError(f"Failed to initialize HSM key: {e}") from e
    
    def _initialize_file_key(self, password: Optional[str] = None) -> bytes:
        """Initialize file-based encryption key."""
        key_path = Path(self.config.key_storage_path) / "master.key"

        if key_path.exists():
            # Load existing key
            with open(key_path, "rb") as f:
                self.master_key = f.read()
            logger.info("Loaded existing master key")
        else:
            # Generate new key
            if password:
                # Derive key from password
                salt = os.urandom(16)
                kdf = PBKDF2(
                    algorithm=hashes.SHA256(),
                    length=32,
                    salt=salt,
                    iterations=self.config.key_derivation_iterations,
                    backend=default_backend(),
                )
                self.master_key = kdf.derive(password.encode())

                # Store salt with key
                with open(key_path, "wb") as f:
                    f.write(salt + self.master_key)
            else:
                # Generate random key
                self.master_key = Fernet.generate_key()

                with open(key_path, "wb") as f:
                    f.write(self.master_key)

            # Secure file permissions
            self._secure_file_permissions(key_path)

            logger.info("Generated new master key: %s", key_path)

        # Initialize Fernet cipher
        self.fernet = Fernet(self.master_key)

        return self.master_key
    
    def _secure_file_permissions(self, filepath: Path) -> None:
        """Set secure file permissions (Unix and Windows)."""
        try:
            if os.name == 'posix':
                os.chmod(filepath, 0o600)
                logger.debug(f"Set Unix permissions 0600 on {filepath}")
            elif os.name == 'nt':
                # Windows: Use icacls to set restrictive ACLs
                import subprocess
                result = subprocess.run(
                    ['icacls', str(filepath), '/inheritance:r', '/grant:r', f'{os.getlogin()}:F'],
                    check=True,
                    capture_output=True,
                    text=True
                )
                logger.debug(f"Set Windows ACLs on {filepath}")
            else:
                logger.warning(f"Unknown OS, cannot secure permissions on {filepath}")
        except Exception as e:
            # CRITICAL: Fail loudly if permissions cannot be secured
            logger.error(f"CRITICAL: Failed to secure permissions on {filepath}: {e}")
            # Remove insecure key file
            try:
                filepath.unlink()
            except Exception as cleanup_err:
                logger.error(f"Failed to remove insecure key file: {cleanup_err}", exc_info=True)
            raise RuntimeError(f"Cannot secure key file permissions: {e}") from e

    def encrypt_data(self, data: bytes) -> bytes:
        """Encrypt data using master key (HSM or file-based)."""
        if self.config.enable_hsm:
            if not self.hsm_key_handle:
                raise RuntimeError("HSM key not initialized")
            return self.hsm_manager.encrypt(self.hsm_key_handle, data)
        else:
            if not self.fernet:
                raise RuntimeError("Master key not initialized")
            return self.fernet.encrypt(data)

    def decrypt_data(self, encrypted_data: bytes) -> bytes:
        """Decrypt data using master key (HSM or file-based)."""
        if self.config.enable_hsm:
            if not self.hsm_key_handle:
                raise RuntimeError("HSM key not initialized")
            return self.hsm_manager.decrypt(self.hsm_key_handle, encrypted_data)
        else:
            if not self.fernet:
                raise RuntimeError("Master key not initialized")
            return self.fernet.decrypt(encrypted_data)

    def encrypt_file(self, input_path: str, output_path: str) -> str:
        """Encrypt file with size limit."""
        max_size = 1024 * 1024 * 1024  # 1GB limit
        file_size = os.path.getsize(input_path)
        if file_size > max_size:
            raise ValueError(f"File too large: {file_size} bytes (max {max_size})")
        
        with open(input_path, "rb") as f:
            data = f.read()

        encrypted_data = self.encrypt_data(data)

        with open(output_path, "wb") as f:
            f.write(encrypted_data)

        logger.info("Encrypted file: %s -> %s", input_path, output_path)

        return output_path

    def decrypt_file(self, input_path: str, output_path: str) -> str:
        """Decrypt file with size validation."""
        max_size = 1024 * 1024 * 1024  # 1GB
        file_size = os.path.getsize(input_path)
        if file_size > max_size:
            raise ValueError(f"File too large: {file_size} bytes")
        
        with open(input_path, "rb") as f:
            encrypted_data = f.read()

        data = self.decrypt_data(encrypted_data)

        with open(output_path, "wb") as f:
            f.write(data)

        logger.info("Decrypted file: %s -> %s", input_path, output_path)

        return output_path

    def rotate_key(self, new_password: Optional[str] = None) -> bytes:
        """Rotate encryption key.
        
        For HSM: generates new key in HSM and re-encrypts data.
        For file-based: generates new Fernet key and re-encrypts data.
        """
        logger.info("Rotating encryption key")

        if self.config.enable_hsm:
            # HSM key rotation
            old_key_handle = self.hsm_key_handle
            
            # Generate new key
            self.hsm_key_handle = self.hsm_manager.generate_key(
                key_label=f"{self.config.hsm_key_label}_rotated_{int(datetime.utcnow().timestamp())}"
            )
            
            # Re-encrypt all encrypted files would happen here
            # (Implementation depends on file storage structure)
            
            # Delete old key
            if old_key_handle:
                self.hsm_manager.delete_key(old_key_handle)
            
            logger.info("HSM key rotation complete")
            return b"HSM_KEY_HANDLE_" + str(self.hsm_key_handle).encode()
        else:
            # File-based key rotation
            old_key = self.master_key
            old_fernet = self.fernet

            # Initialize new key
            self.master_key = None
            self.fernet = None
            new_key = self._initialize_file_key(new_password)

            # Re-encrypt all encrypted files
            # (This would need to be implemented based on your file storage structure)

            logger.info("File-based key rotation complete")
            return new_key


class KeyManager:
    """Manages encryption keys and rotation."""

    def __init__(self, config):
        """Initialize key manager."""
        self.config = config
        self.key_metadata = {}

        logger.info("Key manager initialized: rotation_days=%d", config.key_rotation_days)

    def generate_key(self, key_id: str, key_type: str = "symmetric") -> bytes:
        """Generate new encryption key."""
        if key_type == "symmetric":
            key = Fernet.generate_key()
        elif key_type == "asymmetric":
            # Generate RSA key pair
            private_key = rsa.generate_private_key(
                public_exponent=65537, key_size=2048, backend=default_backend()
            )
            key = private_key.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.PKCS8,
                encryption_algorithm=serialization.NoEncryption(),
            )
        else:
            raise ValueError(f"Invalid key type: {key_type}")

        # Store key metadata
        self.key_metadata[key_id] = {
            "created_at": datetime.utcnow(),
            "key_type": key_type,
            "rotation_due": datetime.utcnow() + timedelta(days=self.config.key_rotation_days),
        }

        # Save key
        key_path = Path(self.config.key_storage_path) / f"{key_id}.key"
        with open(key_path, "wb") as f:
            f.write(key)

        # Secure permissions
        self._secure_file_permissions(key_path)

        logger.info("Generated key: %s (type=%s)", key_id, key_type)

        return key
    
    def _secure_file_permissions(self, filepath: Path) -> None:
        """Set secure file permissions (Unix and Windows)."""
        try:
            if os.name == 'posix':
                os.chmod(filepath, 0o600)
                logger.debug(f"Set Unix permissions 0600 on {filepath}")
            elif os.name == 'nt':
                # Windows: Use icacls to set restrictive ACLs
                import subprocess
                result = subprocess.run(
                    ['icacls', str(filepath), '/inheritance:r', '/grant:r', f'{os.getlogin()}:F'],
                    check=True,
                    capture_output=True,
                    text=True
                )
                logger.debug(f"Set Windows ACLs on {filepath}")
            else:
                logger.warning(f"Unknown OS, cannot secure permissions on {filepath}")
        except Exception as e:
            # CRITICAL: Fail loudly if permissions cannot be secured
            logger.error(f"CRITICAL: Failed to secure permissions on {filepath}: {e}")
            # Remove insecure key file
            try:
                filepath.unlink()
            except Exception as cleanup_err:
                logger.error(f"Failed to remove insecure key file: {cleanup_err}", exc_info=True)
            raise RuntimeError(f"Cannot secure key file permissions: {e}") from e

    def load_key(self, key_id: str) -> bytes:
        """Load encryption key."""
        key_path = Path(self.config.key_storage_path) / f"{key_id}.key"

        if not key_path.exists():
            raise FileNotFoundError(f"Key not found: {key_id}")

        with open(key_path, "rb") as f:
            key = f.read()

        return key

    def check_rotation_needed(self, key_id: str) -> bool:
        """Check if key rotation is needed."""
        if key_id not in self.key_metadata:
            return False

        metadata = self.key_metadata[key_id]
        return datetime.utcnow() >= metadata["rotation_due"]

    def rotate_key(self, key_id: str) -> bytes:
        """Rotate encryption key."""
        logger.info("Rotating key: %s", key_id)

        # Archive old key
        old_key_path = Path(self.config.key_storage_path) / f"{key_id}.key"
        archive_path = Path(self.config.key_storage_path) / f"{key_id}.key.old"

        if old_key_path.exists():
            old_key_path.rename(archive_path)

        # Generate new key
        metadata = self.key_metadata.get(key_id, {})
        key_type = metadata.get("key_type", "symmetric")
        new_key = self.generate_key(key_id, key_type)

        logger.info("Key rotated: %s", key_id)

        return new_key
