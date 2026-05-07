"""
Cryptographically secure random number generation.

Use this module for security-sensitive operations like:
- Token generation
- Session IDs
- Cryptographic keys
- Security challenges

DO NOT use for:
- Machine learning (use numpy.random with fixed seed)
- Simulations (use random module with seed)
"""

import secrets
import string
from typing import List


def generate_token(length: int = 32) -> str:
    """Generate cryptographically secure random token.
    
    Args:
        length: Token length in bytes (default 32 = 256 bits)
        
    Returns:
        URL-safe base64 encoded token
    """
    return secrets.token_urlsafe(length)


def generate_hex_token(length: int = 32) -> str:
    """Generate cryptographically secure hex token.
    
    Args:
        length: Token length in bytes
        
    Returns:
        Hex encoded token
    """
    return secrets.token_hex(length)


def generate_password(length: int = 16) -> str:
    """Generate cryptographically secure random password.
    
    Args:
        length: Password length
        
    Returns:
        Random password with letters, digits, and punctuation
    """
    alphabet = string.ascii_letters + string.digits + string.punctuation
    return ''.join(secrets.choice(alphabet) for _ in range(length))


def generate_session_id() -> str:
    """Generate cryptographically secure session ID.
    
    Returns:
        256-bit session ID
    """
    return secrets.token_urlsafe(32)


def secure_compare(a: str, b: str) -> bool:
    """Timing-safe string comparison.
    
    Prevents timing attacks when comparing secrets.
    
    Args:
        a: First string
        b: Second string
        
    Returns:
        True if strings are equal
    """
    return secrets.compare_digest(a, b)


def random_choice(choices: List) -> any:
    """Cryptographically secure random choice.
    
    Args:
        choices: List of choices
        
    Returns:
        Random element from choices
    """
    return secrets.choice(choices)


def random_int(min_val: int, max_val: int) -> int:
    """Cryptographically secure random integer.
    
    Args:
        min_val: Minimum value (inclusive)
        max_val: Maximum value (inclusive)
        
    Returns:
        Random integer in range [min_val, max_val]
    """
    return secrets.randbelow(max_val - min_val + 1) + min_val
