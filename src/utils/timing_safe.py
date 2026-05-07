"""
Timing Attack Protection

Provides constant-time comparison functions to prevent timing attacks.
"""

import hmac
import secrets
from typing import Union


def constant_time_compare(a: Union[str, bytes], b: Union[str, bytes]) -> bool:
    """Compare strings/bytes in constant time.
    
    Prevents timing attacks when comparing secrets.
    
    Args:
        a: First value
        b: Second value
        
    Returns:
        True if equal
    """
    # Convert to bytes if needed
    if isinstance(a, str):
        a = a.encode('utf-8')
    if isinstance(b, str):
        b = b.encode('utf-8')
    
    return secrets.compare_digest(a, b)


def constant_time_compare_hmac(a: Union[str, bytes], b: Union[str, bytes], key: bytes) -> bool:
    """Compare using HMAC for additional security.
    
    Args:
        a: First value
        b: Second value
        key: HMAC key
        
    Returns:
        True if equal
    """
    if isinstance(a, str):
        a = a.encode('utf-8')
    if isinstance(b, str):
        b = b.encode('utf-8')
    
    hmac_a = hmac.new(key, a, 'sha256').digest()
    hmac_b = hmac.new(key, b, 'sha256').digest()
    
    return secrets.compare_digest(hmac_a, hmac_b)


def verify_signature(data: bytes, signature: bytes, key: bytes) -> bool:
    """Verify HMAC signature in constant time.
    
    Args:
        data: Data that was signed
        signature: Signature to verify
        key: HMAC key
        
    Returns:
        True if signature valid
    """
    expected = hmac.new(key, data, 'sha256').digest()
    return secrets.compare_digest(signature, expected)


def create_signature(data: bytes, key: bytes) -> bytes:
    """Create HMAC signature.
    
    Args:
        data: Data to sign
        key: HMAC key
        
    Returns:
        Signature bytes
    """
    return hmac.new(key, data, 'sha256').digest()
