"""
Docker Secrets Utility for NBA v6.0

Reads secrets from Docker secrets (Swarm mode) or secret files (Compose mode).
Falls back to environment variables for backward compatibility.
"""
from __future__ import annotations
import os
from typing import Optional

class SecretNotFoundError(Exception):
    """Raised when a required secret is not found."""
    pass

def read_secret_strict(secret_name: str) -> str:
    """
    STRICT: Read secret ONLY from environment variables.
    No fallbacks, no defaults - raises if not found.
    
    Args:
        secret_name: Name of the secret
        
    Returns:
        Secret value as string
        
    Raises:
        SecretNotFoundError: If not set in env
    """
    value = os.getenv(secret_name)
    if not value or not value.strip():
        raise SecretNotFoundError(
            f"Required secret not found in environment: {secret_name}\n"
            f"Set {secret_name} as an environment variable."
        )
    return value.strip()
