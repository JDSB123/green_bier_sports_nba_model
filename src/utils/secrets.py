"""
Docker Secrets Utility for NBA v5.0 BETA

Reads secrets from Docker secrets (Swarm mode) or secret files (Compose mode).
Falls back to environment variables for backward compatibility.
"""
from __future__ import annotations
import os
from pathlib import Path
from typing import Optional

# Docker secrets are mounted at /run/secrets/ in containers
DOCKER_SECRETS_DIR = Path("/run/secrets")

# For Docker Compose, secrets can be mounted at a custom path
# Default to /run/secrets, but also check ./secrets for local development
COMPOSE_SECRETS_DIR = Path("/run/secrets")
LOCAL_SECRETS_DIR = Path(__file__).resolve().parent.parent.parent / "secrets"


def read_secret(secret_name: str, default: str = "") -> str:
    """
    Read a secret from Docker secrets or secret files.
    
    Priority order:
    1. Docker Swarm secrets (/run/secrets/{secret_name})
    2. Docker Compose secrets (/run/secrets/{secret_name})
    3. Local secret files (./secrets/{secret_name})
    4. Environment variable ({secret_name})
    5. Default value
    
    Args:
        secret_name: Name of the secret
        default: Default value if secret not found
    
    Returns:
        Secret value as string
    """
    # Try Docker secrets (Swarm/Compose)
    docker_secret_path = DOCKER_SECRETS_DIR / secret_name
    if docker_secret_path.exists() and docker_secret_path.is_file():
        try:
            value = docker_secret_path.read_text(encoding="utf-8").strip()
            if value:
                return value
        except Exception:
            pass
    
    # Try local secret files (for development)
    local_secret_path = LOCAL_SECRETS_DIR / secret_name
    if local_secret_path.exists() and local_secret_path.is_file():
        try:
            value = local_secret_path.read_text(encoding="utf-8").strip()
            if value:
                return value
        except Exception:
            pass
    
    # Fall back to environment variable
    env_value = os.getenv(secret_name, default)
    if env_value:
        return env_value
    
    return default


def read_secret_or_env(secret_name: str, env_name: Optional[str] = None, default: str = "") -> str:
    """
    Read a secret, with optional environment variable fallback.
    
    Args:
        secret_name: Name of the secret file
        env_name: Name of environment variable (defaults to secret_name)
        default: Default value if neither found
    
    Returns:
        Secret value as string
    """
    if env_name is None:
        env_name = secret_name
    
    # Try secret first
    secret_value = read_secret(secret_name, "")
    if secret_value:
        return secret_value
    
    # Fall back to environment variable
    return os.getenv(env_name, default)


def secret_exists(secret_name: str) -> bool:
    """
    Check if a secret exists.
    
    Args:
        secret_name: Name of the secret
    
    Returns:
        True if secret exists, False otherwise
    """
    # Check Docker secrets
    if (DOCKER_SECRETS_DIR / secret_name).exists():
        return True
    
    # Check local secret files
    if (LOCAL_SECRETS_DIR / secret_name).exists():
        return True
    
    # Check environment variable
    if os.getenv(secret_name):
        return True
    
    return False


def get_secret_source(secret_name: str) -> str:
    """
    Get the source of a secret (for debugging).
    
    Args:
        secret_name: Name of the secret
    
    Returns:
        Source description ("docker-secret", "local-file", "environment", or "not-found")
    """
    if (DOCKER_SECRETS_DIR / secret_name).exists():
        return "docker-secret"
    
    if (LOCAL_SECRETS_DIR / secret_name).exists():
        return "local-file"
    
    if os.getenv(secret_name):
        return "environment"
    
    return "not-found"
