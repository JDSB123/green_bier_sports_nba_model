"""
Docker Secrets Utility

Reads secrets from multiple sources in priority order:
1. Environment variables (for Azure Container Apps, GitHub Actions)
2. Docker secrets at /run/secrets/ (for Docker Compose/Swarm)
3. Local secrets directory ./secrets/ (for local development)

This ensures the same code works in all environments.
"""
from __future__ import annotations
import os
from pathlib import Path
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class SecretNotFoundError(Exception):
    """Raised when a required secret is not found."""
    pass


# Standard locations for Docker secrets
DOCKER_SECRETS_DIR = Path("/run/secrets")
LOCAL_SECRETS_DIR = Path("secrets")  # Relative to working directory


def read_secret(secret_name: str, required: bool = True) -> Optional[str]:
    """
    Read a secret from multiple sources in priority order.

    Priority:
    1. Environment variable (e.g., THE_ODDS_API_KEY)
    2. Docker secrets file (/run/secrets/THE_ODDS_API_KEY)
    3. Local secrets file (./secrets/THE_ODDS_API_KEY)

    Args:
        secret_name: Name of the secret (e.g., "THE_ODDS_API_KEY")
        required: If True, raises SecretNotFoundError when not found

    Returns:
        Secret value as string, or None if not found and not required

    Raises:
        SecretNotFoundError: If required=True and secret not found anywhere
    """
    # 1. Try environment variable first (Azure, CI/CD, explicit config)
    value = os.getenv(secret_name)
    if value and value.strip():
        logger.debug(f"Secret {secret_name}: found in environment variable")
        return value.strip()

    # 2. Try Docker secrets directory (/run/secrets/)
    docker_secret_path = DOCKER_SECRETS_DIR / secret_name
    if docker_secret_path.exists():
        try:
            value = docker_secret_path.read_text().strip()
            if value:
                logger.debug(f"Secret {secret_name}: found in Docker secrets")
                return value
        except Exception as e:
            logger.warning(f"Could not read Docker secret {docker_secret_path}: {e}")

    # 3. Try local secrets directory (./secrets/)
    local_secret_path = LOCAL_SECRETS_DIR / secret_name
    if local_secret_path.exists():
        try:
            value = local_secret_path.read_text().strip()
            if value:
                logger.debug(f"Secret {secret_name}: found in local secrets")
                return value
        except Exception as e:
            logger.warning(f"Could not read local secret {local_secret_path}: {e}")

    # 4. Also check /app/secrets/ (baked into container)
    app_secret_path = Path("/app/secrets") / secret_name
    if app_secret_path.exists():
        try:
            value = app_secret_path.read_text().strip()
            if value:
                logger.debug(f"Secret {secret_name}: found in app secrets")
                return value
        except Exception as e:
            logger.warning(f"Could not read app secret {app_secret_path}: {e}")

    # Not found
    if required:
        locations_checked = [
            f"Environment variable: {secret_name}",
            f"Docker secrets: {docker_secret_path}",
            f"Local secrets: {local_secret_path}",
            f"App secrets: {app_secret_path}",
        ]
        raise SecretNotFoundError(
            f"Required secret '{secret_name}' not found.\n"
            f"Checked locations:\n  - " + "\n  - ".join(locations_checked) + "\n\n"
            f"To fix:\n"
            f"  1. Set environment variable: export {secret_name}=your_key\n"
            f"  2. Or create file: echo 'your_key' > secrets/{secret_name}\n"
            f"  3. Or for Docker: mount secrets volume to /run/secrets/"
        )

    return None


def read_secret_strict(secret_name: str) -> str:
    """
    STRICT: Read a required secret from multiple sources.

    This is the main function used by src/config.py for required API keys.

    Args:
        secret_name: Name of the secret

    Returns:
        Secret value as string

    Raises:
        SecretNotFoundError: If not found in any location
    """
    result = read_secret(secret_name, required=True)
    # read_secret with required=True always returns a string or raises
    assert result is not None
    return result


def read_secret_optional(secret_name: str) -> Optional[str]:
    """
    Read an optional secret - returns None if not found.

    Args:
        secret_name: Name of the secret

    Returns:
        Secret value as string, or None if not found
    """
    return read_secret(secret_name, required=False)


def get_secrets_status() -> dict:
    """
    Get status of all known secrets for diagnostics.

    Returns:
        Dictionary with secret names and their source/status
    """
    secrets_to_check = [
        "THE_ODDS_API_KEY",
        "API_BASKETBALL_KEY",
        "BETSAPI_KEY",
        "ACTION_NETWORK_USERNAME",
        "ACTION_NETWORK_PASSWORD",
        "TEAMS_WEBHOOK_URL",
        "SERVICE_API_KEY",
    ]

    status = {}
    for secret_name in secrets_to_check:
        # Check each location
        if os.getenv(secret_name):
            status[secret_name] = "set (env)"
        elif (DOCKER_SECRETS_DIR / secret_name).exists():
            status[secret_name] = "set (docker)"
        elif (LOCAL_SECRETS_DIR / secret_name).exists():
            status[secret_name] = "set (local)"
        elif (Path("/app/secrets") / secret_name).exists():
            status[secret_name] = "set (app)"
        else:
            status[secret_name] = "not_set"

    return status
