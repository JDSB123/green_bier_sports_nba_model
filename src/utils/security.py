"""
Security utilities for NBA v5.0 BETA.

Handles:
- API key validation
- Environment variable validation
- API key masking for logging
- Security configuration
"""
from __future__ import annotations
import os
import re
from typing import List, Dict, Optional
from dataclasses import dataclass

from src.config import settings
from src.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class ValidationResult:
    """Result of security validation."""
    is_valid: bool
    errors: List[str]
    warnings: List[str]


class SecurityError(Exception):
    """Raised when security validation fails."""
    pass


def mask_api_key(key: str, visible_chars: int = 4) -> str:
    """
    Mask an API key for safe logging.
    
    Args:
        key: The API key to mask
        visible_chars: Number of characters to show at the end
    
    Returns:
        Masked key string (e.g., "****1234")
    """
    if not key or len(key) < visible_chars:
        return "****"
    return "*" * (len(key) - visible_chars) + key[-visible_chars:]


def validate_required_api_keys() -> ValidationResult:
    """
    Validate that all required API keys are set and non-empty.
    
    Returns:
        ValidationResult with validation status
    """
    errors = []
    warnings = []
    
    # Required API keys
    required_keys = {
        "THE_ODDS_API_KEY": settings.the_odds_api_key,
        "API_BASKETBALL_KEY": settings.api_basketball_key,
    }
    
    for key_name, key_value in required_keys.items():
        if not key_value or not key_value.strip():
            errors.append(f"{key_name} is not set or is empty")
        elif len(key_value.strip()) < 10:
            warnings.append(f"{key_name} appears to be too short (may be invalid)")
        else:
            logger.debug(f"{key_name}: {mask_api_key(key_value)}")
    
    # Optional API keys (warn if missing but don't fail)
    optional_keys = {
        "ACTION_NETWORK_USERNAME": settings.action_network_username,
        "ACTION_NETWORK_PASSWORD": settings.action_network_password,
        "BETSAPI_KEY": settings.betsapi_key,
    }
    
    for key_name, key_value in optional_keys.items():
        if not key_value or not key_value.strip():
            warnings.append(f"{key_name} is not set (optional, but some features may be unavailable)")
    
    return ValidationResult(
        is_valid=len(errors) == 0,
        errors=errors,
        warnings=warnings
    )


def validate_database_config() -> ValidationResult:
    """
    Validate database configuration.
    
    Returns:
        ValidationResult with validation status
    """
    errors = []
    warnings = []
    
    database_url = os.getenv("DATABASE_URL", "").strip()
    db_password = os.getenv("DB_PASSWORD", "").strip()

    # Database is OPTIONAL in the current single-container production flow.
    # Only validate strictly when the user has configured a database URL.
    if not database_url and not db_password:
        return ValidationResult(is_valid=True, errors=[], warnings=[])

    if database_url and not db_password:
        errors.append("DB_PASSWORD is required when DATABASE_URL is set")
    elif db_password == "nba_dev_password":
        warnings.append("DB_PASSWORD is using default value - change for production")
    elif db_password and len(db_password) < 12:
        warnings.append("DB_PASSWORD is shorter than 12 characters (weak)")
    
    return ValidationResult(
        is_valid=len(errors) == 0,
        errors=errors,
        warnings=warnings
    )


def validate_environment() -> ValidationResult:
    """
    Validate all environment configuration.
    
    Returns:
        ValidationResult with validation status
    """
    api_result = validate_required_api_keys()
    db_result = validate_database_config()
    
    all_errors = api_result.errors + db_result.errors
    all_warnings = api_result.warnings + db_result.warnings
    
    return ValidationResult(
        is_valid=len(all_errors) == 0,
        errors=all_errors,
        warnings=all_warnings
    )


def fail_fast_on_missing_keys() -> None:
    """
    Validate required API keys and fail immediately if missing.
    
    Raises:
        SecurityError: If required API keys are missing
    """
    result = validate_required_api_keys()
    
    if not result.is_valid:
        error_msg = "Security validation failed:\n  " + "\n  ".join(result.errors)
        logger.error(error_msg)
        raise SecurityError(error_msg)
    
    if result.warnings:
        for warning in result.warnings:
            logger.warning(f"Security warning: {warning}")


def sanitize_for_logging(data: Dict[str, any]) -> Dict[str, any]:
    """
    Sanitize dictionary to remove or mask sensitive data.
    
    Args:
        data: Dictionary that may contain sensitive keys
    
    Returns:
        Sanitized dictionary with sensitive values masked
    """
    sensitive_keys = [
        "apiKey", "api_key", "apikey",
        "password", "passwd", "pwd",
        "secret", "token", "auth",
        "THE_ODDS_API_KEY", "API_BASKETBALL_KEY",
        "DB_PASSWORD", "DATABASE_URL",
    ]
    
    sanitized = {}
    for key, value in data.items():
        key_lower = key.lower()
        if any(sensitive in key_lower for sensitive in sensitive_keys):
            if isinstance(value, str):
                sanitized[key] = mask_api_key(value)
            else:
                sanitized[key] = "****"
        else:
            sanitized[key] = value
    
    return sanitized


def get_api_key_status() -> Dict[str, str]:
    """
    Get status of all API keys (masked) for health checks.
    
    Returns:
        Dictionary with key status (masked)
    """
    return {
        "THE_ODDS_API_KEY": "set" if settings.the_odds_api_key else "not_set",
        "API_BASKETBALL_KEY": "set" if settings.api_basketball_key else "not_set",
        "ACTION_NETWORK_USERNAME": "set" if settings.action_network_username else "not_set",
        "BETSAPI_KEY": "set" if settings.betsapi_key else "not_set",
    }
