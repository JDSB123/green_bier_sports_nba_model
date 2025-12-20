"""
Security utilities for NBA v6.0.

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


def validate_premium_features() -> Dict[str, Dict[str, any]]:
    """
    Validate configuration for all premium features.

    Returns detailed status for each premium data source:
    - Required keys for each feature
    - Whether the feature is fully configured
    - What data will be available/missing

    This is meant to be called at startup to give a clear picture
    of what premium data will be available for predictions.
    """
    features = {
        "odds_primary": {
            "name": "The Odds API (Primary Odds)",
            "configured": bool(settings.the_odds_api_key),
            "required": True,
            "keys_needed": ["THE_ODDS_API_KEY"],
            "provides": ["spreads", "totals", "moneylines", "period markets (Q1/1H)"],
        },
        "game_data": {
            "name": "API-Basketball (Team/Game Stats)",
            "configured": bool(settings.api_basketball_key),
            "required": True,
            "keys_needed": ["API_BASKETBALL_KEY"],
            "provides": ["team statistics", "game results", "H2H history", "standings"],
        },
        "betting_splits": {
            "name": "Action Network (Betting Splits)",
            "configured": bool(settings.action_network_username and settings.action_network_password),
            "required": False,
            "keys_needed": ["ACTION_NETWORK_USERNAME", "ACTION_NETWORK_PASSWORD"],
            "provides": ["public betting %", "money %", "RLM detection", "sharp money signals"],
        },
        "odds_backup": {
            "name": "BetsAPI (Backup Odds)",
            "configured": bool(settings.betsapi_key),
            "required": False,
            "keys_needed": ["BETSAPI_KEY"],
            "provides": ["backup odds source", "live odds"],
        },
    }

    # Log summary
    configured_count = sum(1 for f in features.values() if f["configured"])
    total_count = len(features)
    required_missing = [f["name"] for f in features.values() if f["required"] and not f["configured"]]

    if required_missing:
        logger.error(f"MISSING REQUIRED API KEYS: {required_missing}")
    else:
        logger.info(f"Premium data sources: {configured_count}/{total_count} configured")

    optional_missing = [f["name"] for f in features.values() if not f["required"] and not f["configured"]]
    if optional_missing:
        logger.warning(f"Optional features not configured (predictions may be less accurate): {optional_missing}")

    return features
