"""
API Authentication middleware.

Provides API key authentication to protect endpoints.
"""
from __future__ import annotations
import os
from typing import Optional
from fastapi import Request, HTTPException, Security
from fastapi.security import APIKeyHeader, APIKeyQuery
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response

from src.utils.logging import get_logger

logger = get_logger(__name__)

# API Key header name
API_KEY_HEADER_NAME = "X-API-Key"
API_KEY_QUERY_NAME = "api_key"

# Get API key from environment (for service-to-service auth)
SERVICE_API_KEY = os.getenv("SERVICE_API_KEY", "")

# Optional: Allow disabling auth for development
REQUIRE_AUTH = os.getenv("REQUIRE_API_AUTH", "true").lower() == "true"


# FastAPI security schemes
api_key_header = APIKeyHeader(name=API_KEY_HEADER_NAME, auto_error=False)
api_key_query = APIKeyQuery(name=API_KEY_QUERY_NAME, auto_error=False)


def get_api_key_from_request(request: Request) -> Optional[str]:
    """
    Extract API key from request (header or query param).
    
    Args:
        request: FastAPI request object
    
    Returns:
        API key if found, None otherwise
    """
    # Try header first
    api_key = request.headers.get(API_KEY_HEADER_NAME)
    if api_key:
        return api_key
    
    # Try query parameter
    api_key = request.query_params.get(API_KEY_QUERY_NAME)
    if api_key:
        return api_key
    
    return None


def verify_api_key(api_key: Optional[str]) -> bool:
    """
    Verify API key against configured service key.
    
    Args:
        api_key: API key to verify
    
    Returns:
        True if valid, False otherwise
    """
    if not REQUIRE_AUTH:
        return True  # Auth disabled for development
    
    if not SERVICE_API_KEY:
        raise ValueError(
            "REQUIRE_API_AUTH=true but SERVICE_API_KEY is not set. "
            "Either set SERVICE_API_KEY or disable auth with REQUIRE_API_AUTH=false"
        )
    
    if not api_key:
        return False
    
    return api_key == SERVICE_API_KEY


async def get_api_key(
    header_key: Optional[str] = Security(api_key_header),
    query_key: Optional[str] = Security(api_key_query),
) -> str:
    """
    FastAPI dependency to extract and validate API key.
    
    Args:
        header_key: API key from header
        query_key: API key from query parameter
    
    Returns:
        Validated API key
    
    Raises:
        HTTPException: If API key is missing or invalid
    """
    api_key = header_key or query_key
    
    if not api_key:
        raise HTTPException(
            status_code=401,
            detail="API key required. Provide via X-API-Key header or api_key query parameter."
        )
    
    if not verify_api_key(api_key):
        raise HTTPException(
            status_code=403,
            detail="Invalid API key"
        )
    
    return api_key


class APIKeyMiddleware(BaseHTTPMiddleware):
    """
    Middleware to optionally enforce API key authentication.
    
    Can be configured to require auth on all endpoints or specific paths.
    """
    
    def __init__(self, app, require_auth: bool = REQUIRE_AUTH, exempt_paths: Optional[list] = None):
        super().__init__(app)
        self.require_auth = require_auth
        self.exempt_paths = exempt_paths or ["/health", "/metrics", "/docs", "/openapi.json", "/redoc"]
    
    async def dispatch(self, request: Request, call_next):
        # Skip auth for exempt paths
        if any(request.url.path.startswith(path) for path in self.exempt_paths):
            return await call_next(request)
        
        # Skip auth if disabled
        if not self.require_auth:
            return await call_next(request)
        
        # Verify API key
        api_key = get_api_key_from_request(request)
        if not verify_api_key(api_key):
            return Response(
                content='{"detail": "API key required"}',
                status_code=401,
                media_type="application/json"
            )
        
        return await call_next(request)
