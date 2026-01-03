#!/usr/bin/env python3
"""
Debug secrets reading.
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.utils.secrets import read_secret
from src.config import settings

print("Testing secrets reading...")
print("=" * 40)

# Test direct secrets reading
try:
    username = read_secret("ACTION_NETWORK_USERNAME", required=False)
    password = read_secret("ACTION_NETWORK_PASSWORD", required=False)

    print(f"Direct read - Username: '{username}' (len: {len(username) if username else 0})")
    print(f"Direct read - Password: '{password}' (len: {len(password) if password else 0})")
except Exception as e:
    print(f"Direct read failed: {e}")

print()

# Test config system
print("Testing config system...")
try:
    print(f"Config username: '{settings.action_network_username}' (len: {len(settings.action_network_username) if settings.action_network_username else 0})")
    print(f"Config password: '{settings.action_network_password}' (len: {len(settings.action_network_password) if settings.action_network_password else 0})")
except Exception as e:
    print(f"Config read failed: {e}")

print()

# Check environment variables
print("Environment variables:")
print(f"ACTION_NETWORK_USERNAME: {os.getenv('ACTION_NETWORK_USERNAME', 'NOT SET')}")
print(f"ACTION_NETWORK_PASSWORD: {os.getenv('ACTION_NETWORK_PASSWORD', 'NOT SET')}")

print()

# Check file contents directly
print("File contents:")
try:
    with open("secrets/ACTION_NETWORK_USERNAME", "r") as f:
        file_username = f.read().strip()
    print(f"Username file: '{file_username}' (len: {len(file_username)})")
except Exception as e:
    print(f"Username file read failed: {e}")

try:
    with open("secrets/ACTION_NETWORK_PASSWORD", "r") as f:
        file_password = f.read().strip()
    print(f"Password file: '{file_password}' (len: {len(file_password)})")
except Exception as e:
    print(f"Password file read failed: {e}")