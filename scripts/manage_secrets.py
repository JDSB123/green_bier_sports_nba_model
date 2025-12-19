#!/usr/bin/env python3
"""
Docker Secrets Management Script

Manages Docker secrets for NBA v5.0 BETA:
- Creates secret files from .env
- Validates secrets
- Lists secrets
"""
from __future__ import annotations
import argparse
import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SECRETS_DIR = PROJECT_ROOT / "secrets"
ENV_FILE = PROJECT_ROOT / ".env"

# Required secrets
REQUIRED_SECRETS = [
    "THE_ODDS_API_KEY",
    "API_BASKETBALL_KEY",
]

# Optional secrets
OPTIONAL_SECRETS = [
    "SERVICE_API_KEY",
    "ACTION_NETWORK_USERNAME",
    "ACTION_NETWORK_PASSWORD",
    "BETSAPI_KEY",
    "KAGGLE_API_TOKEN",
]


def load_env_file() -> dict[str, str]:
    """Load environment variables from .env file."""
    env_vars = {}
    if not ENV_FILE.exists():
        return env_vars
    
    with open(ENV_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                key, value = line.split("=", 1)
                env_vars[key.strip()] = value.strip().strip('"').strip("'")
    
    return env_vars


def create_secret_file(name: str, value: str) -> None:
    """Create a secret file."""
    SECRETS_DIR.mkdir(exist_ok=True)
    secret_file = SECRETS_DIR / name
    
    # Set restrictive permissions (Unix-like systems)
    secret_file.write_text(value, encoding="utf-8")
    
    # Try to set permissions (Unix only)
    if sys.platform != "win32":
        try:
            os.chmod(secret_file, 0o600)  # rw------- 
        except Exception:
            pass
    
    print(f"✅ Created secret file: {secret_file}")


def create_from_env() -> None:
    """Create secret files from .env file."""
    print("📝 Creating secrets from .env file...")
    
    if not ENV_FILE.exists():
        print(f"❌ .env file not found: {ENV_FILE}")
        print("   Please create .env file first or use 'create' command")
        sys.exit(1)
    
    env_vars = load_env_file()
    
    created = 0
    missing = []
    
    # Create required secrets
    for secret_name in REQUIRED_SECRETS:
        if secret_name in env_vars and env_vars[secret_name]:
            create_secret_file(secret_name, env_vars[secret_name])
            created += 1
        else:
            missing.append(secret_name)
    
    # Create optional secrets
    for secret_name in OPTIONAL_SECRETS:
        if secret_name in env_vars and env_vars[secret_name]:
            create_secret_file(secret_name, env_vars[secret_name])
            created += 1
    
    print(f"\n✅ Created {created} secret file(s)")
    
    if missing:
        print(f"\n⚠️  Missing required secrets: {', '.join(missing)}")
        print("   These will need to be created manually")


def list_secrets() -> None:
    """List all secret files."""
    print("📋 Secret Files:")
    print()
    
    if not SECRETS_DIR.exists():
        print("  No secrets directory found")
        return
    
    required_found = []
    optional_found = []
    other = []
    
    for secret_file in sorted(SECRETS_DIR.glob("*")):
        if secret_file.is_file() and not secret_file.name.startswith("."):
            name = secret_file.name
            if name.endswith(".example"):
                continue
            
            size = secret_file.stat().st_size
            if name in REQUIRED_SECRETS:
                required_found.append((name, size))
            elif name in OPTIONAL_SECRETS:
                optional_found.append((name, size))
            else:
                other.append((name, size))
    
    if required_found:
        print("  Required Secrets:")
        for name, size in required_found:
            print(f"    ✅ {name} ({size} bytes)")
    
    if optional_found:
        print("\n  Optional Secrets:")
        for name, size in optional_found:
            print(f"    ✅ {name} ({size} bytes)")
    
    if other:
        print("\n  Other Files:")
        for name, size in other:
            print(f"    ⚠️  {name} ({size} bytes)")
    
    # Check for missing required secrets
    missing = [s for s in REQUIRED_SECRETS if not (SECRETS_DIR / s).exists()]
    if missing:
        print("\n  Missing Required Secrets:")
        for name in missing:
            print(f"    ❌ {name}")


def validate_secrets() -> None:
    """Validate that all required secrets exist."""
    print("🔍 Validating secrets...")
    print()
    
    if not SECRETS_DIR.exists():
        print("❌ Secrets directory not found")
        sys.exit(1)
    
    all_valid = True
    
    for secret_name in REQUIRED_SECRETS:
        secret_file = SECRETS_DIR / secret_name
        if secret_file.exists():
            value = secret_file.read_text(encoding="utf-8").strip()
            if value:
                print(f"✅ {secret_name}: Set ({len(value)} chars)")
            else:
                print(f"❌ {secret_name}: File exists but is empty")
                all_valid = False
        else:
            print(f"❌ {secret_name}: Missing")
            all_valid = False
    
    if all_valid:
        print("\n✅ All required secrets are valid")
    else:
        print("\n❌ Some required secrets are missing or invalid")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Manage Docker secrets for NBA v5.0 BETA",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/manage_secrets.py create-from-env    # Create secrets from .env
  python scripts/manage_secrets.py list                # List secret files
  python scripts/manage_secrets.py validate            # Validate secrets
        """
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Create from env
    subparsers.add_parser(
        "create-from-env",
        help="Create secret files from .env file"
    )
    
    # List secrets
    subparsers.add_parser(
        "list",
        help="List all secret files"
    )
    
    # Validate secrets
    subparsers.add_parser(
        "validate",
        help="Validate that all required secrets exist"
    )
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    if args.command == "create-from-env":
        create_from_env()
    elif args.command == "list":
        list_secrets()
    elif args.command == "validate":
        validate_secrets()
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
