#!/bin/bash
# Docker entrypoint script that reads secrets and sets environment variables
# This allows services to use secrets while maintaining backward compatibility

# Read secrets from /run/secrets/ and set as environment variables
# This is a fallback for services that don't support reading secrets directly

if [ -d "/run/secrets" ]; then
    for secret_file in /run/secrets/*; do
        if [ -f "$secret_file" ]; then
            secret_name=$(basename "$secret_file")
            # Only set if not already set (env vars take precedence)
            if [ -z "${!secret_name}" ]; then
                export "$secret_name=$(cat "$secret_file")"
            fi
        fi
    done
fi

# Execute the original command
exec "$@"
