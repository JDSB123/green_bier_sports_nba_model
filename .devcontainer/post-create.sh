#!/bin/bash
# Post-create script for GitHub Codespaces

set -e

echo "Setting up NBA Prediction System in Codespaces..."

bash scripts/setup_codespace.sh
