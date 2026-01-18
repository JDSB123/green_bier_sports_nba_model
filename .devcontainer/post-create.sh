#!/bin/bash
# Post-create script for GitHub Codespaces

set -e

echo "Setting up NBA Prediction System in Codespaces..."

USE_SYSTEM_PYTHON=1 bash scripts/setup_codespace.sh
