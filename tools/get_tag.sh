#!/usr/bin/env bash
set -euo pipefail

# Check if this run was triggered by a tag
if [[ "${GITHUB_REF:-}" == refs/tags/* ]]; then
    TAG_NAME="${GITHUB_REF#refs/tags/}"
    echo "Detected tag push: $TAG_NAME"
    # For GitHub Actions output
    echo "version=$TAG_NAME" >> "${GITHUB_OUTPUT:-/dev/stdout}"
    exit 0
fi

# Detected version from Git
TAG_NAME=$(git describe --tags --abbrev=0)
echo "version=$TAG_NAME" >> "${GITHUB_OUTPUT:-/dev/stdout}"
